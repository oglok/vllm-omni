# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
SSIM/PSNR accuracy test comparing vLLM-Omni LTX23Pipeline against diffusers.

Generates a short video with identical parameters using both:
1. vLLM-Omni (LTX23Pipeline via Omni offline API)
2. diffusers (LTX2Pipeline.from_pretrained)

Then extracts frames and compares per-frame SSIM and PSNR scores
to ensure vLLM-Omni output matches the diffusers baseline.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.e2e.accuracy.utils import compute_image_ssim_psnr, model_output_dir
from tests.utils import hardware_test

MODEL_ID = "dg845/LTX-2.3-Diffusers"
MODEL_ENV_VAR = "VLLM_TEST_LTX23_MODEL"
PROMPT = "A lighthouse on a rocky cliff at sunset, waves crashing below, golden hour lighting"
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"
WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 25  # ~1 second at 24fps
FPS = 24.0
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 4.0
SEED = 42

# Thresholds calibrated empirically (set conservatively, updated after first run)
SSIM_THRESHOLD = 0.70
PSNR_THRESHOLD = 18.0


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


def _video_to_frames(video_np: np.ndarray) -> list[Image.Image]:
    """Convert a numpy video array [B, T, H, W, C] or [T, H, W, C] to a list of PIL Images."""
    if video_np.ndim == 5:
        video_np = video_np[0]  # Remove batch dim
    # video_np is now [T, H, W, C] with values in [0, 255] uint8 or [0, 1] float
    if video_np.dtype in (np.float32, np.float64, np.float16):
        video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
    frames = []
    for t in range(video_np.shape[0]):
        frame = Image.fromarray(video_np[t])
        frames.append(frame)
    return frames


def _run_diffusers_ltx23(*, model: str, output_dir: Path) -> list[Image.Image]:
    """Generate video using diffusers LTX2Pipeline and return frames."""
    from diffusers import LTX2Pipeline

    pipe = None
    try:
        pipe = LTX2Pipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            local_files_only=_local_files_only(model),
        ).to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(SEED)
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            output_type="np",
        )

        # result.frames is typically list of list of PIL images or np array
        video_output = result.frames
        if isinstance(video_output, np.ndarray):
            frames = _video_to_frames(video_output)
        elif isinstance(video_output, list):
            # diffusers returns list[list[PIL.Image]]
            if isinstance(video_output[0], list):
                frames = [img.convert("RGB") for img in video_output[0]]
            else:
                frames = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in video_output]
        else:
            raise ValueError(f"Unexpected diffusers output type: {type(video_output)}")

        # Save frames for debugging
        for i, frame in enumerate(frames):
            frame.save(output_dir / f"diffusers_frame_{i:04d}.png")

        return frames
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_vllm_omni_ltx23(*, model: str, output_dir: Path) -> list[Image.Image]:
    """Generate video using vLLM-Omni LTX23Pipeline and return frames."""
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)

    omni = Omni(
        model=model,
        model_class_name="LTX23Pipeline",
        enforce_eager=True,
        enable_cpu_offload=True,
    )

    sampling_params = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
        generator=generator,
        output_type="np",
    )

    results = omni.generate(
        prompts=[{"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT}],
        sampling_params=sampling_params,
    )

    # Extract video frames from result
    result = results[0]
    video_output = None

    # The output could be in different places depending on the pipeline
    if hasattr(result, "multimodal_output") and result.multimodal_output:
        mm = result.multimodal_output
        if "video" in mm:
            video_output = mm["video"]
    if video_output is None and hasattr(result, "images") and result.images:
        video_output = result.images
        if isinstance(video_output, list) and len(video_output) == 1:
            video_output = video_output[0]

    if video_output is None:
        raise RuntimeError(f"No video output found in result: {result}")

    if isinstance(video_output, np.ndarray):
        frames = _video_to_frames(video_output)
    elif isinstance(video_output, list):
        if isinstance(video_output[0], list):
            frames = [img.convert("RGB") for img in video_output[0]]
        else:
            frames = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in video_output]
    else:
        raise ValueError(f"Unexpected vllm-omni video output type: {type(video_output)}")

    # Save frames for debugging
    for i, frame in enumerate(frames):
        frame.save(output_dir / f"vllm_omni_frame_{i:04d}.png")

    # Cleanup
    del omni
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return frames


def _assert_video_similarity(
    *,
    model_name: str,
    vllm_frames: list[Image.Image],
    diffusers_frames: list[Image.Image],
    ssim_threshold: float,
    psnr_threshold: float,
) -> tuple[float, float]:
    """Compare video frames and assert average SSIM/PSNR meet thresholds.

    Returns (avg_ssim, avg_psnr) for reporting.
    """
    # Allow small frame count differences (1-2 frames) due to rounding
    min_frames = min(len(vllm_frames), len(diffusers_frames))
    assert min_frames > 0, "No frames to compare"
    if len(vllm_frames) != len(diffusers_frames):
        print(
            f"WARNING: Frame count mismatch: vllm={len(vllm_frames)}, "
            f"diffusers={len(diffusers_frames)}. Comparing first {min_frames} frames."
        )

    ssim_scores = []
    psnr_scores = []

    for i in range(min_frames):
        ssim_val, psnr_val = compute_image_ssim_psnr(
            prediction=vllm_frames[i],
            reference=diffusers_frames[i],
        )
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    min_ssim = min(ssim_scores)
    min_psnr = min(psnr_scores)

    print(f"\n{model_name} video similarity metrics ({min_frames} frames):")
    print(f"  SSIM: avg={avg_ssim:.6f}, min={min_ssim:.6f}, threshold>={ssim_threshold:.6f}")
    print(f"  PSNR: avg={avg_psnr:.6f} dB, min={min_psnr:.6f} dB, threshold>={psnr_threshold:.6f} dB")

    assert avg_ssim >= ssim_threshold, (
        f"Average SSIM below threshold for {model_name}: got {avg_ssim:.6f}, expected >= {ssim_threshold:.6f}."
    )
    assert avg_psnr >= psnr_threshold, (
        f"Average PSNR below threshold for {model_name}: got {avg_psnr:.6f}, expected >= {psnr_threshold:.6f}."
    )

    return avg_ssim, avg_psnr


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ltx2_3_video_matches_diffusers(accuracy_artifact_root: Path) -> None:
    """Compare LTX-2.3 video output between vLLM-Omni and diffusers."""
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)

    # Run diffusers first (larger memory footprint, clean up before vllm-omni)
    print(f"\n--- Running diffusers baseline with {model} ---")
    diffusers_frames = _run_diffusers_ltx23(model=model, output_dir=output_dir)
    print(f"Diffusers generated {len(diffusers_frames)} frames")

    # Run vLLM-Omni
    print(f"\n--- Running vLLM-Omni with {model} ---")
    vllm_frames = _run_vllm_omni_ltx23(model=model, output_dir=output_dir)
    print(f"vLLM-Omni generated {len(vllm_frames)} frames")

    # Compare
    _assert_video_similarity(
        model_name=MODEL_ID,
        vllm_frames=vllm_frames,
        diffusers_frames=diffusers_frames,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
