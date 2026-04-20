# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
SSIM/PSNR accuracy test: vLLM-Omni custom transformer vs diffusers transformer.

Loads the diffusers LTX2Pipeline twice:
1. With the **diffusers** transformer (baseline)
2. With the **vLLM-Omni** custom transformer swapped in

Both runs use the exact same pipeline code (same denoising loop, CFG,
scheduler, etc.) -- the only variable is the transformer implementation.
This isolates transformer numerical parity from pipeline-level differences.
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
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 4.0
SEED = 42

# Thresholds: vLLM-Omni transformer should be near-identical to diffusers.
# Other models achieve 0.94-0.97. We target 0.95+ with torch.nn.RMSNorm fix.
SSIM_THRESHOLD = 0.95
PSNR_THRESHOLD = 28.0


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


def _video_to_frames(video_np: np.ndarray) -> list[Image.Image]:
    """Convert numpy video to list of PIL Images."""
    while video_np.ndim > 4:
        video_np = video_np[0]
    if video_np.dtype in (np.float32, np.float64, np.float16):
        video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
    return [Image.fromarray(video_np[t]) for t in range(video_np.shape[0])]


def _extract_frames(result) -> list[Image.Image]:
    """Extract frames from diffusers pipeline output."""
    video = result.frames
    if isinstance(video, np.ndarray):
        return _video_to_frames(video)
    if isinstance(video, list):
        if isinstance(video[0], list):
            return [img.convert("RGB") for img in video[0]]
        if isinstance(video[0], Image.Image):
            return [img.convert("RGB") for img in video]
    raise ValueError(f"Unexpected output type: {type(video)}")


def _run_diffusers_pipeline(model: str, output_dir: Path) -> list[Image.Image]:
    """Generate video using stock diffusers LTX2Pipeline."""
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16, local_files_only=_local_files_only(model))
    pipe = pipe.to("cuda")

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
    frames = _extract_frames(result)
    for i, f in enumerate(frames):
        f.save(output_dir / f"diffusers_frame_{i:04d}.png")

    del pipe, result
    gc.collect()
    torch.cuda.empty_cache()
    return frames


def _run_with_custom_transformer(model: str, output_dir: Path) -> list[Image.Image]:
    """Generate video using diffusers pipeline but with vLLM-Omni's custom transformer.

    This swaps the transformer module while keeping all other pipeline components
    (scheduler, VAE, text encoder, connectors) from diffusers. The denoising loop,
    CFG, and all pipeline-level logic are diffusers' code.
    """
    from diffusers import LTX2Pipeline
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import create_transformer_from_config, load_transformer_config

    # Initialize vLLM context for TP-aware layers
    vllm_config = VllmConfig()
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29503")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1)

    local = _local_files_only(model)

    # Load stock diffusers pipeline
    pipe = LTX2Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16, local_files_only=local)

    # Build our custom transformer
    transformer_config = load_transformer_config(model, "transformer", local)
    our_transformer = create_transformer_from_config(transformer_config)

    # Load weights into our transformer from diffusers' state dict
    diffusers_state = dict(pipe.transformer.named_parameters())

    def _weight_iter():
        for name, param in diffusers_state.items():
            yield name, param.data

    our_transformer.load_weights(_weight_iter())
    our_transformer = our_transformer.to(dtype=torch.bfloat16, device="cuda").eval()

    # Add compatibility shims for diffusers pipeline integration
    our_transformer.dtype = torch.bfloat16
    if not hasattr(our_transformer, "cache_context"):
        from contextlib import nullcontext

        our_transformer.cache_context = lambda name: nullcontext()

    # Swap transformer -- our transformer is already on CUDA
    del pipe.transformer
    pipe.transformer = our_transformer
    # Move remaining pipeline components (VAE, text encoder, etc.) to CUDA
    # without touching the transformer (which lacks diffusers' dtype property)
    for name, component in pipe.components.items():
        if name != "transformer" and hasattr(component, "to"):
            try:
                component.to("cuda")
            except Exception:
                pass

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
    frames = _extract_frames(result)
    for i, f in enumerate(frames):
        f.save(output_dir / f"vllm_omni_frame_{i:04d}.png")

    del pipe, result, our_transformer
    gc.collect()
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
    """Compare video frames and assert SSIM/PSNR meet thresholds."""
    min_frames = min(len(vllm_frames), len(diffusers_frames))
    assert min_frames > 0, "No frames to compare"

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

    print(f"\n{model_name} video similarity ({min_frames} frames):")
    print(f"  SSIM: avg={avg_ssim:.6f}, min={min(ssim_scores):.6f}, threshold>={ssim_threshold:.6f}")
    print(f"  PSNR: avg={avg_psnr:.6f} dB, min={min(psnr_scores):.6f} dB, threshold>={psnr_threshold:.6f} dB")

    assert avg_ssim >= ssim_threshold, f"SSIM below threshold: got {avg_ssim:.6f}, expected >= {ssim_threshold:.6f}."
    assert avg_psnr >= psnr_threshold, f"PSNR below threshold: got {avg_psnr:.6f}, expected >= {psnr_threshold:.6f}."
    return avg_ssim, avg_psnr


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ltx2_3_video_matches_diffusers(accuracy_artifact_root: Path = None, tmp_path: Path = None) -> None:
    """Compare LTX-2.3 video: vLLM-Omni custom transformer vs diffusers transformer.

    Uses diffusers' LTX2Pipeline for both runs, swapping only the transformer
    module to isolate numerical parity of the custom transformer implementation.
    """
    model = _model_name()
    root = accuracy_artifact_root or tmp_path or Path("/tmp/ltx23_accuracy")
    root.mkdir(parents=True, exist_ok=True)
    output_dir = model_output_dir(root, MODEL_ID)

    print(f"\n--- Running diffusers baseline with {model} ---")
    diffusers_frames = _run_diffusers_pipeline(model=model, output_dir=output_dir)
    print(f"Diffusers: {len(diffusers_frames)} frames")

    print("\n--- Running with vLLM-Omni custom transformer ---")
    vllm_frames = _run_with_custom_transformer(model=model, output_dir=output_dir)
    print(f"vLLM-Omni: {len(vllm_frames)} frames")

    _assert_video_similarity(
        model_name=MODEL_ID,
        vllm_frames=vllm_frames,
        diffusers_frames=diffusers_frames,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
