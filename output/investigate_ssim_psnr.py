#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Investigation script: Why SSIM/PSNR scores are low between vLLM-Omni and diffusers.

Runs 4 experiments:
  1. diffusers vs diffusers (same seed) -- baseline for perfect match
  2. vllm-omni (no CPU offload) vs diffusers -- isolate transformer differences
  3. vllm-omni (with CPU offload) vs diffusers -- reproduce original low scores
  4. vllm-omni (no CPU offload, enforce_eager) vs diffusers -- isolate torch.compile

For each, computes per-frame SSIM and PSNR.
"""

import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# Must be run from repo root
sys.path.insert(0, os.getcwd())

MODEL_ID = "dg845/LTX-2.3-Diffusers"
PROMPT = "A lighthouse on a rocky cliff at sunset, waves crashing below, golden hour lighting"
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"
WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 25
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 4.0
SEED = 42
OUTPUT_DIR = "/tmp/ssim_investigation"


def compute_ssim_psnr(img1: Image.Image, img2: Image.Image):
    """Compute SSIM and PSNR between two PIL images."""
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    a1 = torch.from_numpy(np.array(img1)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    a2 = torch.from_numpy(np.array(img2)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    return ssim_metric(a1, a2).item(), psnr_metric(a1, a2).item()


def video_to_frames(video_np):
    """Convert numpy video to list of PIL Images."""
    while video_np.ndim > 4:
        video_np = video_np[0]
    if video_np.dtype in (np.float32, np.float64, np.float16):
        video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
    return [Image.fromarray(video_np[t]) for t in range(video_np.shape[0])]


def run_diffusers(run_name: str) -> list[Image.Image]:
    """Generate video with diffusers LTX2Pipeline."""
    from diffusers import LTX2Pipeline

    print(f"\n  [{run_name}] Loading diffusers pipeline...")
    pipe = LTX2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    print(f"  [{run_name}] Generating {NUM_FRAMES} frames...")
    t0 = time.time()
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
    elapsed = time.time() - t0
    print(f"  [{run_name}] Generation took {elapsed:.1f}s")

    video = result.frames
    if isinstance(video, list):
        if isinstance(video[0], list):
            frames = [img.convert("RGB") for img in video[0]]
        elif isinstance(video[0], np.ndarray):
            frames = video_to_frames(np.stack(video))
        else:
            frames = [v.convert("RGB") if isinstance(v, Image.Image) else v for v in video]
    else:
        frames = video_to_frames(video)

    # Save frames
    subdir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(subdir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(subdir, f"frame_{i:04d}.png"))

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return frames


def run_vllm_omni(run_name: str, enable_cpu_offload: bool, enforce_eager: bool) -> list[Image.Image]:
    """Generate video with vLLM-Omni LTX23Pipeline."""
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    print(f"\n  [{run_name}] Loading vLLM-Omni pipeline (cpu_offload={enable_cpu_offload}, eager={enforce_eager})...")
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)

    omni = Omni(
        model=MODEL_ID,
        model_class_name="LTX23Pipeline",
        enforce_eager=enforce_eager,
        enable_cpu_offload=enable_cpu_offload,
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

    prompt_dict = {"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT}
    print(f"  [{run_name}] Generating {NUM_FRAMES} frames...")
    t0 = time.time()
    results = omni.generate(prompt_dict, sampling_params)
    elapsed = time.time() - t0
    print(f"  [{run_name}] Generation took {elapsed:.1f}s")

    result = results[0] if isinstance(results, list) else results
    video_output = None
    if isinstance(result, OmniRequestOutput) and result.images:
        if len(result.images) == 1 and isinstance(result.images[0], tuple) and len(result.images[0]) == 2:
            video_output = result.images[0][0]
        elif len(result.images) == 1 and isinstance(result.images[0], dict):
            video_output = result.images[0].get("frames") or result.images[0].get("video")
        else:
            video_output = result.images

    if video_output is None:
        raise RuntimeError(f"No video output in result: {result}")

    if isinstance(video_output, np.ndarray):
        frames = video_to_frames(video_output)
    elif isinstance(video_output, list):
        if isinstance(video_output[0], np.ndarray):
            frames = video_to_frames(np.stack(video_output))
        elif isinstance(video_output[0], Image.Image):
            frames = [img.convert("RGB") for img in video_output]
        else:
            raise ValueError(f"Unexpected element type: {type(video_output[0])}")
    else:
        raise ValueError(f"Unexpected output type: {type(video_output)}")

    subdir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(subdir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(subdir, f"frame_{i:04d}.png"))

    del omni
    gc.collect()
    torch.cuda.empty_cache()
    return frames


def compare_frames(name: str, frames_a: list[Image.Image], frames_b: list[Image.Image]) -> dict:
    """Compare two sets of frames and return metrics."""
    n = min(len(frames_a), len(frames_b))
    ssim_scores = []
    psnr_scores = []
    for i in range(n):
        s, p = compute_ssim_psnr(frames_a[i], frames_b[i])
        ssim_scores.append(s)
        psnr_scores.append(p)

    result = {
        "name": name,
        "num_frames": n,
        "ssim_avg": sum(ssim_scores) / len(ssim_scores),
        "ssim_min": min(ssim_scores),
        "ssim_max": max(ssim_scores),
        "psnr_avg": sum(psnr_scores) / len(psnr_scores),
        "psnr_min": min(psnr_scores),
        "psnr_max": max(psnr_scores),
    }

    print(f"\n  === {name} ({n} frames) ===")
    print(f"  SSIM: avg={result['ssim_avg']:.6f}  min={result['ssim_min']:.6f}  max={result['ssim_max']:.6f}")
    print(f"  PSNR: avg={result['psnr_avg']:.2f} dB  min={result['psnr_min']:.2f} dB  max={result['psnr_max']:.2f} dB")

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    # ---- Experiment 1: diffusers vs diffusers ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: diffusers vs diffusers (determinism baseline)")
    print("=" * 70)
    diffusers_a = run_diffusers("diffusers_run_a")
    diffusers_b = run_diffusers("diffusers_run_b")
    r1 = compare_frames("diffusers_vs_diffusers", diffusers_a, diffusers_b)
    all_results.append(r1)

    # Keep diffusers_a as the reference for all subsequent comparisons
    reference_frames = diffusers_a

    # ---- Experiment 2: vllm-omni (NO cpu offload) vs diffusers ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: vllm-omni (no CPU offload, eager) vs diffusers")
    print("=" * 70)
    vllm_no_offload = run_vllm_omni("vllm_no_offload_eager", enable_cpu_offload=False, enforce_eager=True)
    r2 = compare_frames("vllm_no_offload_eager_vs_diffusers", vllm_no_offload, reference_frames)
    all_results.append(r2)

    # ---- Experiment 3: vllm-omni (WITH cpu offload, eager) vs diffusers ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: vllm-omni (CPU offload, eager) vs diffusers")
    print("=" * 70)
    vllm_offload = run_vllm_omni("vllm_offload_eager", enable_cpu_offload=True, enforce_eager=True)
    r3 = compare_frames("vllm_offload_eager_vs_diffusers", vllm_offload, reference_frames)
    all_results.append(r3)

    # ---- Experiment 4: vllm (no offload) vs vllm (offload) -- isolate offloading ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: vllm-omni (no offload) vs vllm-omni (offload)")
    print("=" * 70)
    r4 = compare_frames("vllm_no_offload_vs_offload", vllm_no_offload, vllm_offload)
    all_results.append(r4)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<45} {'SSIM avg':>10} {'PSNR avg':>12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<45} {r['ssim_avg']:>10.6f} {r['psnr_avg']:>10.2f} dB")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
