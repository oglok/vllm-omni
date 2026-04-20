#!/usr/bin/env python3
"""
Direct pipeline comparison: bypass the Omni API entirely.
Instantiate LTX23Pipeline directly and run the forward pass,
comparing against diffusers.
"""

import gc
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())

MODEL_ID = "dg845/LTX-2.3-Diffusers"
PROMPT = "A lighthouse on a rocky cliff at sunset, waves crashing below, golden hour lighting"
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"
WIDTH, HEIGHT, NUM_FRAMES = 512, 384, 25
NUM_INFERENCE_STEPS, GUIDANCE_SCALE, SEED = 20, 4.0, 42


def video_to_frames(v):
    if isinstance(v, list):
        if isinstance(v[0], list):
            return [f.convert("RGB") for f in v[0]]
        return v
    while v.ndim > 4:
        v = v[0]
    if v.dtype in (np.float32, np.float64, np.float16):
        v = np.clip(v * 255, 0, 255).astype(np.uint8)
    return [Image.fromarray(v[t]) for t in range(v.shape[0])]


def run_diffusers():
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=gen,
        output_type="np",
    )
    frames = video_to_frames(result.frames)
    del pipe, result
    gc.collect()
    torch.cuda.empty_cache()
    return frames


def run_direct_pipeline():
    """Run our LTX23Pipeline directly (no Omni API)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import glob as glob_mod

    vllm_config = VllmConfig()
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29502")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1)

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Create a minimal od_config
    od_config = OmniDiffusionConfig(model=MODEL_ID)

    print("  Loading LTX23Pipeline directly...")
    pipeline = LTX23Pipeline(od_config=od_config)

    # Load transformer weights
    config_path = hf_hub_download(MODEL_ID, "transformer/config.json")
    model_dir = os.path.dirname(config_path)
    shard_files = sorted(glob_mod.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        for i in range(1, 9):
            sf = hf_hub_download(MODEL_ID, f"transformer/model-{i:05d}-of-00008.safetensors")
            shard_files.append(sf)

    def weight_iter():
        for sf in shard_files:
            state = load_file(sf)
            for k, v in state.items():
                yield k, v
            del state

    pipeline.load_weights(weight_iter())
    pipeline = pipeline.to(dtype=torch.bfloat16)
    # Move transformer to GPU (everything else stays CPU for offloading)
    pipeline.transformer = pipeline.transformer.to("cuda")
    pipeline.eval()

    print("  Running forward...")
    sampling_params = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
        output_type="np",
    )

    req = OmniDiffusionRequest(
        request_id="test-direct",
        request_ids=["test-direct"],
        prompts=[{"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT}],
        sampling_params=sampling_params,
    )

    with torch.no_grad():
        result = pipeline.forward(req)

    video = result.output
    if isinstance(video, tuple):
        video = video[0]  # (video, audio)
    frames = video_to_frames(video)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return frames


def main():
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

    print("=== Running diffusers ===")
    diff_frames = run_diffusers()
    print(f"  Got {len(diff_frames)} frames")

    print("\n=== Running direct LTX23Pipeline ===")
    our_frames = run_direct_pipeline()
    print(f"  Got {len(our_frames)} frames")

    print("\n=== Comparing ===")
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_m = PeakSignalNoiseRatio(data_range=1.0)
    ssims = []
    n = min(len(diff_frames), len(our_frames))
    for i in range(n):
        a = torch.from_numpy(np.array(diff_frames[i])).float().permute(2, 0, 1).unsqueeze(0) / 255
        b = torch.from_numpy(np.array(our_frames[i])).float().permute(2, 0, 1).unsqueeze(0) / 255
        ssims.append(ssim_m(a, b).item())

    avg_ssim = sum(ssims) / len(ssims)
    min_ssim = min(ssims)
    print(f"  SSIM: avg={avg_ssim:.6f} min={min_ssim:.6f}")
    if avg_ssim > 0.95:
        print("  SUCCESS: SSIM > 0.95!")
    elif avg_ssim > 0.90:
        print(f"  CLOSE: SSIM {avg_ssim:.4f}")
    else:
        print(f"  STILL LOW: SSIM {avg_ssim:.4f}")


if __name__ == "__main__":
    main()
