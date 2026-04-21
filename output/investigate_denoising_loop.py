#!/usr/bin/env python3
"""
Compare denoising loop step-by-step between diffusers LTX2Pipeline
and our LTX23Pipeline.

Hooks into both pipelines to capture:
- Pre-transformer latents at each step
- Transformer output at each step
- Post-scheduler latents at each step
- Scheduler sigmas

This finds the exact step and component where divergence begins.
"""

import copy
import gc
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

MODEL_ID = "dg845/LTX-2.3-Diffusers"
PROMPT = "A lighthouse on a rocky cliff at sunset"
NEGATIVE_PROMPT = "blurry, low quality"
WIDTH, HEIGHT, NUM_FRAMES = 512, 384, 25
NUM_INFERENCE_STEPS = 5  # Only 5 steps to keep it fast
GUIDANCE_SCALE = 4.0
SEED = 42


def compare_tensors(name, a, b):
    if a is None or b is None:
        print(f"  {name}: one is None")
        return
    a, b = a.float().cpu(), b.float().cpu()
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH a={list(a.shape)} b={list(b.shape)}")
        return
    diff = (a - b).abs()
    mx = diff.max().item()
    mn = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
    status = "OK" if mx < 1e-4 else ("WARN" if mx < 0.01 else "DIVERGED")
    if status != "OK":
        print(f"  {name}: {status} max={mx:.6e} mean={mn:.6e} cos={cos:.8f}")
    return mx


@torch.no_grad()
def run_diffusers():
    """Run diffusers and capture per-step latents."""
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    captures = OrderedDict()

    # Hook the scheduler step to capture latents
    orig_scheduler_step = pipe.scheduler.step.__func__
    step_count = [0]

    def hooked_step(self, model_output, timestep, sample, **kwargs):
        i = step_count[0]
        captures[f"step_{i}_pre_scheduler_video_velocity"] = model_output.detach().cpu().float()
        captures[f"step_{i}_pre_scheduler_video_latents"] = sample.detach().cpu().float()
        result = orig_scheduler_step(self, model_output, timestep, sample, **kwargs)
        captures[f"step_{i}_post_scheduler_video_latents"] = result[0].detach().cpu().float()
        captures[f"step_{i}_sigma"] = torch.tensor(self.sigmas[i].item() if i < len(self.sigmas) else 0.0)
        step_count[0] += 1
        return result

    import types

    pipe.scheduler.step = types.MethodType(hooked_step, pipe.scheduler)

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

    # Save scheduler sigmas
    captures["scheduler_sigmas"] = torch.tensor([s for s in pipe.scheduler.sigmas])
    captures["num_steps"] = step_count[0]

    del pipe, result
    gc.collect()
    torch.cuda.empty_cache()
    return captures


@torch.no_grad()
def run_our_pipeline():
    """Run our LTX23Pipeline via Omni and capture per-step latents."""
    # We need to hook into our pipeline's denoising loop
    # The cleanest way: monkey-patch the scheduler step

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import load_transformer_config, create_transformer_from_config
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    vllm_config = VllmConfig()
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29504")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1)

    # Load pipeline components
    od_config = OmniDiffusionConfig(model=MODEL_ID)
    pipeline = LTX23Pipeline(od_config=od_config)

    # Load transformer weights from safetensors
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import glob as glob_mod

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

    pipeline.load_weights(weight_iter())
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.transformer = pipeline.transformer.to("cuda")
    pipeline.eval()

    captures = OrderedDict()

    # Hook the scheduler step
    orig_step = pipeline.scheduler.step.__func__
    step_count = [0]

    def hooked_step(self, model_output, timestep, sample, **kwargs):
        i = step_count[0]
        captures[f"step_{i}_pre_scheduler_video_velocity"] = model_output.detach().cpu().float()
        captures[f"step_{i}_pre_scheduler_video_latents"] = sample.detach().cpu().float()
        result = orig_step(self, model_output, timestep, sample, **kwargs)
        captures[f"step_{i}_post_scheduler_video_latents"] = result[0].detach().cpu().float()
        captures[f"step_{i}_sigma"] = torch.tensor(self.sigmas[i].item() if i < len(self.sigmas) else 0.0)
        step_count[0] += 1
        return result

    import types

    pipeline.scheduler.step = types.MethodType(hooked_step, pipeline.scheduler)

    # Create request
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
        request_id="test",
        request_ids=["test"],
        prompts=[{"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT}],
        sampling_params=sampling_params,
    )

    result = pipeline.forward(req)

    captures["scheduler_sigmas"] = torch.tensor([s for s in pipeline.scheduler.sigmas])
    captures["num_steps"] = step_count[0]

    del pipeline, result
    gc.collect()
    torch.cuda.empty_cache()
    return captures


def main():
    print("=" * 70)
    print("DENOISING LOOP COMPARISON")
    print(f"Steps: {NUM_INFERENCE_STEPS}, Guidance: {GUIDANCE_SCALE}, Seed: {SEED}")
    print("=" * 70)

    print("\n--- Running diffusers ---")
    diff_caps = run_diffusers()
    print(f"  Captured {len(diff_caps)} entries, {diff_caps['num_steps']} scheduler steps")

    print("\n--- Running our pipeline ---")
    our_caps = run_our_pipeline()
    print(f"  Captured {len(our_caps)} entries, {our_caps['num_steps']} scheduler steps")

    print("\n" + "=" * 70)
    print("STEP-BY-STEP COMPARISON")
    print("=" * 70)

    # Compare scheduler sigmas
    print("\nScheduler sigmas:")
    compare_tensors("sigmas", diff_caps["scheduler_sigmas"], our_caps["scheduler_sigmas"])

    n_steps = min(diff_caps["num_steps"], our_caps["num_steps"])
    for i in range(n_steps):
        print(f"\n--- Step {i} ---")
        sigma_d = diff_caps.get(f"step_{i}_sigma", torch.tensor(0.0)).item()
        sigma_o = our_caps.get(f"step_{i}_sigma", torch.tensor(0.0)).item()
        print(f"  sigma: diffusers={sigma_d:.8f} ours={sigma_o:.8f} diff={abs(sigma_d - sigma_o):.2e}")

        compare_tensors(
            f"pre_scheduler_velocity",
            diff_caps.get(f"step_{i}_pre_scheduler_video_velocity"),
            our_caps.get(f"step_{i}_pre_scheduler_video_velocity"),
        )
        compare_tensors(
            f"pre_scheduler_latents",
            diff_caps.get(f"step_{i}_pre_scheduler_video_latents"),
            our_caps.get(f"step_{i}_pre_scheduler_video_latents"),
        )
        compare_tensors(
            f"post_scheduler_latents",
            diff_caps.get(f"step_{i}_post_scheduler_video_latents"),
            our_caps.get(f"step_{i}_post_scheduler_video_latents"),
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
