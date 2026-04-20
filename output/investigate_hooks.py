#!/usr/bin/env python3
"""
Hook-based comparison: run each transformer separately through a full forward pass,
capture block-0 intermediate outputs (every submodule), save to disk, then compare.

This avoids OOM by only having one transformer on GPU at a time.
"""

import gc
import json
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.getcwd())

MODEL_ID = "dg845/LTX-2.3-Diffusers"
SAVE_DIR = "/tmp/hook_outputs"


def setup_hooks(model, prefix=""):
    """Register forward hooks on all leaf modules of a model, capturing inputs and outputs."""
    captures = OrderedDict()

    def make_hook(name):
        def hook_fn(module, input, output):
            # Store first input and output tensors
            inp = input[0] if isinstance(input, tuple) and len(input) > 0 else input
            out = output[0] if isinstance(output, tuple) and len(output) > 0 else output
            if isinstance(inp, torch.Tensor):
                captures[f"{name}.input"] = inp.detach().cpu().float()
            if isinstance(out, torch.Tensor):
                captures[f"{name}.output"] = out.detach().cpu().float()

        return hook_fn

    handles = []
    for name, module in model.named_modules():
        if name and not list(module.children()):  # leaf modules only
            h = module.register_forward_hook(make_hook(f"{prefix}{name}"))
            handles.append(h)
    return captures, handles


def run_diffusers_forward():
    """Load diffusers transformer, run forward, save block-0 outputs."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    local_files_only = os.path.exists(MODEL_ID)

    print("Loading diffusers transformer to GPU...")
    transformer = (
        LTX2VideoTransformer3DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=local_files_only
        )
        .to("cuda")
        .eval()
    )

    # Hook only block 0 and model-level modules (not all 48 blocks)
    captures = OrderedDict()
    handles = []

    def make_hook(name):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) and len(input) > 0 else input
            out = output[0] if isinstance(output, tuple) and len(output) > 0 else output
            if isinstance(inp, torch.Tensor):
                captures[f"{name}.input"] = inp.detach().cpu().float()
            if isinstance(out, torch.Tensor):
                captures[f"{name}.output"] = out.detach().cpu().float()

        return hook_fn

    # Hook block 0 and top-level modules
    for name, module in transformer.named_modules():
        # Only hook block 0 submodules, and top-level modules (not blocks 1-47)
        if name.startswith("transformer_blocks.0.") or (
            name and "transformer_blocks" not in name and not list(module.children())
        ):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    # Also hook the full block 0 to get its output
    def block0_hook(module, input, output):
        if isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    captures[f"transformer_blocks.0.full_output_{i}"] = o.detach().cpu().float()
        elif isinstance(output, torch.Tensor):
            captures["transformer_blocks.0.full_output"] = output.detach().cpu().float()

    handles.append(transformer.transformer_blocks[0].register_forward_hook(block0_hook))

    # Prepare inputs
    batch = 1
    video_seq = 768
    audio_seq = 100
    text_seq = 128  # small for speed

    torch.manual_seed(42)
    video_hidden = torch.randn(batch, video_seq, 128, dtype=torch.bfloat16, device="cuda")
    audio_hidden = torch.randn(batch, audio_seq, 128, dtype=torch.bfloat16, device="cuda")
    encoder_hidden = torch.randn(batch, text_seq, 4096, dtype=torch.bfloat16, device="cuda")
    audio_encoder_hidden = torch.randn(batch, text_seq, 2048, dtype=torch.bfloat16, device="cuda")
    timestep = torch.tensor([500.0], device="cuda", dtype=torch.bfloat16)
    encoder_mask = torch.ones(batch, text_seq, device="cuda")

    # Save inputs for verification
    torch.save(
        {
            "video_hidden": video_hidden.cpu(),
            "audio_hidden": audio_hidden.cpu(),
            "encoder_hidden": encoder_hidden.cpu(),
            "audio_encoder_hidden": audio_encoder_hidden.cpu(),
            "timestep": timestep.cpu(),
            "encoder_mask": encoder_mask.cpu(),
        },
        os.path.join(SAVE_DIR, "inputs.pt"),
    )

    video_coords = transformer.rope.prepare_video_coords(batch, 4, 12, 16, "cuda", fps=24.0)
    audio_coords = transformer.audio_rope.prepare_audio_coords(batch, audio_seq, "cuda")

    print("Running diffusers forward...")
    with torch.no_grad():
        out_v, out_a = transformer(
            hidden_states=video_hidden,
            audio_hidden_states=audio_hidden,
            encoder_hidden_states=encoder_hidden,
            audio_encoder_hidden_states=audio_encoder_hidden,
            timestep=timestep,
            sigma=timestep,
            encoder_attention_mask=encoder_mask,
            audio_encoder_attention_mask=encoder_mask,
            num_frames=4,
            height=12,
            width=16,
            fps=24.0,
            audio_num_frames=audio_seq,
            video_coords=video_coords,
            audio_coords=audio_coords,
            return_dict=False,
        )

    captures["final_video_output"] = out_v.detach().cpu().float()
    captures["final_audio_output"] = out_a.detach().cpu().float()

    # Save RoPE for comparison
    torch.save(
        {
            "video_coords": video_coords.cpu(),
            "audio_coords": audio_coords.cpu(),
        },
        os.path.join(SAVE_DIR, "coords.pt"),
    )

    # Remove hooks
    for h in handles:
        h.remove()

    # Save captures
    torch.save(captures, os.path.join(SAVE_DIR, "diffusers_captures.pt"))
    print(f"  Saved {len(captures)} captures")

    # Free
    del transformer
    gc.collect()
    torch.cuda.empty_cache()


def run_ours_forward():
    """Load our transformer, run forward with same inputs, save block-0 outputs."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import create_transformer_from_config, load_transformer_config

    local_files_only = os.path.exists(MODEL_ID)

    vllm_config = VllmConfig()
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1)

    print("Loading our transformer...")
    transformer_config = load_transformer_config(MODEL_ID, "transformer", local_files_only)
    transformer = create_transformer_from_config(transformer_config)

    # Load weights from diffusers state dict
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    print("Loading weights...")
    # Find and load all safetensor shards
    model_dir = hf_hub_download(MODEL_ID, "transformer/config.json", local_files_only=local_files_only)
    model_dir = os.path.dirname(model_dir)

    import glob as glob_mod

    shard_files = sorted(glob_mod.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        # Try to download
        shard_files = []
        for i in range(1, 9):
            sf = hf_hub_download(
                MODEL_ID, f"transformer/model-{i:05d}-of-00008.safetensors", local_files_only=local_files_only
            )
            shard_files.append(sf)

    def weight_iter():
        for sf in shard_files:
            state = load_file(sf)
            for k, v in state.items():
                yield k, v
            del state

    loaded = transformer.load_weights(weight_iter())
    print(f"  Loaded {len(loaded)} params")

    transformer = transformer.to(dtype=torch.bfloat16, device="cuda").eval()

    # Hook block 0 and top-level modules
    captures = OrderedDict()
    handles = []

    def make_hook(name):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) and len(input) > 0 else input
            out = output[0] if isinstance(output, tuple) and len(output) > 0 else output
            if isinstance(inp, torch.Tensor):
                captures[f"{name}.input"] = inp.detach().cpu().float()
            if isinstance(out, torch.Tensor):
                captures[f"{name}.output"] = out.detach().cpu().float()

        return hook_fn

    for name, module in transformer.named_modules():
        if name.startswith("transformer_blocks.0.") or (
            name and "transformer_blocks" not in name and not list(module.children())
        ):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    def block0_hook(module, input, output):
        if isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    captures[f"transformer_blocks.0.full_output_{i}"] = o.detach().cpu().float()
        elif isinstance(output, torch.Tensor):
            captures["transformer_blocks.0.full_output"] = output.detach().cpu().float()

    handles.append(transformer.transformer_blocks[0].register_forward_hook(block0_hook))

    # Load same inputs
    inputs = torch.load(os.path.join(SAVE_DIR, "inputs.pt"))
    coords = torch.load(os.path.join(SAVE_DIR, "coords.pt"))

    video_hidden = inputs["video_hidden"].to("cuda")
    audio_hidden = inputs["audio_hidden"].to("cuda")
    encoder_hidden = inputs["encoder_hidden"].to("cuda")
    audio_encoder_hidden = inputs["audio_encoder_hidden"].to("cuda")
    timestep = inputs["timestep"].to("cuda")
    encoder_mask = inputs["encoder_mask"].to("cuda")
    video_coords = coords["video_coords"].to("cuda")
    audio_coords = coords["audio_coords"].to("cuda")

    print("Running our forward...")
    with torch.no_grad():
        out_v, out_a = transformer(
            hidden_states=video_hidden,
            audio_hidden_states=audio_hidden,
            encoder_hidden_states=encoder_hidden,
            audio_encoder_hidden_states=audio_encoder_hidden,
            timestep=timestep,
            sigma=timestep,
            encoder_attention_mask=encoder_mask,
            audio_encoder_attention_mask=encoder_mask,
            num_frames=4,
            height=12,
            width=16,
            fps=24.0,
            audio_num_frames=audio_hidden.shape[1],
            video_coords=video_coords,
            audio_coords=audio_coords,
            return_dict=False,
        )

    captures["final_video_output"] = out_v.detach().cpu().float()
    captures["final_audio_output"] = out_a.detach().cpu().float()

    for h in handles:
        h.remove()

    torch.save(captures, os.path.join(SAVE_DIR, "ours_captures.pt"))
    print(f"  Saved {len(captures)} captures")

    del transformer
    gc.collect()
    torch.cuda.empty_cache()


def compare_captures():
    """Load both captures and find the first divergence."""
    diff_caps = torch.load(os.path.join(SAVE_DIR, "diffusers_captures.pt"))
    our_caps = torch.load(os.path.join(SAVE_DIR, "ours_captures.pt"))

    print(f"\nDiffusers captures: {len(diff_caps)} entries")
    print(f"Our captures: {len(our_caps)} entries")

    # Find common keys
    common = sorted(set(diff_caps.keys()) & set(our_caps.keys()))
    only_diff = sorted(set(diff_caps.keys()) - set(our_caps.keys()))
    only_ours = sorted(set(our_caps.keys()) - set(diff_caps.keys()))

    print(f"Common keys: {len(common)}")
    if only_diff:
        print(f"Only in diffusers ({len(only_diff)}):")
        for k in only_diff[:10]:
            print(f"  {k}: shape={list(diff_caps[k].shape)}")
    if only_ours:
        print(f"Only in ours ({len(only_ours)}):")
        for k in only_ours[:10]:
            print(f"  {k}: shape={list(our_caps[k].shape)}")

    print("\n" + "=" * 80)
    print("COMPARISON (sorted by name, showing first divergences)")
    print("=" * 80)

    first_diverged = None
    match_count = 0
    mismatch_count = 0

    for key in common:
        a = diff_caps[key]
        b = our_caps[key]
        if a.shape != b.shape:
            print(f"  {key}: SHAPE MISMATCH diff={list(a.shape)} ours={list(b.shape)}")
            mismatch_count += 1
            if first_diverged is None:
                first_diverged = key
            continue

        diff = (a - b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        if max_diff < 1e-4:
            match_count += 1
        else:
            status = "WARN" if max_diff < 0.1 else "DIVERGED"
            print(f"  {key}: {status} max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} cos={cos_sim:.8f}")
            mismatch_count += 1
            if first_diverged is None:
                first_diverged = key

    print(f"\nSUMMARY: {match_count} matched, {mismatch_count} diverged")
    if first_diverged:
        print(f"FIRST DIVERGENCE AT: {first_diverged}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: Run diffusers transformer")
    print("=" * 80)
    run_diffusers_forward()

    print("\n" + "=" * 80)
    print("PHASE 2: Run our transformer")
    print("=" * 80)
    run_ours_forward()

    print("\n" + "=" * 80)
    print("PHASE 3: Compare captures")
    print("=" * 80)
    compare_captures()


if __name__ == "__main__":
    main()
