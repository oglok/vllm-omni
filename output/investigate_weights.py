#!/usr/bin/env python3
"""
Diagnose weight loading differences between vllm-omni custom transformer
and diffusers transformer for LTX-2.3.

1. Loads both transformers
2. Compares every named parameter
3. Logs any skipped weights from our load_weights
4. Runs a single forward pass through block 0 with identical inputs
"""

import gc
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

MODEL_ID = "dg845/LTX-2.3-Diffusers"


def compare_tensors(name, a, b):
    """Compare two tensors."""
    a = a.float().cpu()
    b = b.float().cpu()
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH a={list(a.shape)} b={list(b.shape)}")
        return False
    diff = (a - b).abs()
    max_diff = diff.max().item()
    status = "OK" if max_diff < 1e-6 else ("WARN" if max_diff < 1e-3 else "DIVERGED")
    if status != "OK":
        print(f"  {name}: {status} shape={list(a.shape)} max_diff={max_diff:.6e}")
    return status == "OK"


@torch.no_grad()
def main():
    local_files_only = os.path.exists(MODEL_ID)

    # ---- Load diffusers transformer ----
    print("=" * 70)
    print("Loading diffusers transformer...")
    print("=" * 70)
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel as DiffusersTransformer

    diffusers_transformer = DiffusersTransformer.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=local_files_only
    )
    diffusers_params = dict(diffusers_transformer.named_parameters())
    print(f"  Diffusers transformer: {len(diffusers_params)} parameters")

    # ---- Load our custom transformer ----
    print("\nLoading vllm-omni custom transformer...")
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import create_transformer_from_config, load_transformer_config
    from safetensors.torch import load_file as safetensors_load

    # Need vLLM config context and parallel state for CustomOp/TP initialization
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    vllm_config = VllmConfig()
    _ctx = set_current_vllm_config(vllm_config)
    _ctx.__enter__()

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1)

    transformer_config = load_transformer_config(MODEL_ID, "transformer", local_files_only)
    our_transformer = create_transformer_from_config(transformer_config)

    # Direct parameter name comparison -- no weight loading needed
    print("\nComparing parameter names...")

    our_params = dict(our_transformer.named_parameters())
    print(f"  Our transformer: {len(our_params)} parameters")
    print(f"  Diffusers transformer: {len(diffusers_params)} parameters")
    print(f"  Parameter count difference: {len(diffusers_params) - len(our_params)}")

    # Account for QKV fusion: diffusers has to_q, to_k, to_v; we have to_qkv
    # Map diffusers names to expected our names
    stacked_params_mapping = [
        (".attn1.to_qkv", ".attn1.to_q"),
        (".attn1.to_qkv", ".attn1.to_k"),
        (".attn1.to_qkv", ".attn1.to_v"),
        (".audio_attn1.to_qkv", ".audio_attn1.to_q"),
        (".audio_attn1.to_qkv", ".audio_attn1.to_k"),
        (".audio_attn1.to_qkv", ".audio_attn1.to_v"),
    ]

    diffusers_names = set(diffusers_params.keys())
    our_names = set(our_params.keys())

    # Map diffusers names to expected our names
    mapped_diffusers = set()
    for dname in diffusers_names:
        mapped = dname
        for our_name, diff_name in stacked_params_mapping:
            dn = diff_name.lstrip(".")
            on = our_name.lstrip(".")
            if dn in dname:
                mapped = dname.replace(dn, on)
                break
        mapped_diffusers.add(mapped)

    # Deduplicate (q/k/v all map to the same qkv)
    mapped_diffusers_unique = set()
    for m in mapped_diffusers:
        mapped_diffusers_unique.add(m)

    in_diffusers_not_ours = mapped_diffusers_unique - our_names
    in_ours_not_diffusers = our_names - mapped_diffusers_unique

    print(f"\n  Parameters in diffusers (mapped) but NOT in ours: {len(in_diffusers_not_ours)}")
    if in_diffusers_not_ours:
        # Group by block for readability
        from collections import defaultdict

        by_component = defaultdict(list)
        for name in sorted(in_diffusers_not_ours):
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "transformer_blocks":
                component = ".".join(parts[2:])
                by_component[component].append(parts[1])
            else:
                by_component[name].append("")

        print("    Grouped by component (across blocks):")
        for component, blocks in sorted(by_component.items()):
            if blocks[0] == "":
                print(f"      {component}")
            else:
                print(f"      *.{component} (blocks: {blocks[0]}..{blocks[-1]}, count={len(blocks)})")

    print(f"\n  Parameters in ours but NOT in diffusers (mapped): {len(in_ours_not_diffusers)}")
    if in_ours_not_diffusers:
        for name in sorted(in_ours_not_diffusers)[:20]:
            print(f"    {name}")

    loaded = set()  # placeholder

    # ---- Compare parameters by loading weights from diffusers into our model ----
    print("\n" + "=" * 70)
    print("WEIGHT VALUE COMPARISON")
    print("=" * 70)
    print("  Loading diffusers weights into our transformer via load_weights...")

    # Use diffusers state dict as weight source for our load_weights
    def _diffusers_weight_iter():
        for name, param in diffusers_params.items():
            yield name, param.data

    loaded = our_transformer.load_weights(_diffusers_weight_iter())
    print(f"  Loaded {len(loaded)} param names")

    # Now compare parameter values
    qkv_stacking = [
        (".attn1.to_qkv", ".attn1.to_q", ".attn1.to_k", ".attn1.to_v"),
        (".audio_attn1.to_qkv", ".audio_attn1.to_q", ".audio_attn1.to_k", ".audio_attn1.to_v"),
    ]

    match_count = 0
    mismatch_count = 0

    # Compare non-QKV params
    for dname, dparam in sorted(diffusers_params.items()):
        is_qkv = any(
            qkv[1].lstrip(".") in dname or qkv[2].lstrip(".") in dname or qkv[3].lstrip(".") in dname
            for qkv in qkv_stacking
        )
        if is_qkv:
            continue
        if dname in our_params:
            if compare_tensors(dname, our_params[dname].data, dparam.data):
                match_count += 1
            else:
                mismatch_count += 1

    # Compare QKV fused params
    print("\n  QKV Fused Comparison:")
    for qkv_name, q_name, k_name, v_name in qkv_stacking:
        qn = qkv_name.lstrip(".")
        qq = q_name.lstrip(".")
        kk = k_name.lstrip(".")
        vv = v_name.lstrip(".")

        # Find all blocks
        for oname, oparam in sorted(our_params.items()):
            if qn not in oname:
                continue
            # Build the expected diffusers names
            q_dname = oname.replace(qn, qq)
            k_dname = oname.replace(qn, kk)
            v_dname = oname.replace(qn, vv)

            if q_dname not in diffusers_params:
                continue

            q_w = diffusers_params[q_dname].data.float().cpu()
            k_w = diffusers_params[k_dname].data.float().cpu()
            v_w = diffusers_params[v_dname].data.float().cpu()
            expected = torch.cat([q_w, k_w, v_w], dim=0)
            actual = oparam.data.float().cpu()

            if actual.shape != expected.shape:
                print(f"    {oname}: SHAPE MISMATCH actual={list(actual.shape)} expected={list(expected.shape)}")
                mismatch_count += 1
            else:
                diff = (actual - expected).abs().max().item()
                if diff < 1e-6:
                    match_count += 1
                else:
                    print(f"    {oname}: DIVERGED max_diff={diff:.6e}")
                    q_size = q_w.shape[0]
                    k_size = k_w.shape[0]
                    print(f"      Q [{0}:{q_size}]: max_diff={(actual[:q_size] - q_w).abs().max():.6e}")
                    print(
                        f"      K [{q_size}:{q_size + k_size}]: max_diff={(actual[q_size : q_size + k_size] - k_w).abs().max():.6e}"
                    )
                    print(f"      V [{q_size + k_size}:]: max_diff={(actual[q_size + k_size :] - v_w).abs().max():.6e}")
                    mismatch_count += 1

    print(f"\n  SUMMARY:")
    print(f"    Matched: {match_count}")
    print(f"    Mismatched: {mismatch_count}")

    # ---- Single block forward comparison ----
    if mismatch_count == 0:
        print("\n" + "=" * 70)
        print("All weights match! Running single-block forward comparison...")
        print("=" * 70)

        # Can't fit both 44GB transformers on GPU simultaneously
        # Compare block 0 only -- extract and move individually
        diffusers_block0 = diffusers_transformer.transformer_blocks[0].to("cuda")
        our_block0 = our_transformer.transformer_blocks[0].to("cuda")

        # Also need model-level modules for timestep/RoPE
        diffusers_transformer.time_embed = diffusers_transformer.time_embed.to("cuda")
        diffusers_transformer.audio_time_embed = diffusers_transformer.audio_time_embed.to("cuda")
        diffusers_transformer.rope = (
            diffusers_transformer.rope.to("cuda")
            if hasattr(diffusers_transformer.rope, "to")
            else diffusers_transformer.rope
        )
        diffusers_transformer.audio_rope = (
            diffusers_transformer.audio_rope.to("cuda")
            if hasattr(diffusers_transformer.audio_rope, "to")
            else diffusers_transformer.audio_rope
        )
        our_transformer.time_embed = our_transformer.time_embed.to("cuda")
        our_transformer.audio_time_embed = our_transformer.audio_time_embed.to("cuda")
        our_transformer.rope = (
            our_transformer.rope.to("cuda") if hasattr(our_transformer.rope, "to") else our_transformer.rope
        )
        our_transformer.audio_rope = (
            our_transformer.audio_rope.to("cuda")
            if hasattr(our_transformer.audio_rope, "to")
            else our_transformer.audio_rope
        )
        if hasattr(diffusers_transformer, "prompt_adaln") and diffusers_transformer.prompt_adaln is not None:
            diffusers_transformer.prompt_adaln = diffusers_transformer.prompt_adaln.to("cuda")
            our_transformer.prompt_adaln = our_transformer.prompt_adaln.to("cuda")
        if (
            hasattr(diffusers_transformer, "audio_prompt_adaln")
            and diffusers_transformer.audio_prompt_adaln is not None
        ):
            diffusers_transformer.audio_prompt_adaln = diffusers_transformer.audio_prompt_adaln.to("cuda")
            our_transformer.audio_prompt_adaln = our_transformer.audio_prompt_adaln.to("cuda")
        # cross-attn timestep modules
        for attr in [
            "av_cross_attn_video_scale_shift",
            "av_cross_attn_audio_scale_shift",
            "av_cross_attn_video_a2v_gate",
            "av_cross_attn_audio_v2a_gate",
        ]:
            if hasattr(diffusers_transformer, attr):
                setattr(diffusers_transformer, attr, getattr(diffusers_transformer, attr).to("cuda"))
                setattr(our_transformer, attr, getattr(our_transformer, attr).to("cuda"))

        # Create test inputs for block 0
        batch = 1
        video_seq = 768  # 4*12*16
        audio_seq = 100
        text_seq = 1024

        video_hidden = torch.randn(batch, video_seq, 4096, dtype=torch.bfloat16, device="cuda")
        audio_hidden = torch.randn(batch, audio_seq, 2048, dtype=torch.bfloat16, device="cuda")
        encoder_hidden = torch.randn(batch, text_seq, 4096, dtype=torch.bfloat16, device="cuda")
        audio_encoder_hidden = torch.randn(batch, text_seq, 2048, dtype=torch.bfloat16, device="cuda")
        timestep = torch.tensor([500.0], device="cuda")

        # Compute shared inputs for block 0
        print("  Computing timestep embeddings and RoPE...")

        video_coords = diffusers_transformer.rope.prepare_video_coords(batch, 4, 12, 16, "cuda", fps=24.0)
        audio_coords = diffusers_transformer.audio_rope.prepare_audio_coords(batch, audio_seq, "cuda")
        video_rotary_emb = diffusers_transformer.rope(video_coords)
        audio_rotary_emb = diffusers_transformer.audio_rope(audio_coords)

        # Timestep embeddings
        ts_bf16 = timestep.flatten().to(torch.bfloat16)
        diff_temb, diff_embedded_ts = diffusers_transformer.time_embed(ts_bf16)
        our_temb, our_embedded_ts = our_transformer.time_embed(ts_bf16)
        compare_tensors("time_embed output", our_temb, diff_temb)

        diff_temb_audio, _ = diffusers_transformer.audio_time_embed(ts_bf16)
        our_temb_audio, _ = our_transformer.audio_time_embed(ts_bf16)
        compare_tensors("audio_time_embed output", our_temb_audio, diff_temb_audio)

        # Prompt modulation
        diff_temb_prompt = our_temb_prompt = None
        diff_temb_prompt_audio = our_temb_prompt_audio = None
        if hasattr(diffusers_transformer, "prompt_adaln") and diffusers_transformer.prompt_adaln is not None:
            diff_temb_prompt, _ = diffusers_transformer.prompt_adaln(ts_bf16)
            our_temb_prompt, _ = our_transformer.prompt_adaln(ts_bf16)
            compare_tensors("prompt_adaln output", our_temb_prompt, diff_temb_prompt)
            diff_temb_prompt_audio, _ = diffusers_transformer.audio_prompt_adaln(ts_bf16)
            our_temb_prompt_audio, _ = our_transformer.audio_prompt_adaln(ts_bf16)
            compare_tensors("audio_prompt_adaln output", our_temb_prompt_audio, diff_temb_prompt_audio)

        # Cross-attn timestep modules
        diff_ca_ss, _ = diffusers_transformer.av_cross_attn_video_scale_shift(ts_bf16)
        our_ca_ss, _ = our_transformer.av_cross_attn_video_scale_shift(ts_bf16)
        compare_tensors("av_cross_attn_video_scale_shift", our_ca_ss, diff_ca_ss)

        print("\n  Running block 0 forward on both...")

        # Build block kwargs for diffusers block 0
        encoder_mask = torch.ones(batch, text_seq, device="cuda")
        diff_block_out_v, diff_block_out_a = diffusers_block0(
            hidden_states=video_hidden.clone(),
            audio_hidden_states=audio_hidden.clone(),
            encoder_hidden_states=encoder_hidden.clone(),
            audio_encoder_hidden_states=audio_encoder_hidden.clone(),
            temb=diff_temb,
            temb_audio=diff_temb_audio,
            temb_ca_scale_shift=diff_ca_ss,
            temb_ca_audio_scale_shift=diff_ca_ss,  # simplified
            temb_ca_gate=diff_ca_ss,  # simplified
            temb_ca_audio_gate=diff_ca_ss,  # simplified
            temb_prompt=diff_temb_prompt,
            temb_prompt_audio=diff_temb_prompt_audio,
            video_rotary_emb=video_rotary_emb,
            audio_rotary_emb=audio_rotary_emb,
            ca_video_rotary_emb=video_rotary_emb,
            ca_audio_rotary_emb=audio_rotary_emb,
            encoder_attention_mask=encoder_mask,
            audio_encoder_attention_mask=encoder_mask,
        )

        our_block_out_v, our_block_out_a = our_block0(
            hidden_states=video_hidden.clone(),
            audio_hidden_states=audio_hidden.clone(),
            encoder_hidden_states=encoder_hidden.clone(),
            audio_encoder_hidden_states=audio_encoder_hidden.clone(),
            temb=our_temb,
            temb_audio=our_temb_audio,
            temb_ca_scale_shift=our_ca_ss,
            temb_ca_audio_scale_shift=our_ca_ss,
            temb_ca_gate=our_ca_ss,
            temb_ca_audio_gate=our_ca_ss,
            temb_prompt=our_temb_prompt,
            temb_prompt_audio=our_temb_prompt_audio,
            video_rotary_emb=video_rotary_emb,
            audio_rotary_emb=audio_rotary_emb,
            ca_video_rotary_emb=video_rotary_emb,
            ca_audio_rotary_emb=audio_rotary_emb,
            encoder_attention_mask=encoder_mask,
            audio_encoder_attention_mask=encoder_mask,
        )

        print("\n  Block 0 output comparison:")
        compare_tensors("block0_video_output", our_block_out_v, diff_block_out_v)
        compare_tensors("block0_audio_output", our_block_out_a, diff_block_out_a)
    else:
        print("\n  SKIPPING forward comparison due to weight mismatches")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
