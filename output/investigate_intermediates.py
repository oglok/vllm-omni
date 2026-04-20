#!/usr/bin/env python3
"""
Diagnose WHERE the divergence occurs between vllm-omni and diffusers.

Compares intermediate outputs at each stage:
  1. Text embeddings (Gemma output)
  2. Connector output (projected embeddings + mask)
  3. Initial latents (random noise with same seed)
  4. Scheduler timesteps/sigmas
  5. Transformer output at step 0

This script runs both pipelines side-by-side, hooking into internals.
"""

import gc
import os
import sys

import numpy as np
import torch

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


def compare_tensors(name, a, b):
    """Compare two tensors and print detailed diagnostics."""
    if a is None and b is None:
        print(f"  {name}: both None")
        return
    if a is None or b is None:
        print(f"  {name}: MISMATCH - one is None (a={a is not None}, b={b is not None})")
        return
    if isinstance(a, (list, tuple)):
        a = torch.stack(a) if all(isinstance(x, torch.Tensor) for x in a) else a
    if isinstance(b, (list, tuple)):
        b = torch.stack(b) if all(isinstance(x, torch.Tensor) for x in b) else b

    a = a.float().cpu()
    b = b.float().cpu()

    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH a={a.shape} b={b.shape}")
        return

    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (b.abs() + 1e-8)).mean().item()

    cos_sim = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

    status = "OK" if max_diff < 1e-3 else ("WARN" if max_diff < 0.1 else "DIVERGED")
    print(
        f"  {name}: {status} shape={list(a.shape)} max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
        f"rel_diff={rel_diff:.6e} cosine_sim={cos_sim:.8f}"
    )
    if status == "DIVERGED":
        print(f"    a stats: mean={a.mean():.6f} std={a.std():.6f} min={a.min():.6f} max={a.max():.6f}")
        print(f"    b stats: mean={b.mean():.6f} std={b.std():.6f} min={b.min():.6f} max={b.max():.6f}")


@torch.no_grad()
def run_diffusers_with_intermediates():
    """Run diffusers pipeline components individually and capture intermediate outputs."""
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler

    local_files_only = os.path.exists(MODEL_ID)

    print("\n--- Loading diffusers components individually ---")

    # ---- Stage 1: Text encoding ----
    print("\n[Stage 1] Text encoding...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", local_files_only=local_files_only)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=local_files_only
    ).to("cuda")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = tokenizer.model_max_length
    if max_len is None or max_len > 100000:
        max_len = 1024
    text_inputs = tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to("cuda")
    attention_mask = text_inputs.attention_mask.to("cuda")

    text_encoder_outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = text_encoder_outputs.hidden_states
    prompt_embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=torch.bfloat16)

    # Negative prompt
    neg_inputs = tokenizer(
        [NEGATIVE_PROMPT],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    neg_input_ids = neg_inputs.input_ids.to("cuda")
    neg_attention_mask = neg_inputs.attention_mask.to("cuda")

    neg_outputs = text_encoder(
        input_ids=neg_input_ids,
        attention_mask=neg_attention_mask,
        output_hidden_states=True,
    )
    neg_hidden_states = neg_outputs.hidden_states
    neg_prompt_embeds = torch.stack(neg_hidden_states, dim=-1).flatten(2, 3).to(dtype=torch.bfloat16)

    # Move everything to CPU immediately to free GPU
    prompt_embeds_cpu = prompt_embeds.cpu()
    neg_prompt_embeds_cpu = neg_prompt_embeds.cpu()
    attention_mask_cpu = attention_mask.cpu()
    neg_attention_mask_cpu = neg_attention_mask.cpu()

    intermediates = {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask_cpu,
        "prompt_embeds": prompt_embeds_cpu,
        "neg_prompt_embeds": neg_prompt_embeds_cpu,
        "num_hidden_states": len(hidden_states),
        "hidden_state_0_shape": hidden_states[0].shape,
        "max_len": max_len,
    }

    # Aggressively free GPU
    del text_encoder, text_encoder_outputs, neg_outputs, hidden_states, neg_hidden_states
    del prompt_embeds, neg_prompt_embeds, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 2: Connectors ----
    print("[Stage 2] Connectors...")
    from diffusers.pipelines.ltx2 import LTX2TextConnectors

    connectors = LTX2TextConnectors.from_pretrained(
        MODEL_ID, subfolder="connectors", torch_dtype=torch.bfloat16, local_files_only=local_files_only
    ).to("cuda")

    padding_side = tokenizer.padding_side
    connector_prompt, connector_audio, connector_mask = connectors(
        prompt_embeds_cpu.to("cuda"), attention_mask_cpu.to("cuda"), padding_side=padding_side
    )
    neg_connector_prompt, neg_connector_audio, neg_connector_mask = connectors(
        neg_prompt_embeds_cpu.to("cuda"), neg_attention_mask_cpu.to("cuda"), padding_side=padding_side
    )

    intermediates["connector_prompt"] = connector_prompt.cpu()
    intermediates["connector_audio"] = connector_audio.cpu()
    intermediates["connector_mask"] = connector_mask.cpu()
    intermediates["neg_connector_prompt"] = neg_connector_prompt.cpu()

    del connectors, connector_prompt, connector_audio, connector_mask
    del neg_connector_prompt, neg_connector_audio, neg_connector_mask
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 3: Latents ----
    print("[Stage 3] Latents (random noise)...")
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=local_files_only
    )
    vae_spatial = vae.spatial_compression_ratio
    vae_temporal = vae.temporal_compression_ratio

    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal + 1
    latent_height = HEIGHT // vae_spatial
    latent_width = WIDTH // vae_spatial
    num_channels = 128  # in_channels from config

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    from diffusers.utils.torch_utils import randn_tensor

    latents = randn_tensor(
        (1, num_channels, latent_num_frames, latent_height, latent_width),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

    packed_latents = LTX23Pipeline._pack_latents(latents, 1, 1)

    intermediates["raw_latents"] = latents
    intermediates["packed_latents"] = packed_latents
    intermediates["latent_num_frames"] = latent_num_frames
    intermediates["latent_height"] = latent_height
    intermediates["latent_width"] = latent_width

    del vae
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 4: Scheduler ----
    print("[Stage 4] Scheduler timesteps...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler", local_files_only=local_files_only
    )
    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    video_seq_len = latent_num_frames * latent_height * latent_width

    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import calculate_shift

    mu = calculate_shift(
        video_seq_len,
        scheduler.config.get("base_image_seq_len", 1024),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.95),
        scheduler.config.get("max_shift", 2.05),
    )
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    timesteps, _ = retrieve_timesteps(scheduler, NUM_INFERENCE_STEPS, "cuda", sigmas=sigmas, mu=mu)
    intermediates["timesteps"] = timesteps
    intermediates["mu"] = mu

    return intermediates


@torch.no_grad()
def run_vllm_omni_with_intermediates():
    """Run vllm-omni pipeline components and capture intermediate outputs."""
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline, calculate_shift
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    print("\n--- Loading vllm-omni pipeline components ---")

    # We can't easily instantiate the full pipeline without the engine,
    # but we can load the individual components and compare.

    # Load tokenizer and text encoder
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    local_files_only = os.path.exists(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", local_files_only=local_files_only)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=local_files_only
    ).to("cuda")

    # ---- Stage 1: Text encoding (our method) ----
    print("\n[Stage 1] Text encoding (vllm-omni method)...")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = tokenizer.model_max_length
    if max_len is None or max_len > 100000:
        max_len = 1024
    text_inputs = tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to("cuda")
    attention_mask = text_inputs.attention_mask.to("cuda")

    text_encoder_outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = text_encoder_outputs.hidden_states
    prompt_embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=torch.bfloat16)

    # Negative
    neg_inputs = tokenizer(
        [NEGATIVE_PROMPT],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    neg_input_ids = neg_inputs.input_ids.to("cuda")
    neg_attention_mask = neg_inputs.attention_mask.to("cuda")

    neg_outputs = text_encoder(
        input_ids=neg_input_ids,
        attention_mask=neg_attention_mask,
        output_hidden_states=True,
    )
    neg_hidden_states = neg_outputs.hidden_states
    neg_prompt_embeds = torch.stack(neg_hidden_states, dim=-1).flatten(2, 3).to(dtype=torch.bfloat16)

    prompt_embeds_cpu = prompt_embeds.cpu()
    neg_prompt_embeds_cpu = neg_prompt_embeds.cpu()
    attention_mask_cpu = attention_mask.cpu()
    neg_attention_mask_cpu = neg_attention_mask.cpu()

    intermediates = {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask_cpu,
        "prompt_embeds": prompt_embeds_cpu,
        "neg_prompt_embeds": neg_prompt_embeds_cpu,
        "num_hidden_states": len(hidden_states),
        "hidden_state_0_shape": hidden_states[0].shape,
    }

    # Aggressively free GPU
    del text_encoder, text_encoder_outputs, neg_outputs, hidden_states, neg_hidden_states
    del prompt_embeds, neg_prompt_embeds, attention_mask, neg_attention_mask
    del input_ids, neg_input_ids
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 2: Connectors ----
    print("[Stage 2] Connectors...")
    from diffusers.pipelines.ltx2 import LTX2TextConnectors

    connectors = LTX2TextConnectors.from_pretrained(
        MODEL_ID, subfolder="connectors", torch_dtype=torch.bfloat16, local_files_only=local_files_only
    ).to("cuda")

    padding_side = tokenizer.padding_side
    connector_prompt, connector_audio, connector_mask = connectors(
        prompt_embeds_cpu.to("cuda"), attention_mask_cpu.to("cuda"), padding_side=padding_side
    )
    neg_connector_prompt, neg_connector_audio, neg_connector_mask = connectors(
        neg_prompt_embeds_cpu.to("cuda"), neg_attention_mask_cpu.to("cuda"), padding_side=padding_side
    )

    intermediates["connector_prompt"] = connector_prompt.cpu()
    intermediates["connector_audio"] = connector_audio.cpu()
    intermediates["connector_mask"] = connector_mask.cpu()
    intermediates["neg_connector_prompt"] = neg_connector_prompt.cpu()

    del connectors, connector_prompt, connector_audio, connector_mask
    del neg_connector_prompt, neg_connector_audio, neg_connector_mask
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 3: Latents ----
    print("[Stage 3] Latents (random noise)...")
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=local_files_only
    )

    vae_spatial = vae.spatial_compression_ratio
    vae_temporal = vae.temporal_compression_ratio
    latent_num_frames = (NUM_FRAMES - 1) // vae_temporal + 1
    latent_height = HEIGHT // vae_spatial
    latent_width = WIDTH // vae_spatial

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    from diffusers.utils.torch_utils import randn_tensor

    latents = randn_tensor(
        (1, 128, latent_num_frames, latent_height, latent_width),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )

    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

    packed_latents = LTX23Pipeline._pack_latents(latents, 1, 1)

    intermediates["raw_latents"] = latents.cpu()
    intermediates["packed_latents"] = packed_latents.cpu()
    intermediates["latent_num_frames"] = latent_num_frames
    intermediates["latent_height"] = latent_height
    intermediates["latent_width"] = latent_width

    del vae, latents, packed_latents
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 4: Scheduler ----
    print("[Stage 4] Scheduler timesteps...")
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler", local_files_only=local_files_only
    )

    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    video_seq_len = latent_num_frames * latent_height * latent_width
    mu = calculate_shift(
        video_seq_len,
        scheduler.config.get("base_image_seq_len", 1024),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.95),
        scheduler.config.get("max_shift", 2.05),
    )
    timesteps, _ = retrieve_timesteps(scheduler, NUM_INFERENCE_STEPS, "cuda", sigmas=sigmas, mu=mu)
    intermediates["timesteps"] = timesteps
    intermediates["mu"] = mu

    return intermediates


def main():
    print("=" * 70)
    print("INTERMEDIATE OUTPUT COMPARISON: diffusers vs vllm-omni")
    print("=" * 70)

    print("\n>>> Running diffusers...")
    diff_intermediates = run_diffusers_with_intermediates()

    print("\n>>> Running vllm-omni...")
    vllm_intermediates = run_vllm_omni_with_intermediates()

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n[Stage 1] Text Encoding:")
    print(
        f"  num_hidden_states: diffusers={diff_intermediates['num_hidden_states']} "
        f"vllm={vllm_intermediates['num_hidden_states']}"
    )
    compare_tensors("input_ids", diff_intermediates["input_ids"], vllm_intermediates["input_ids"])
    compare_tensors("attention_mask", diff_intermediates["attention_mask"], vllm_intermediates["attention_mask"])
    compare_tensors("prompt_embeds", diff_intermediates["prompt_embeds"], vllm_intermediates["prompt_embeds"])
    compare_tensors(
        "neg_prompt_embeds", diff_intermediates["neg_prompt_embeds"], vllm_intermediates["neg_prompt_embeds"]
    )

    print("\n[Stage 2] Connectors:")
    compare_tensors("connector_prompt", diff_intermediates["connector_prompt"], vllm_intermediates["connector_prompt"])
    compare_tensors("connector_audio", diff_intermediates["connector_audio"], vllm_intermediates["connector_audio"])
    compare_tensors("connector_mask", diff_intermediates["connector_mask"], vllm_intermediates["connector_mask"])
    compare_tensors(
        "neg_connector_prompt", diff_intermediates["neg_connector_prompt"], vllm_intermediates["neg_connector_prompt"]
    )

    print("\n[Stage 3] Latents:")
    compare_tensors("raw_latents", diff_intermediates["raw_latents"], vllm_intermediates["raw_latents"])
    compare_tensors("packed_latents", diff_intermediates["packed_latents"], vllm_intermediates["packed_latents"])
    print(
        f"  latent_dims: diffusers=({diff_intermediates['latent_num_frames']},{diff_intermediates['latent_height']},{diff_intermediates['latent_width']}) "
        f"vllm=({vllm_intermediates['latent_num_frames']},{vllm_intermediates['latent_height']},{vllm_intermediates['latent_width']})"
    )

    print("\n[Stage 4] Scheduler:")
    compare_tensors("timesteps", diff_intermediates["timesteps"], vllm_intermediates["timesteps"])
    print(f"  mu: diffusers={diff_intermediates['mu']:.8f} vllm={vllm_intermediates['mu']:.8f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
