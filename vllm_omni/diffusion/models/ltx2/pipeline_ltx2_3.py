# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 (22B) fully independent pipeline for text-to-video generation.

This pipeline is self-contained and does NOT inherit from LTX2Pipeline.
It follows the diffusers ``LTX2Pipeline.__call__`` logic directly, adapted
for vllm-omni's serving framework with CPU offloading for the text encoder
and decode components.

Requires ``diffusers >= 0.38.0`` for LTX-2.3 component support.

Usage::

    pip install git+https://github.com/huggingface/diffusers.git
    vllm serve dg845/LTX-2.3-Diffusers --omni --model-class-name LTX23Pipeline
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

# BWE vocoder for 48kHz audio (LTX-2.3)
try:
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
except ImportError:
    LTX2VocoderWithBWE = None

# Prefer diffusers transformer which has gated attention, split RoPE, and the
# full LTX-2.3 forward pass. The custom vllm-omni transformer has TP/SP but
# lacks gated attention (to_gate_logits) which LTX-2.3 requires.
# TODO: add gated attention to custom transformer, then switch back.
try:
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel as DiffusersLTX2Transformer,
    )
except ImportError:
    DiffusersLTX2Transformer = None

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: shift schedule
# ---------------------------------------------------------------------------
def _calculate_shift(image_seq_len, base_seq_len=1024, max_seq_len=4096, base_shift=0.95, max_shift=2.05):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# ---------------------------------------------------------------------------
# Post-process function (with audio sample rate for BWE vocoder)
# ---------------------------------------------------------------------------
def _detect_vocoder_output_sample_rate(model: str) -> int | None:
    try:
        from huggingface_hub import hf_hub_download

        vocoder_config_path = os.path.join(model, "vocoder", "config.json")
        if not os.path.exists(vocoder_config_path):
            vocoder_config_path = hf_hub_download(model, "vocoder/config.json")
        with open(vocoder_config_path) as f:
            return json.load(f).get("output_sampling_rate")
    except Exception:
        return None


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):  # noqa: F811
    """Post-process function that includes ``audio_sample_rate`` for BWE vocoder."""
    output_sr = _detect_vocoder_output_sample_rate(od_config.model)

    def post_process_func(output):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            result: dict = {"video": video, "audio": audio}
            if output_sr is not None:
                result["audio_sample_rate"] = output_sr
            return result
        return output

    return post_process_func


# ---------------------------------------------------------------------------
# LTX23Pipeline
# ---------------------------------------------------------------------------
class LTX23Pipeline(nn.Module, ProgressBarMixin):
    """Fully independent LTX-2.3 text-to-video pipeline.

    Does NOT inherit from LTX2Pipeline. Uses diffusers components and logic
    directly, with CPU offloading for text encoder, VAE, and vocoder.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)

        # --- Weight sources (transformer loaded via DiffusersPipelineLoader) ---
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        # --- Text encoder: stays on CPU, moved to GPU temporarily during encoding ---
        with torch.device("cpu"):
            self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
            )

        # --- Connectors: stays on CPU, moved to GPU during forward ---
        self.connectors = LTX2TextConnectors.from_pretrained(
            model, subfolder="connectors", torch_dtype=dtype, local_files_only=local_files_only
        )
        self._connectors_on_device = False

        # --- VAE / audio VAE: CPU (moved to GPU during decode) ---
        self.vae = AutoencoderKLLTX2Video.from_pretrained(
            model, subfolder="vae", torch_dtype=dtype, local_files_only=local_files_only
        )
        self.audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            model, subfolder="audio_vae", torch_dtype=dtype, local_files_only=local_files_only
        )

        # --- Vocoder: BWE variant for 48kHz, CPU ---
        vocoder_cls = LTX2VocoderWithBWE or LTX2Vocoder
        try:
            self.vocoder = vocoder_cls.from_pretrained(
                model, subfolder="vocoder", torch_dtype=dtype, local_files_only=local_files_only
            )
        except (TypeError, OSError, ValueError):
            self.vocoder = LTX2Vocoder.from_pretrained(
                model, subfolder="vocoder", torch_dtype=dtype, local_files_only=local_files_only
            )

        # --- Transformer: diffusers version with gated attention + split RoPE ---
        from .pipeline_ltx2 import load_transformer_config

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        if DiffusersLTX2Transformer is not None:
            kwargs = {k: v for k, v in transformer_config.items() if k != "_class_name"}
            self.transformer = DiffusersLTX2Transformer(**kwargs)
        else:
            from .pipeline_ltx2 import create_transformer_from_config

            self.transformer = create_transformer_from_config(transformer_config)

        # --- Scheduler ---
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # --- Derived attributes ---
        self.vae_spatial_compression_ratio = getattr(self.vae, "spatial_compression_ratio", 32)
        self.vae_temporal_compression_ratio = getattr(self.vae, "temporal_compression_ratio", 8)
        self.audio_vae_mel_compression_ratio = getattr(self.audio_vae, "mel_compression_ratio", 4)
        self.audio_vae_temporal_compression_ratio = getattr(self.audio_vae, "temporal_compression_ratio", 4)
        self.transformer_spatial_patch_size = getattr(getattr(self.transformer, "config", None), "patch_size", 1)
        self.transformer_temporal_patch_size = getattr(getattr(self.transformer, "config", None), "patch_size_t", 1)
        self.audio_sampling_rate = getattr(getattr(self.audio_vae, "config", None), "sample_rate", 16000)
        self.audio_hop_length = getattr(getattr(self.audio_vae, "config", None), "mel_hop_length", 160)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

        tokenizer_max_length = 1024
        if self.tokenizer is not None:
            ml = self.tokenizer.model_max_length
            if ml is None or ml > 100000:
                enc_cfg = getattr(self.text_encoder, "config", None)
                ml = getattr(enc_cfg, "max_position_embeddings", None) or getattr(enc_cfg, "max_seq_len", None) or 1024
            tokenizer_max_length = int(ml)
        self.tokenizer_max_length = tokenizer_max_length

        self._guidance_scale = None
        self._guidance_rescale = None
        self._interrupt = False
        self._num_timesteps = None
        self._current_timestep = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # ------------------------------------------------------------------
    # Text encoding (diffusers approach: flatten all hidden states)
    # ------------------------------------------------------------------
    def _get_gemma_prompt_embeds(self, prompt, device=None, dtype=None, max_sequence_length=1024):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        len(prompt)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text_inputs = self.tokenizer(
            [p.strip() for p in prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Move text encoder to GPU, run, move back
        self.text_encoder.to(device)
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        # Stack all hidden states and flatten: [B, seq, hidden, layers] -> [B, seq, hidden*layers]
        prompt_embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=dtype)
        return prompt_embeds, attention_mask

    def encode_prompt(self, prompt, negative_prompt=None, do_cfg=False, device=None, max_seq_len=1024):
        device = device or self.device

        prompt_embeds, prompt_mask = self._get_gemma_prompt_embeds(
            prompt, device=device, max_sequence_length=max_seq_len
        )

        negative_embeds = None
        negative_mask = None
        if do_cfg:
            neg = negative_prompt or ""
            if isinstance(neg, str):
                neg = [neg] * (len(prompt) if isinstance(prompt, list) else 1)
            negative_embeds, negative_mask = self._get_gemma_prompt_embeds(
                neg, device=device, max_sequence_length=max_seq_len
            )

        return prompt_embeds, prompt_mask, negative_embeds, negative_mask

    # ------------------------------------------------------------------
    # Latent helpers (copied from diffusers LTX2Pipeline -- static methods)
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_latents(latents, patch_size=1, patch_size_t=1):
        B, C, F, H, W = latents.shape
        pF, pH, pW = F // patch_size_t, H // patch_size, W // patch_size
        latents = latents.reshape(B, -1, pF, patch_size_t, pH, patch_size, pW, patch_size)
        return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)

    @staticmethod
    def _unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
        B = latents.size(0)
        latents = latents.reshape(B, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)

    @staticmethod
    def _normalize_latents(latents, mean, std, scaling_factor=1.0):
        mean = mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        return (latents - mean) * scaling_factor / std

    @staticmethod
    def _denormalize_latents(latents, mean, std, scaling_factor=1.0):
        mean = mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        return latents * std / scaling_factor + mean

    @staticmethod
    def _normalize_audio_latents(latents, mean, std):
        return (latents - mean.to(latents.device, latents.dtype)) / std.to(latents.device, latents.dtype)

    @staticmethod
    def _denormalize_audio_latents(latents, mean, std):
        return (latents * std.to(latents.device, latents.dtype)) + mean.to(latents.device, latents.dtype)

    @staticmethod
    def _pack_audio_latents(latents):
        # [B, C, L, M] -> [B, L, C*M]
        return latents.transpose(1, 2).flatten(2, 3)

    @staticmethod
    def _unpack_audio_latents(latents, latent_length, num_mel_bins):
        # [B, L, C*M] -> [B, C, L, M]
        return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)

    # ------------------------------------------------------------------
    # Forward (main inference loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest | None = None,
        prompt=None,
        negative_prompt=None,
        height=None,
        width=None,
        num_frames=None,
        frame_rate=None,
        num_inference_steps=None,
        timesteps=None,
        sigmas=None,
        guidance_scale=4.0,
        guidance_rescale=0.0,
        noise_scale=0.0,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        return_dict=True,
        attention_kwargs=None,
        max_sequence_length=None,
    ):
        device = self.device

        # --- Extract params from request ---
        if req is not None:
            sp = req.sampling_params
            prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
            if not all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
                negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]
            height = sp.height or height or 512
            width = sp.width or width or 768
            num_frames = sp.num_frames or num_frames or 121
            frame_rate = sp.fps or frame_rate or 24
            num_inference_steps = sp.num_inference_steps or num_inference_steps or 30
        else:
            height = height or 512
            width = width or 768
            num_frames = num_frames or 121
            frame_rate = frame_rate or 24
            num_inference_steps = num_inference_steps or 30

        batch_size = 1
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False
        max_sequence_length = max_sequence_length or self.tokenizer_max_length

        # --- 1. Encode prompts ---
        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_cfg=self.do_classifier_free_guidance,
            device=device,
            max_seq_len=max_sequence_length,
        )

        # --- 2. Connectors ---
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)

        if not self._connectors_on_device:
            self.connectors.to(device)
            self._connectors_on_device = True

        tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
        connector_video, connector_audio, connector_mask = self.connectors(
            prompt_embeds, prompt_mask, padding_side=tokenizer_padding_side
        )

        # --- 3. Prepare video latents ---
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        num_channels = self.transformer.config.in_channels

        if latents is None:
            shape = (
                batch_size,
                num_channels,
                (num_frames - 1) // self.vae_temporal_compression_ratio + 1,
                height // self.vae_spatial_compression_ratio,
                width // self.vae_spatial_compression_ratio,
            )
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
            latents = self._pack_latents(
                latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )

        # --- 4. Prepare audio latents ---
        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        num_mel_bins = getattr(getattr(self.audio_vae, "config", None), "mel_bins", 64)
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        audio_channels = getattr(getattr(self.audio_vae, "config", None), "latent_channels", 8)

        if audio_latents is None:
            audio_shape = (batch_size, audio_channels, audio_num_frames, latent_mel_bins)
            audio_latents = randn_tensor(audio_shape, generator=generator, device=device, dtype=torch.float32)
            audio_latents = self._pack_audio_latents(audio_latents)

        # --- 5. Prepare timesteps ---
        # Ensure at least 2 steps (1 step causes scheduler index errors with flow matching)
        num_inference_steps = max(num_inference_steps, 2)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        video_seq_len = latent_num_frames * latent_height * latent_width
        mu = _calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        _ = retrieve_timesteps(audio_scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # --- 6. Prepare coords ---
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device
        )
        if self.do_classifier_free_guidance:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        # --- 7. Denoising loop ---
        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                latent_input = (
                    torch.cat([latents] * 2).to(prompt_embeds.dtype)
                    if self.do_classifier_free_guidance
                    else latents.to(prompt_embeds.dtype)
                )
                audio_input = (
                    torch.cat([audio_latents] * 2).to(prompt_embeds.dtype)
                    if self.do_classifier_free_guidance
                    else audio_latents.to(prompt_embeds.dtype)
                )
                ts = t.expand(latent_input.shape[0])

                noise_pred_video, noise_pred_audio = self.transformer(
                    hidden_states=latent_input,
                    audio_hidden_states=audio_input,
                    encoder_hidden_states=connector_video,
                    audio_encoder_hidden_states=connector_audio,
                    timestep=ts,
                    sigma=ts,
                    encoder_attention_mask=connector_mask,
                    audio_encoder_attention_mask=connector_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )
                noise_pred_video = noise_pred_video.float()
                noise_pred_audio = noise_pred_audio.float()

                if self.do_classifier_free_guidance:
                    v_uncond, v_text = noise_pred_video.chunk(2)
                    noise_pred_video = v_uncond + guidance_scale * (v_text - v_uncond)
                    a_uncond, a_text = noise_pred_audio.chunk(2)
                    noise_pred_audio = a_uncond + guidance_scale * (a_text - a_uncond)
                    if guidance_rescale > 0:
                        noise_pred_video = rescale_noise_cfg(
                            noise_pred_video, v_text, guidance_rescale=guidance_rescale
                        )
                        noise_pred_audio = rescale_noise_cfg(
                            noise_pred_audio, a_text, guidance_rescale=guidance_rescale
                        )

                latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
                audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]
                pbar.update()

        # --- 8. Decode ---
        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )

        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)

        if output_type == "latent":
            video = latents
            audio = audio_latents
        else:
            # Move decode components to GPU
            self.vae.to(device)
            self.audio_vae.to(device)
            self.vocoder.to(device)

            latents = latents.to(self.vae.dtype)
            if not self.vae.config.timestep_conditioning:
                ts = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size
                ts = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                dns = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[:, None, None, None, None]
                latents = (1 - dns) * latents + dns * noise

            video = self.vae.decode(latents, ts, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            audio_latents = audio_latents.to(self.audio_vae.dtype)
            mel = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = self.vocoder(mel)

            # Move back to CPU
            self.vae.to("cpu")
            self.audio_vae.to("cpu")
            self.vocoder.to("cpu")
            torch.cuda.empty_cache()

        return DiffusionOutput(output=(video, audio))

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


# ---------------------------------------------------------------------------
# Two-stage and I2V stubs
# ---------------------------------------------------------------------------
class LTX23TwoStagesPipeline(LTX23Pipeline):
    """LTX-2.3 two-stage pipeline (placeholder)."""

    pass


class LTX23ImageToVideoPipeline(LTX23Pipeline):
    """LTX-2.3 image-to-video pipeline (placeholder -- currently T2V only)."""

    support_image_input = True
    pass


class LTX23ImageToVideoTwoStagesPipeline(LTX23Pipeline):
    """LTX-2.3 two-stage I2V pipeline (placeholder)."""

    support_image_input = True
    pass
