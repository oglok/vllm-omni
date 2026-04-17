# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 (22B) pipeline for text-to-video generation.

Extends :class:`LTX2Pipeline` with LTX-2.3-specific behavior:

1. **Diffusers transformer**: Uses the diffusers ``LTX2VideoTransformer3DModel``
   which has the full LTX-2.3 forward pass (cross-attn adaln, prompt modulation,
   perturbed attention / STG).  The vllm-omni custom fork has TP/SP but lacks
   these features.

2. **CPU offloading**: Keeps text encoder, connectors, VAE, audio VAE, and
   vocoder on CPU.  Moves them to GPU only when needed (encoding / decoding),
   because the 22B transformer alone uses ~44GB of the 96GB VRAM budget.

3. **VocoderWithBWE**: Uses ``LTX2VocoderWithBWE`` for 48kHz audio output
   (bandwidth extension from 16kHz).

4. **Connector call**: Passes raw ``prompt_attention_mask`` + ``padding_side``
   instead of a pre-computed additive mask (diffusers 0.38.0+ API).

5. **Sigma parameter**: Passes the current timestep as ``sigma`` to the
   transformer for the ``prompt_adaln`` computation.

Usage::

    pip install git+https://github.com/huggingface/diffusers.git  # >= 0.38.0
    vllm serve dg845/LTX-2.3-Diffusers --omni --model-class-name LTX23Pipeline
"""

from __future__ import annotations

import json
import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig

from .pipeline_ltx2 import (
    LTX2Pipeline,
    LTX2TwoStagesPipeline,
    load_transformer_config,
)
from .pipeline_ltx2_image2video import (
    LTX2ImageToVideoTwoStagesPipeline,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Diffusers transformer (full LTX-2.3 forward pass support)
# ---------------------------------------------------------------------------
try:
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel as DiffusersLTX2Transformer,
    )
except ImportError:
    DiffusersLTX2Transformer = None

# BWE vocoder for 48kHz audio
try:
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
except ImportError:
    LTX2VocoderWithBWE = None


# ---------------------------------------------------------------------------
# Audio sample rate detection for post-processing
# ---------------------------------------------------------------------------
def _detect_vocoder_output_sample_rate(model: str) -> int | None:
    """Read ``output_sampling_rate`` from the vocoder config.

    Returns 48000 for LTX-2.3 (BWE vocoder), None for LTX-2.
    """
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
class LTX23Pipeline(LTX2Pipeline):
    """LTX-2.3 text-to-video pipeline.

    Inherits the full denoising loop from :class:`LTX2Pipeline` and overrides
    only what differs for LTX-2.3.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Skip LTX2Pipeline.__init__ and call nn.Module.__init__ directly,
        # because we need to change how components are loaded and placed.
        from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
        from diffusers.pipelines.ltx2 import LTX2TextConnectors
        from diffusers.video_processor import VideoProcessor
        from torch import nn
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = self._get_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)

        # --- Weight sources (transformer loaded via DiffusersPipelineLoader) ---
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

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

        # --- Text encoder: CPU (moved to GPU temporarily during encoding) ---
        with torch.device("cpu"):
            self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
            )

        # --- Connectors: CPU (moved to GPU on first use) ---
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

        # --- Transformer: diffusers version with full LTX-2.3 forward ---
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

        # --- Derived attributes (same as LTX2Pipeline) ---
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
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_max_length = self.tokenizer.model_max_length
            if tokenizer_max_length is None or tokenizer_max_length > 100000:
                encoder_config = getattr(self.text_encoder, "config", None)
                config_max_len = getattr(encoder_config, "max_position_embeddings", None)
                if config_max_len is None:
                    config_max_len = getattr(encoder_config, "max_seq_len", None)
                tokenizer_max_length = config_max_len or 1024
        self.tokenizer_max_length = int(tokenizer_max_length)

        self._guidance_scale = None
        self._guidance_rescale = None
        self._attention_kwargs = None
        self._interrupt = False
        self._num_timesteps = None
        self._current_timestep = None

    @staticmethod
    def _get_device():
        from vllm_omni.diffusion.distributed.utils import get_local_device

        return get_local_device()

    # ------------------------------------------------------------------
    # Override: text encoder CPU <-> GPU offloading
    # ------------------------------------------------------------------
    def _get_gemma_prompt_embeds(
        self, prompt, num_videos_per_prompt=1, max_sequence_length=1024, scale_factor=8, device=None, dtype=None
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        # Move text encoder to GPU, run, move back
        self.text_encoder.to(device)
        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=device,
            padding_side=self.tokenizer.padding_side,
            scale_factor=scale_factor,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        _, seq_len, _ = prompt_embeds.shape
        prompt_attention_mask = prompt_attention_mask[:, :seq_len]
        return prompt_embeds, prompt_attention_mask

    # ------------------------------------------------------------------
    # Override: connector call (padding_side instead of additive_mask)
    #           + sigma parameter for transformer
    #           + CPU offload for VAE/vocoder during decode
    # ------------------------------------------------------------------
    def forward(
        self,
        req=None,
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
        # --- Resolve parameters from request (same as parent) ---
        import copy

        import numpy as np
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
        from diffusers.utils.torch_utils import randn_tensor

        device = self.device
        prompt, negative_prompt, height, width, num_frames, frame_rate, num_inference_steps = (
            self._resolve_request_params(
                req, prompt, negative_prompt, height, width, num_frames, frame_rate, num_inference_steps
            )
        )

        batch_size = 1
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        max_sequence_length = max_sequence_length or self.tokenizer_max_length

        # --- Encode prompts (uses our overridden _get_gemma_prompt_embeds) ---
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # --- Connectors: padding_side API (diffusers 0.38.0+) ---
        tokenizer_padding_side = "left"
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")

        if not self._connectors_on_device:
            self.connectors.to(device)
            self._connectors_on_device = True

        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=tokenizer_padding_side
        )

        negative_connector_prompt_embeds = None
        negative_connector_audio_prompt_embeds = None
        negative_connector_attention_mask = None
        if self.do_classifier_free_guidance:
            (
                negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask,
            ) = self.connectors(
                negative_prompt_embeds, negative_prompt_attention_mask, padding_side=tokenizer_padding_side
            )

        # --- Prepare latents (same as parent) ---
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents,
        )

        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)

        num_mel_bins = getattr(getattr(self.audio_vae, "config", None), "mel_bins", 64)
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        audio_channels = getattr(self.transformer.config, "audio_in_channels", 64)

        audio_latents, original_audio_num_frames, padded_audio_num_frames = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            audio_channels,
            num_mel_bins,
            audio_latent_length=audio_num_frames,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        # --- Denoising loop ---
        video_sequence_length = latent_num_frames * latent_height * latent_width
        from .pipeline_ltx2 import _VideoAudioScheduler, calculate_shift

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        video_audio_scheduler = _VideoAudioScheduler(self.scheduler, audio_scheduler)
        _ = retrieve_timesteps(audio_scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], padded_audio_num_frames, audio_latents.device
        )

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents.to(prompt_embeds.dtype)
                audio_latent_model_input = audio_latents.to(prompt_embeds.dtype)
                timestep = t.expand(latent_model_input.shape[0])
                do_true_cfg = self.do_classifier_free_guidance

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "audio_hidden_states": audio_latent_model_input,
                    "encoder_hidden_states": connector_prompt_embeds,
                    "audio_encoder_hidden_states": connector_audio_prompt_embeds,
                    "timestep": timestep,
                    "sigma": t.expand(latent_model_input.shape[0]),  # LTX-2.3: needed for prompt_adaln
                    "encoder_attention_mask": connector_attention_mask,
                    "audio_encoder_attention_mask": connector_attention_mask,
                    "num_frames": latent_num_frames,
                    "height": latent_height,
                    "width": latent_width,
                    "fps": frame_rate,
                    "audio_num_frames": padded_audio_num_frames,
                    "video_coords": video_coords,
                    "audio_coords": audio_coords,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }

                negative_kwargs = None
                if do_true_cfg:
                    negative_kwargs = {
                        **positive_kwargs,
                        "encoder_hidden_states": negative_connector_prompt_embeds,
                        "audio_encoder_hidden_states": negative_connector_audio_prompt_embeds,
                        "encoder_attention_mask": negative_connector_attention_mask,
                        "audio_encoder_attention_mask": negative_connector_attention_mask,
                    }

                noise_pred_video, noise_pred_audio = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents, audio_latents = self.scheduler_step_maybe_with_cfg(
                    (noise_pred_video, noise_pred_audio),
                    (t, t),
                    (latents, audio_latents),
                    do_true_cfg=do_true_cfg,
                    per_request_scheduler=video_audio_scheduler,
                )
                latents, audio_latents = self._synchronize_cfg_parallel_step_output(
                    (latents, audio_latents),
                    do_true_cfg=do_true_cfg,
                )
                pbar.update()

        # --- Decode ---
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
        audio_latents = self._unpad_audio_latents(audio_latents, original_audio_num_frames)
        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(
            audio_latents, original_audio_num_frames, num_mel_bins=latent_mel_bins
        )

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
            generated_mel = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = self.vocoder(generated_mel)

            # Move back to CPU
            self.vae.to("cpu")
            self.audio_vae.to("cpu")
            self.vocoder.to("cpu")
            torch.cuda.empty_cache()

        if not return_dict:
            return DiffusionOutput(output=(video, audio))
        return DiffusionOutput(output=(video, audio))

    def _resolve_request_params(
        self, req, prompt, negative_prompt, height, width, num_frames, frame_rate, num_inference_steps
    ):
        """Extract parameters from OmniDiffusionRequest, same logic as LTX2Pipeline."""
        if req is not None:
            sp = req.sampling_params
            # req.prompts is a list of str or dict with "prompt"/"negative_prompt" keys
            prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
            if not all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
                negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]
            height = sp.height or height or 512
            width = sp.width or width or 768
            num_frames = sp.num_frames or num_frames or 121
            frame_rate = sp.fps or frame_rate or 24
            num_inference_steps = sp.num_inference_steps or num_inference_steps or 50
        else:
            height = height or 512
            width = width or 768
            num_frames = num_frames or 121
            frame_rate = frame_rate or 24
            num_inference_steps = num_inference_steps or 50
        return prompt, negative_prompt, height, width, num_frames, frame_rate, num_inference_steps


# ---------------------------------------------------------------------------
# Two-stage and I2V aliases
# ---------------------------------------------------------------------------
class LTX23TwoStagesPipeline(LTX2TwoStagesPipeline):
    """LTX-2.3 two-stage pipeline. Uses LTX23Pipeline as the inner pipeline."""

    pass


class LTX23ImageToVideoPipeline(LTX23Pipeline):
    """LTX-2.3 image-to-video pipeline.

    Inherits LTX-2.3 overrides (CPU offload, BWE vocoder, diffusers transformer,
    padding_side connectors).  Uses T2V forward from :class:`LTX23Pipeline`.

    .. note::
        Image conditioning (encoding a reference image into the initial latents)
        requires a dedicated ``forward()`` override that is not yet implemented.
        Currently this pipeline generates video from text only, like
        :class:`LTX23Pipeline`.  A proper I2V forward should be added by
        porting the image encoding logic from :class:`LTX2ImageToVideoPipeline`
        with the LTX-2.3 connector API.
    """

    support_image_input = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, resample="bilinear")


class LTX23ImageToVideoTwoStagesPipeline(LTX2ImageToVideoTwoStagesPipeline):
    """LTX-2.3 two-stage image-to-video pipeline."""

    pass
