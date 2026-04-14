# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 (22B) pipeline for text-to-video and image-to-video generation.

LTX-2.3 is distributed as raw safetensors files (no diffusers subdirectory
structure). This pipeline handles loading from that format while reusing the
same inference logic as the LTX-2 pipeline.

Key differences from LTX-2 (19B):
  - 22B parameters (vs 19B)
  - Model config stored in safetensors file metadata (no separate config.json)
  - All component weights (transformer, VAE, audio_vae, vocoder, connectors) in
    a single safetensors file
  - Caption projection lives in connectors (``caption_proj_before_connector=True``),
    NOT in the transformer
  - Gemma3 text encoder loaded from a separate path
  - VocoderWithBWE for improved audio quality (24kHz vs 16kHz)

Usage::

    # Zero-config: text encoder auto-downloaded from Lightricks/LTX-2
    vllm serve Lightricks/LTX-2.3 --omni --model-class-name LTX23Pipeline

    # Or with explicit text encoder path
    vllm serve Lightricks/LTX-2.3 \\
        --omni \\
        --model-class-name LTX23Pipeline \\
        --custom-pipeline-args '{"text_encoder_path": "/path/to/gemma-3"}'

    # Or override the text encoder HuggingFace repo
    vllm serve Lightricks/LTX-2.3 \\
        --omni \\
        --model-class-name LTX23Pipeline \\
        --custom-pipeline-args '{"text_encoder_model": "my-org/custom-gemma3"}'
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable

import safetensors
import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.lora.request import LoRARequest

from .pipeline_ltx2 import (
    LTX2Pipeline,
    create_transformer_from_config,
    get_ltx2_post_process_func,  # noqa: F401 - loaded by registry via getattr
)
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Weight key prefix constants for the raw Lightricks safetensors format.
# Each component's weights are namespaced under a distinct prefix.
# ---------------------------------------------------------------------------
_RAW_TRANSFORMER_PREFIX = "model.diffusion_model."
_RAW_VAE_PREFIX = "vae."
_RAW_AUDIO_VAE_PREFIX = "audio_vae."
_RAW_VOCODER_PREFIX = "vocoder."
_RAW_CONNECTORS_PREFIX = "embeddings_proj."

# Default HuggingFace repo for the text encoder.  LTX-2.3 uses the same
# Gemma-3 text encoder as LTX-2, but the LTX-2.3 repo ships only the
# raw safetensors (no diffusers subdirectories).  When the user does not
# provide an explicit text_encoder_path we fall back to downloading the
# text_encoder/ and tokenizer/ subfolders from this repo automatically.
_DEFAULT_TEXT_ENCODER_REPO = "Lightricks/LTX-2"


def _load_safetensors_metadata(model_path: str) -> dict:
    """Read model configuration from safetensors file metadata.

    LTX-2.3 stores the full model config (transformer, VAE, audio_vae,
    vocoder, etc.) as a JSON string in the safetensors file header under
    the ``config`` key.

    Args:
        model_path: Path to the ``.safetensors`` file or directory
            containing safetensors files.

    Returns:
        Parsed config dict, or empty dict if metadata is unavailable.
    """
    sft_path = _resolve_safetensors_path(model_path)
    if sft_path is None:
        return {}
    with safetensors.safe_open(sft_path, framework="pt") as f:
        meta = f.metadata()
        if meta is None or "config" not in meta:
            return {}
        return json.loads(meta["config"])


def _resolve_safetensors_path(model_path: str) -> str | None:
    """Resolve a model path to the primary safetensors file.

    Handles:
    - Direct path to a ``.safetensors`` file
    - Local directory containing safetensors files
    - HuggingFace repo ID (e.g. ``Lightricks/LTX-2.3``) -- resolved to
      the local HF cache snapshot directory
    """
    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        return model_path

    # If it's not a local path, try resolving as a HuggingFace repo ID
    if not os.path.exists(model_path) and "/" in model_path:
        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(model_path, allow_patterns=["*.safetensors"])
            logger.info("Resolved HF repo %s to %s", model_path, local_dir)
            model_path = local_dir
        except Exception as e:
            logger.debug("Failed to resolve %s as HF repo: %s", model_path, e)
            return None

    if os.path.isdir(model_path):
        sft_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".safetensors")]
        if not sft_files:
            return None

        # Prefer the primary model file (dev or distilled, not LoRA/upscaler)
        for keyword in ("-dev.", "-distilled."):
            for f in sft_files:
                basename = os.path.basename(f)
                if keyword in basename and "lora" not in basename.lower() and "upscaler" not in basename.lower():
                    return f

        # Fall back to the largest safetensors file
        return max(sft_files, key=os.path.getsize)

    return None


def _load_component_weights_from_safetensors(
    sft_path: str,
    key_prefix: str,
    strip_prefix: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a subset of weights from a safetensors file, filtered by key prefix.

    Args:
        sft_path: Path to the safetensors file.
        key_prefix: Only load keys starting with this prefix.
        strip_prefix: If True, strip the prefix from key names.
        device: Device to load tensors onto.

    Returns:
        Dict mapping (optionally stripped) key names to tensors.
    """
    state_dict = {}
    with safetensors.safe_open(sft_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            if key.startswith(key_prefix):
                tensor = f.get_tensor(key)
                new_key = key[len(key_prefix) :] if strip_prefix else key
                state_dict[new_key] = tensor
    return state_dict


def _get_text_encoder_path(od_config: OmniDiffusionConfig) -> str:
    """Resolve the text encoder path, downloading automatically if needed.

    Resolution order:

    1. ``custom_pipeline_args["text_encoder_path"]`` – explicit local path
       or HuggingFace model ID provided by the user.
    2. ``custom_pipeline_args["text_encoder_model"]`` – HuggingFace repo to
       download the text encoder from (defaults to
       :data:`_DEFAULT_TEXT_ENCODER_REPO`).
    3. ``text_encoder/`` subdirectory inside the model path.
    4. Auto-download from :data:`_DEFAULT_TEXT_ENCODER_REPO`.

    Returns:
        Local path to a directory containing ``text_encoder/`` and
        ``tokenizer/`` subdirectories.
    """
    custom_args = getattr(od_config, "custom_pipeline_args", None) or {}

    # 1. Explicit local path
    text_encoder_path = custom_args.get("text_encoder_path")
    if text_encoder_path and os.path.exists(text_encoder_path):
        return text_encoder_path

    # 2. text_encoder/ already present inside the model directory
    model = od_config.model
    if os.path.isdir(model):
        te_path = os.path.join(model, "text_encoder")
        if os.path.exists(te_path):
            return model

    # 3. Explicit HF repo ID (non-local path in text_encoder_path, or
    #    the separate text_encoder_model key)
    te_repo = (
        custom_args.get("text_encoder_model")
        or text_encoder_path  # may be an HF ID like "google/gemma-3-12b"
        or _DEFAULT_TEXT_ENCODER_REPO
    )

    # 4. Download text_encoder/ and tokenizer/ from the resolved repo
    logger.info(
        "LTX-2.3: downloading text encoder from %s "
        "(override with --custom-pipeline-args "
        '\'{"text_encoder_model": "your/repo"}\')',
        te_repo,
    )
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        te_repo,
        allow_patterns=["text_encoder/*", "tokenizer/*"],
    )
    return local_dir


class LTX23Pipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    """LTX-2.3 (22B) text-to-video pipeline.

    Loads model weights from the Lightricks raw safetensors format
    (single ``.safetensors`` file with all components). Reuses the
    same denoising and decoding logic as :class:`LTX2Pipeline`.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model

        # Resolve the primary safetensors file
        sft_path = _resolve_safetensors_path(model)
        if sft_path is None:
            raise FileNotFoundError(
                f"No safetensors file found at {model}. LTX-2.3 expects either a "
                f"direct path to a .safetensors file or a directory containing one."
            )
        self._sft_path = sft_path

        # Read model config from safetensors metadata
        sft_config = _load_safetensors_metadata(model)
        transformer_config = sft_config.get("transformer", {})
        logger.info(
            "LTX-2.3: loaded config from safetensors metadata (num_layers=%s, num_attention_heads=%s)",
            transformer_config.get("num_layers", "default"),
            transformer_config.get("num_attention_heads", "default"),
        )

        # --- Transformer (loaded via weights_sources, not here) ---
        self.transformer = create_transformer_from_config(transformer_config)

        # Weight sources: load transformer weights from raw safetensors.
        # Keys are prefixed with "model.diffusion_model." in the raw file;
        # our custom load_weights() remaps them to "transformer.*".
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=os.path.dirname(sft_path) if os.path.isfile(model) else model,
                subfolder=None,
                revision=None,
                prefix="",
                fall_back_to_pt=False,
                allow_patterns_overrides=[os.path.basename(sft_path)],
            ),
        ]

        # --- Text encoder and tokenizer (from separate path) ---
        # Auto-downloads from Lightricks/LTX-2 if no explicit path is given.
        text_encoder_path = _get_text_encoder_path(od_config)
        te_local = os.path.exists(text_encoder_path)

        # Determine if text encoder is in a subfolder or standalone
        te_has_subfolder = os.path.isdir(os.path.join(text_encoder_path, "text_encoder"))
        te_subfolder = "text_encoder" if te_has_subfolder else None
        tok_subfolder = "tokenizer" if os.path.isdir(os.path.join(text_encoder_path, "tokenizer")) else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_path,
            subfolder=tok_subfolder,
            local_files_only=te_local,
        )
        with torch.device("cpu"):
            self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                text_encoder_path,
                subfolder=te_subfolder,
                torch_dtype=dtype,
                local_files_only=te_local,
            ).to(self.device)

        # --- Load non-transformer components from raw safetensors ---
        # VAE (video)
        vae_sd = _load_component_weights_from_safetensors(sft_path, _RAW_VAE_PREFIX, device="cpu")
        if vae_sd:
            self.vae = AutoencoderKLLTX2Video()
            self.vae.load_state_dict(vae_sd, strict=False)
            self.vae = self.vae.to(device=self.device, dtype=dtype)
            logger.info("LTX-2.3: loaded video VAE (%d params)", len(vae_sd))
        else:
            logger.warning("LTX-2.3: no video VAE weights found; falling back to defaults")
            self.vae = AutoencoderKLLTX2Video().to(device=self.device, dtype=dtype)

        # Audio VAE
        audio_vae_sd = _load_component_weights_from_safetensors(sft_path, _RAW_AUDIO_VAE_PREFIX, device="cpu")
        if audio_vae_sd:
            self.audio_vae = AutoencoderKLLTX2Audio()
            self.audio_vae.load_state_dict(audio_vae_sd, strict=False)
            self.audio_vae = self.audio_vae.to(device=self.device, dtype=dtype)
            logger.info("LTX-2.3: loaded audio VAE (%d params)", len(audio_vae_sd))
        else:
            logger.warning("LTX-2.3: no audio VAE weights found; falling back to defaults")
            self.audio_vae = AutoencoderKLLTX2Audio().to(device=self.device, dtype=dtype)

        # Vocoder
        vocoder_sd = _load_component_weights_from_safetensors(sft_path, _RAW_VOCODER_PREFIX, device="cpu")
        if vocoder_sd:
            self.vocoder = LTX2Vocoder()
            self.vocoder.load_state_dict(vocoder_sd, strict=False)
            self.vocoder = self.vocoder.to(device=self.device, dtype=dtype)
            logger.info("LTX-2.3: loaded vocoder (%d params)", len(vocoder_sd))
        else:
            logger.warning("LTX-2.3: no vocoder weights found; falling back to defaults")
            self.vocoder = LTX2Vocoder().to(device=self.device, dtype=dtype)

        # Connectors (embeddings processor)
        connectors_sd = _load_component_weights_from_safetensors(sft_path, _RAW_CONNECTORS_PREFIX, device="cpu")
        if connectors_sd:
            self.connectors = LTX2TextConnectors()
            self.connectors.load_state_dict(connectors_sd, strict=False)
            self.connectors = self.connectors.to(device=self.device, dtype=dtype)
            logger.info("LTX-2.3: loaded connectors (%d params)", len(connectors_sd))
        else:
            logger.warning("LTX-2.3: no connector weights found; falling back to defaults")
            self.connectors = LTX2TextConnectors().to(device=self.device, dtype=dtype)

        # Scheduler (no weights, just configuration)
        self.scheduler = FlowMatchEulerDiscreteScheduler()

        # --- Derived attributes (same as LTX2Pipeline) ---
        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer", None) is not None else 1
        )
        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        )

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

    # Reuse all inference methods from LTX2Pipeline
    _pack_text_embeds = LTX2Pipeline._pack_text_embeds
    _get_gemma_prompt_embeds = LTX2Pipeline._get_gemma_prompt_embeds
    encode_prompt = LTX2Pipeline.encode_prompt
    prepare_latents = LTX2Pipeline.prepare_latents
    prepare_audio_latents = LTX2Pipeline.prepare_audio_latents
    _pack_latents = LTX2Pipeline._pack_latents
    _unpack_latents = LTX2Pipeline._unpack_latents
    _normalize_latents = LTX2Pipeline._normalize_latents
    _denormalize_latents = LTX2Pipeline._denormalize_latents
    _create_noised_state = LTX2Pipeline._create_noised_state

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # forward() is inherited from LTX2Pipeline via method reference
    forward = LTX2Pipeline.forward

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with key remapping for the raw Lightricks format.

        The raw safetensors file uses ``model.diffusion_model.*`` prefix for
        transformer weights. This method strips that prefix and remaps to
        ``transformer.*`` so that :class:`AutoWeightsLoader` can route them
        to ``self.transformer``.

        Non-transformer weights (VAE, audio_vae, vocoder, connectors) are
        already loaded eagerly during ``__init__``, so they are skipped here.
        """

        def _remap_weights():
            for name, tensor in weights:
                if name.startswith(_RAW_TRANSFORMER_PREFIX):
                    # model.diffusion_model.X -> transformer.X
                    new_name = "transformer." + name[len(_RAW_TRANSFORMER_PREFIX) :]
                    yield new_name, tensor
                # Skip non-transformer weights (already loaded in __init__)

        loader = AutoWeightsLoader(self)
        return loader.load_weights(_remap_weights())


class LTX23TwoStagesPipeline(nn.Module):
    """LTX-2.3 two-stage pipeline with latent upsampling.

    Stage 1: Generate low-resolution video + audio
    Stage 2: Upsample latents and refine with distilled LoRA
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.model_path = od_config.model
        self.distilled = False

        # Detect distilled model
        model_basename = os.path.basename(os.path.normpath(self.model_path))
        if "distilled" in model_basename:
            self.distilled = True
        else:
            raise NotImplementedError(
                f"{self.model_path} is not supported for {self.__class__.__name__}. "
                "LTX23TwoStagesPipeline requires a distilled model."
            )

        self.pipe = LTX23Pipeline(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipe.vae,
            od_config=od_config,
        )

        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.pipe,
            device=self.device,
            dtype=self.dtype,
            max_cached_adapters=od_config.max_cpu_loras,
        )

        # Reuse the same weights_sources as the inner pipeline
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=(
                    os.path.dirname(self.pipe._sft_path) if os.path.isfile(od_config.model) else od_config.model
                ),
                subfolder=None,
                revision=None,
                prefix="pipe.",
                fall_back_to_pt=False,
                allow_patterns_overrides=[os.path.basename(self.pipe._sft_path)],
            ),
        ]

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt=None,
        negative_prompt=None,
        height=None,
        width=None,
        num_frames=None,
        frame_rate=None,
        num_inference_steps=None,
        timesteps=None,
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
        video_latent, audio_latent = self.pipe(
            req=req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            sigmas=DISTILLED_SIGMA_VALUES if self.distilled else None,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            noise_scale=noise_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type="latent",
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
        ).output

        upscaled_video_latent = self.upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        if not self.distilled:
            # Load Stage 2 distilled LoRA (LTX-2.3 uses 22b naming)
            lora_path = os.path.join(
                self.model_path,
                "ltx-2.3-22b-distilled-lora-384.safetensors",
            )
            lora_request = LoRARequest(
                lora_name="stage_2_distilled",
                lora_int_id=1,
                lora_path=lora_path,
            )
            self.lora_manager.set_active_adapter(lora_request, lora_scale=1.0)

            new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )
            self.pipe.scheduler = new_scheduler

        stage_2_req = copy.copy(req)
        stage_2_req.sampling_params = req.sampling_params.clone()
        stage_2_req.sampling_params.num_inference_steps = 3

        video, audio = self.pipe(
            req=stage_2_req,
            latents=upscaled_video_latent,
            audio_latents=audio_latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_dict=False,
        ).output

        return DiffusionOutput(output=(video, audio))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route raw safetensors weights through to inner pipeline."""

        def _remap_weights():
            for name, tensor in weights:
                if name.startswith("pipe."):
                    # Strip "pipe." prefix added by ComponentSource, then let
                    # inner pipeline's load_weights handle the rest
                    inner_name = name[len("pipe.") :]
                    yield inner_name, tensor

        return self.pipe.load_weights(_remap_weights())
