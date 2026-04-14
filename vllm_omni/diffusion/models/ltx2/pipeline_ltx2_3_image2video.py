# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 image-to-video pipeline.

Extends :class:`LTX23Pipeline` with image conditioning support, mirroring the
relationship between :class:`LTX2ImageToVideoPipeline` and :class:`LTX2Pipeline`.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.video_processor import VideoProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.lora.request import LoRARequest

from .pipeline_ltx2 import get_ltx2_post_process_func as _get_ltx2_post_process_func
from .pipeline_ltx2_3 import LTX23Pipeline
from .pipeline_ltx2_image2video import LTX2ImageToVideoPipeline
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

logger = init_logger(__name__)


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):
    return _get_ltx2_post_process_func(od_config)


class LTX23ImageToVideoPipeline(LTX23Pipeline):
    """LTX-2.3 image-to-video pipeline.

    Extends :class:`LTX23Pipeline` with image conditioning (``support_image_input = True``).
    Reuses the image processing and conditioning logic from :class:`LTX2ImageToVideoPipeline`.
    """

    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__(od_config=od_config, prefix=prefix)
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio,
            resample="bilinear",
        )

    # Reuse I2V forward from LTX2ImageToVideoPipeline.
    # _create_noised_state is already inherited from LTX23Pipeline -> LTX2Pipeline.
    forward = LTX2ImageToVideoPipeline.forward


class LTX23ImageToVideoTwoStagesPipeline(nn.Module):
    """LTX-2.3 two-stage image-to-video pipeline with latent upsampling."""

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

        model_basename = os.path.basename(os.path.normpath(self.model_path))
        if "distilled" in model_basename:
            self.distilled = True
        else:
            raise NotImplementedError(
                f"{self.model_path} is not supported for {self.__class__.__name__}. "
                "LTX23ImageToVideoTwoStagesPipeline requires a distilled model."
            )

        self.pipe = LTX23ImageToVideoPipeline(od_config=od_config, prefix=prefix)
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
        def _remap_weights():
            for name, tensor in weights:
                if name.startswith("pipe."):
                    inner_name = name[len("pipe.") :]
                    yield inner_name, tensor

        return self.pipe.load_weights(_remap_weights())
