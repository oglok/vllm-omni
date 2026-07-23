# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parler-TTS pipeline topology (frozen).

Single-stage AR TTS: text description + prompt text -> speech waveform in one
pass.  The T5 text encoder, decoder, and DAC audio codec all run inside
``ParlerTTSForGeneration.forward()``, which uses the VoxCPM-style generator
pattern (one audio chunk yielded per forward call) to drive progressive
streaming through the AR scheduler.
"""

import vllm_omni.model_executor.models.parler_tts.configuration_parler_tts  # noqa: F401 — register AutoConfig
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

PARLER_TTS_PIPELINE = PipelineConfig(
    model_type="parler_tts",
    model_arch="ParlerTTSForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="parler_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2],
            },
        ),
    ),
)
