# SPDX-License-Identifier: Apache-2.0
"""Parler-TTS serving adapter.

Parler-TTS uses a voice *description* string (e.g. "A female speaker with
a moderate speed") instead of reference audio for voice conditioning.
The ``voice`` field in the API request is treated as the description.
"""

from typing import TYPE_CHECKING

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import (
        OpenAICreateSpeechRequest,
    )


@register_tts_adapter
class ParlerTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"parler_tts"})
    name = "parler_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        tts_params: dict = {
            "text": [request.input],
        }
        if request.voice:
            tts_params["description"] = [request.voice]

        prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=self.name,
        )
