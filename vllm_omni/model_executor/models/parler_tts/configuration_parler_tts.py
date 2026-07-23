# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Parler-TTS config registration for AutoConfig.

The parler_tts HuggingFace checkpoint has model_type="parler_tts" but no
auto_map, so transformers cannot load the config without the parler_tts
pip package. This module registers a lightweight PretrainedConfig subclass
so that AutoConfig.from_pretrained() works before the full parler_tts
package is imported (which happens later in the model __init__).
"""

from transformers import AutoConfig, PretrainedConfig


class ParlerTTSDecoderConfig(PretrainedConfig):
    model_type = "parler_tts_decoder"


class ParlerTTSConfig(PretrainedConfig):
    model_type = "parler_tts"
    is_composition = True

    def __init__(self, vocab_size=1024, prompt_cross_attention=False, **kwargs):
        self.vocab_size = vocab_size
        self.prompt_cross_attention = prompt_cross_attention
        super().__init__(**kwargs)


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoConfig.register("parler_tts_decoder", ParlerTTSDecoderConfig)
