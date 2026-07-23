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
    sub_configs = {"text_encoder": PretrainedConfig, "decoder": ParlerTTSDecoderConfig}

    def __init__(self, vocab_size=1024, prompt_cross_attention=False, **kwargs):
        self.vocab_size = vocab_size
        self.prompt_cross_attention = prompt_cross_attention

        for attr, cfg_cls in (
            ("text_encoder", PretrainedConfig),
            ("decoder", ParlerTTSDecoderConfig),
            ("audio_encoder", PretrainedConfig),
        ):
            sub = kwargs.pop(attr, None)
            if isinstance(sub, dict):
                setattr(self, attr, cfg_cls.from_dict(sub))
            elif sub is not None:
                setattr(self, attr, sub)
            elif not hasattr(self, attr):
                setattr(self, attr, cfg_cls())

        super().__init__(**kwargs)

    def get_text_config(self, **kwargs):
        return self.decoder


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoConfig.register("parler_tts_decoder", ParlerTTSDecoderConfig)
