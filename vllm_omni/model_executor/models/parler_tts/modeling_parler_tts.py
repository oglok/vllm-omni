# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parler-TTS single-stage model for vLLM-Omni.

Runs in a single AR worker stage.  The T5 text encoder, autoregressive
decoder, and DAC audio codec are all loaded here.

Architecture: encoder-decoder (seq2seq).
  - Text encoder: Flan-T5-XL (2048-dim, 24 layers) — encodes voice description
  - Decoder: 25 layers, 1536 hidden, 24 heads, 9 codebooks, vocab 1088
  - Audio codec: DAC 44kHz 8kbps, 9 codebooks, codebook size 1024, frame rate 86

The model takes two text inputs:
  - ``description``: voice characteristics (speaker, style, recording quality)
  - ``text``: the text to speak

Streaming is supported via the VoxCPM-style generator pattern:
  - On first forward() for a request, generate() runs the full model and the
    audio is chunked for streaming.
  - Each subsequent forward() call pops one audio chunk from the generator
    and returns it as multimodal_outputs.
  - compute_logits() emits EOS only when the last chunk has been yielded.

Delta output semantics: each forward() yields only new audio samples (not
the full waveform re-decoded from step 0).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def _patch_parler_tts_for_transformers5() -> None:
    """Apply compatibility patches for parler_tts with transformers >= 5.x.

    The upstream parler_tts package pins transformers==4.46.1 and uses
    internal APIs that were removed or changed in transformers 5.x:
      1. ``isin_mps_friendly`` removed from ``transformers.pytorch_utils``
      2. ``ParlerTTSConfig.__init__`` fails on no-args instantiation
         (needed by ``to_diff_dict()``)
      3. ``tie_weights()`` signature gained ``**kwargs``

    These patches are applied once at import time so the rest of the
    module can use ``parler_tts`` normally.
    """
    import transformers

    if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
        transformers.pytorch_utils.isin_mps_friendly = torch.isin

    try:
        import parler_tts  # noqa: F401
    except ImportError:
        return

    from parler_tts.configuration_parler_tts import ParlerTTSConfig

    _orig_config_init = ParlerTTSConfig.__init__

    def _patched_config_init(self, vocab_size=1024, prompt_cross_attention=False, **kwargs):
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            super(ParlerTTSConfig, self).__init__(**kwargs)
            self.vocab_size = vocab_size
            self.prompt_cross_attention = prompt_cross_attention
            self.has_no_defaults_at_init = True
            return
        _orig_config_init(self, vocab_size=vocab_size, prompt_cross_attention=prompt_cross_attention, **kwargs)

    ParlerTTSConfig.__init__ = _patched_config_init

    from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration

    _orig_tie_weights = ParlerTTSForConditionalGeneration.tie_weights

    def _patched_tie_weights(self, **kwargs):
        if not hasattr(self.config, "tie_encoder_decoder"):
            return
        _orig_tie_weights(self)

    ParlerTTSForConditionalGeneration.tie_weights = _patched_tie_weights


_DEFAULT_CHUNK_SAMPLES = 44100
_DEFAULT_SAMPLE_RATE = 44100


def _pick(info: dict, key: str, default):
    """Extract scalar from additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


class ParlerTTSForGeneration(nn.Module):
    """Single-stage Parler-TTS model with streaming audio output.

    Uses the VoxCPM-style generator pattern: generate() output is
    chunked per-request and one chunk yielded per forward() call.
    The AR scheduler keeps the request alive until compute_logits()
    emits EOS.
    """

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        device = current_omni_platform.get_torch_device()
        self._device: torch.device = device

        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        elif device.type == "cuda":
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        logger.info("Loading Parler-TTS from %s (dtype=%s)", self.model_path, model_dtype)

        _patch_parler_tts_for_transformers5()

        try:
            from parler_tts import ParlerTTSForConditionalGeneration
        except ImportError:
            raise ImportError(
                "parler_tts is required for Parler-TTS. "
                "Install with: pip install git+https://github.com/huggingface/parler-tts.git"
            )
        from transformers import AutoTokenizer

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=model_dtype,
        ).to(device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self._sample_rate: int = int(getattr(self.config, "sampling_rate", _DEFAULT_SAMPLE_RATE))
        logger.info("Parler-TTS loaded on %s (sample_rate=%d)", device, self._sample_rate)

        self._lock = threading.Lock()
        self._stream_gens: dict[str, Any] = {}
        self._ar_last_chunk_flags: list[bool] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return {name for name, _ in self.named_parameters()}

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "description": "A speaker.", "_is_dummy": True}] * num_reqs

    def _create_stream_gen(self, info: dict[str, Any]):
        """Create a streaming generator for a request.

        Yields (waveform_chunk, is_last) tuples. The full generation runs
        eagerly, then the output is chunked into ~1s pieces for streaming.
        """
        text: str = str(_pick(info, "text", "") or "")
        description: str = str(
            _pick(
                info,
                "description",
                "A female speaker delivers a slightly expressive and animated speech "
                "with a moderate speed and pitch. The recording is of very high quality "
                "with the speaker's voice sounding clear and very close.",
            )
        )

        if not text.strip():
            logger.warning("Parler-TTS received empty text; yielding silence.")
            yield torch.zeros((self._sample_rate,), dtype=torch.float32), True
            return

        device = self._device or torch.device("cpu")

        input_ids = self._tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(device)

        with torch.inference_mode():
            generation = self._model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )

        audio = generation.cpu().float().squeeze()
        if audio.ndim == 0:
            yield torch.zeros((self._sample_rate,), dtype=torch.float32), True
            return

        if audio.ndim == 2:
            audio = audio.mean(dim=0).contiguous()

        chunk_size = int(_pick(info, "chunk_samples", _DEFAULT_CHUNK_SAMPLES))
        total_samples = audio.shape[0]

        if total_samples <= chunk_size:
            yield audio, True
            return

        offset = 0
        while offset < total_samples:
            end = min(offset + chunk_size, total_samples)
            chunk = audio[offset:end]
            is_last = end >= total_samples
            yield chunk, is_last
            offset = end

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        device = self._device or torch.device("cpu")
        hidden = int(getattr(self.config, "hidden_size", 768))
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        return torch.zeros((n, hidden), device=device, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._make_dummy_hidden(input_ids)

        infos = runtime_additional_information or [{}]

        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            self._ar_last_chunk_flags = [True] * len(infos)
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "model_outputs": [empty] * len(infos),
                    "sr": [sr_tensor] * len(infos),
                },
            )

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        last_chunk_flags: list[bool] = []

        for info in infos:
            if info.get("_is_dummy"):
                outputs.append(empty)
                srs.append(sr_tensor)
                last_chunk_flags.append(True)
                continue

            request_key = str(info.get("global_request_id") or info.get("_omni_req_id") or id(info))

            if request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(info)

            generator = self._stream_gens[request_key]
            try:
                chunk, is_last = next(generator)
            except StopIteration:
                self._stream_gens.pop(request_key, None)
                outputs.append(empty)
                last_chunk_flags.append(True)
            else:
                if is_last:
                    self._stream_gens.pop(request_key, None)
                outputs.append(chunk)
                last_chunk_flags.append(bool(is_last))

            srs.append(sr_tensor)

        self._ar_last_chunk_flags = last_chunk_flags

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for req_id in finished_req_ids:
            gen = self._stream_gens.pop(str(req_id), None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    logger.exception(
                        "Parler-TTS failed to close stream gen for request %s",
                        req_id,
                    )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            device = self._device or torch.device("cpu")
            hidden_states = torch.zeros((0, 1), device=device, dtype=torch.float32)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = int(getattr(self.config, "vocab_size", 32128))
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros(
            (num_rows, vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0

        flags = self._ar_last_chunk_flags
        for row in range(num_rows):
            is_last = flags[row] if row < len(flags) else True
            if is_last:
                logits[row, eos_id] = 1.0e6
            else:
                logits[row, eos_id] = -1.0e9
                logits[row, safe_id] = 1.0e6
        return logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden = int(getattr(self.config, "hidden_size", 768))
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )
