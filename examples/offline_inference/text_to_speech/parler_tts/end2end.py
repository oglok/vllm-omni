# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Parler-TTS via vLLM-Omni.

Single-stage pipeline: the T5 text encoder, autoregressive decoder, and DAC
audio codec all run inside one generation stage. Output is 44.1 kHz mono WAV.

Parler-TTS uses a *description* string to condition the voice (speaker
identity, style, recording quality) rather than reference audio for cloning.

Usage:
  # Basic synthesis with default voice description.
  python end2end.py --text "Hello, how are you doing today?"

  # Custom voice description.
  python end2end.py \\
    --text "Hello, how are you doing today?" \\
    --description "Jon's voice is monotone yet slightly fast in delivery, \\
      with a very close recording that almost has no background noise."

  # Named speakers (trained on 34 speakers): Jon, Lea, Gary, Jenna, Mike, Laura.
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams

from vllm_omni.utils.tracking_parser import TrackingArgumentParser

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402

MODEL = "parler-tts/parler-tts-large-v1"

_DEFAULT_DESCRIPTION = (
    "A female speaker delivers a slightly expressive and animated speech "
    "with a moderate speed and pitch. The recording is of very high quality "
    "with the speaker's voice sounding clear and very close."
)


def build_request(
    text: str,
    description: str = _DEFAULT_DESCRIPTION,
) -> dict:
    """Build an Omni request payload for Parler-TTS."""
    additional: dict = {
        "text": [text],
        "description": [description],
    }
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": additional,
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 44100) -> None:
    audio_np = waveform.float().numpy()
    sf.write(path, audio_np, sample_rate)
    print(f"  Saved {path} ({audio_np.shape}, {sample_rate} Hz)")


def main(args) -> None:
    omni = Omni(
        model=MODEL,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing: {args.text!r}")
    print(f"  description: {args.description!r}")
    inputs = build_request(
        text=args.text,
        description=args.description,
    )
    params_list = sampling_params

    for stage_outputs in omni.generate(inputs, params_list):
        for i, req_output in enumerate(stage_outputs.request_output):
            for j, out in enumerate(req_output.outputs):
                mm = out.multimodal_output
                if mm is None:
                    print(f"  [req {i}] No audio output.")
                    continue
                audio = mm.get("audio")
                sr_tensor = mm.get("sr")
                if audio is None:
                    print(f"  [req {i}] No waveform in multimodal_output.")
                    continue
                sr = int(sr_tensor.item()) if sr_tensor is not None else 44100
                out_path = str(output_dir / f"output_{i}_{j}.wav")
                save_audio(audio.cpu(), out_path, sr)

    print("Done.")


def parse_args():
    parser = TrackingArgumentParser(description="Parler-TTS offline inference")
    parser.add_argument(
        "--text",
        default="Hello, how are you doing today?",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--description",
        default=_DEFAULT_DESCRIPTION,
        help="Voice description for conditioning.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get(
                "XDG_CACHE_HOME",
                os.path.join(os.path.expanduser("~"), ".cache"),
            ),
            "parler_tts_output",
        ),
        help="Directory for WAV outputs.",
    )
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Path to a deploy YAML; leave unset to auto-load vllm_omni/deploy/parler_tts.yaml.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
