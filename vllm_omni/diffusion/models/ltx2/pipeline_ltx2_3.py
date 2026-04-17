# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 pipeline aliases.

LTX-2.3 uses the same pipeline architecture as LTX-2 when loaded from
diffusers format (``dg845/LTX-2.3-Diffusers``).  The diffusers 0.38.0+
component classes (``AutoencoderKLLTX2Video``, ``LTX2VocoderWithBWE``,
``LTX2VideoTransformer3DModel``, etc.) handle the architectural differences
internally via their config files.

These aliases exist so that:
1. Users can explicitly request LTX-2.3 via ``--model-class-name LTX23Pipeline``
2. The registry can map LTX-2.3 to the correct post-process function and
   Cache-DiT enablers
3. Future LTX-2.3-specific behavior can be added without modifying LTX2Pipeline

Usage::

    # Diffusers format (recommended)
    vllm serve dg845/LTX-2.3-Diffusers --omni --model-class-name LTX23Pipeline

    # Also works with the base class name
    vllm serve dg845/LTX-2.3-Diffusers --omni --model-class-name LTX2Pipeline

Requires ``diffusers >= 0.38.0`` for LTX-2.3 component support.
"""

from .pipeline_ltx2 import (
    LTX2Pipeline,
    LTX2TwoStagesPipeline,
    get_ltx2_post_process_func,  # noqa: F401 - loaded by registry via getattr
)


class LTX23Pipeline(LTX2Pipeline):
    """LTX-2.3 text-to-video pipeline.

    Identical to :class:`LTX2Pipeline`. The diffusers component classes
    handle the LTX-2.3 architectural differences (22B transformer,
    VocoderWithBWE, cross-attn adaln, etc.) via their config files.
    """

    pass


class LTX23TwoStagesPipeline(LTX2TwoStagesPipeline):
    """LTX-2.3 two-stage pipeline with latent upsampling.

    Identical to :class:`LTX2TwoStagesPipeline`.
    """

    pass
