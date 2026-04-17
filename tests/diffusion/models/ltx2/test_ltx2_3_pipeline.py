# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the LTX-2.3 pipeline.

Tests verify:
1. LTX23Pipeline inherits from LTX2Pipeline correctly
2. All pipeline variants are registered in the diffusion registry
3. Post-process function detects vocoder output sample rate
4. LTX-2.3-specific classes exist and have correct inheritance
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Try direct import (works when vllm is installed, e.g. CI).
try:
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline, LTX2TwoStagesPipeline
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
        LTX23ImageToVideoPipeline,
        LTX23ImageToVideoTwoStagesPipeline,
        LTX23Pipeline,
        LTX23TwoStagesPipeline,
        _detect_vocoder_output_sample_rate,
    )
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_image2video import (
        LTX2ImageToVideoTwoStagesPipeline,
    )

    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
class TestInheritance:
    """Verify LTX-2.3 pipelines inherit from the correct LTX-2 base classes."""

    def test_ltx23_inherits_ltx2(self):
        assert issubclass(LTX23Pipeline, LTX2Pipeline)

    def test_ltx23_two_stages_inherits_ltx2(self):
        assert issubclass(LTX23TwoStagesPipeline, LTX2TwoStagesPipeline)

    def test_ltx23_i2v_inherits_ltx23(self):
        assert issubclass(LTX23ImageToVideoPipeline, LTX23Pipeline)

    def test_ltx23_i2v_two_stages_inherits_ltx2(self):
        assert issubclass(LTX23ImageToVideoTwoStagesPipeline, LTX2ImageToVideoTwoStagesPipeline)

    def test_ltx23_has_support_image_input(self):
        assert getattr(LTX23ImageToVideoPipeline, "support_image_input", False) is True

    def test_ltx23_overrides_init(self):
        assert LTX23Pipeline.__init__ is not LTX2Pipeline.__init__

    def test_ltx23_overrides_forward(self):
        assert LTX23Pipeline.forward is not LTX2Pipeline.forward

    def test_ltx23_overrides_get_gemma_prompt_embeds(self):
        assert LTX23Pipeline._get_gemma_prompt_embeds is not LTX2Pipeline._get_gemma_prompt_embeds

    def test_ltx23_inherits_encode_prompt(self):
        assert LTX23Pipeline.encode_prompt is LTX2Pipeline.encode_prompt

    def test_ltx23_inherits_prepare_latents(self):
        assert LTX23Pipeline.prepare_latents is LTX2Pipeline.prepare_latents


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
class TestRegistryIntegration:
    """Verify LTX-2.3 pipeline classes are registered in the diffusion registry."""

    def test_all_pipelines_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
            "LTX23TwoStagesPipeline",
            "LTX23ImageToVideoTwoStagesPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_MODELS, f"{name} not in _DIFFUSION_MODELS"

    def test_all_post_process_funcs_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
            "LTX23TwoStagesPipeline",
            "LTX23ImageToVideoTwoStagesPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_POST_PROCESS_FUNCS, f"{name} not in _DIFFUSION_POST_PROCESS_FUNCS"

    def test_all_cache_dit_enablers_registered(self):
        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
            "LTX23TwoStagesPipeline",
            "LTX23ImageToVideoTwoStagesPipeline",
        ]
        for name in expected:
            assert name in CUSTOM_DIT_ENABLERS, f"{name} not in CUSTOM_DIT_ENABLERS"

    def test_pipeline_module_paths_correct(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23Pipeline"][0] == "ltx2"
        assert _DIFFUSION_MODELS["LTX23Pipeline"][1] == "pipeline_ltx2_3"
        assert _DIFFUSION_MODELS["LTX23Pipeline"][2] == "LTX23Pipeline"

        assert _DIFFUSION_MODELS["LTX23ImageToVideoPipeline"][1] == "pipeline_ltx2_3_image2video"


# ---------------------------------------------------------------------------
# Vocoder sample rate detection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
class TestVocoderSampleRate:
    """Test the vocoder output sample rate detection function."""

    def test_returns_none_for_nonexistent_model(self):
        result = _detect_vocoder_output_sample_rate("/nonexistent/model")
        assert result is None

    def test_reads_from_local_config(self, tmp_path):
        import json

        vocoder_dir = tmp_path / "vocoder"
        vocoder_dir.mkdir()
        config = {"output_sampling_rate": 48000, "input_sampling_rate": 16000}
        (vocoder_dir / "config.json").write_text(json.dumps(config))

        result = _detect_vocoder_output_sample_rate(str(tmp_path))
        assert result == 48000

    def test_returns_none_for_ltx2_vocoder(self, tmp_path):
        """LTX-2 vocoder config has no output_sampling_rate key."""
        import json

        vocoder_dir = tmp_path / "vocoder"
        vocoder_dir.mkdir()
        config = {"upsample_rates": [8, 8], "model_in_dim": 64}
        (vocoder_dir / "config.json").write_text(json.dumps(config))

        result = _detect_vocoder_output_sample_rate(str(tmp_path))
        assert result is None
