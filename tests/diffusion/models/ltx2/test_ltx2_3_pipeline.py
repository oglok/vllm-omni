# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the LTX-2.3 pipeline.

Tests verify:
1. LTX23Pipeline is independent (does NOT inherit from LTX2Pipeline)
2. All pipeline variants are registered in the diffusion registry
3. Post-process function detects vocoder output sample rate
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

try:
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
        LTX23ImageToVideoPipeline,
        LTX23ImageToVideoTwoStagesPipeline,
        LTX23Pipeline,
        LTX23TwoStagesPipeline,
        _detect_vocoder_output_sample_rate,
    )

    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
class TestPipelineStructure:
    """Verify LTX-2.3 pipeline is independent."""

    def test_ltx23_is_nn_module(self):
        from torch import nn

        assert issubclass(LTX23Pipeline, nn.Module)

    def test_ltx23_does_not_inherit_ltx2(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        assert not issubclass(LTX23Pipeline, LTX2Pipeline)

    def test_ltx23_has_forward(self):
        assert hasattr(LTX23Pipeline, "forward")

    def test_ltx23_has_load_weights(self):
        assert hasattr(LTX23Pipeline, "load_weights")

    def test_ltx23_i2v_exists(self):
        assert LTX23ImageToVideoPipeline is not None

    def test_ltx23_two_stages_exists(self):
        assert LTX23TwoStagesPipeline is not None

    def test_ltx23_i2v_two_stages_exists(self):
        assert LTX23ImageToVideoTwoStagesPipeline is not None


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
class TestRegistryIntegration:
    """Verify LTX-2.3 pipeline classes are registered."""

    def test_all_pipelines_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        for name in [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
            "LTX23TwoStagesPipeline",
            "LTX23ImageToVideoTwoStagesPipeline",
        ]:
            assert name in _DIFFUSION_MODELS, f"{name} not in _DIFFUSION_MODELS"

    def test_all_post_process_funcs_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        for name in [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
            "LTX23TwoStagesPipeline",
            "LTX23ImageToVideoTwoStagesPipeline",
        ]:
            assert name in _DIFFUSION_POST_PROCESS_FUNCS, f"{name} not in _DIFFUSION_POST_PROCESS_FUNCS"

    def test_all_cache_dit_enablers_registered(self):
        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS

        for name in [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
            "LTX23TwoStagesPipeline",
            "LTX23ImageToVideoTwoStagesPipeline",
        ]:
            assert name in CUSTOM_DIT_ENABLERS, f"{name} not in CUSTOM_DIT_ENABLERS"

    def test_pipeline_module_paths_correct(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23Pipeline"] == ("ltx2", "pipeline_ltx2_3", "LTX23Pipeline")
        assert _DIFFUSION_MODELS["LTX23ImageToVideoPipeline"][1] == "pipeline_ltx2_3_image2video"


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
class TestVocoderSampleRate:
    """Test vocoder output sample rate detection."""

    def test_returns_none_for_nonexistent_model(self):
        assert _detect_vocoder_output_sample_rate("/nonexistent/model") is None

    def test_reads_from_local_config(self, tmp_path):
        import json

        (tmp_path / "vocoder").mkdir()
        (tmp_path / "vocoder" / "config.json").write_text(json.dumps({"output_sampling_rate": 48000}))
        assert _detect_vocoder_output_sample_rate(str(tmp_path)) == 48000

    def test_returns_none_for_ltx2_vocoder(self, tmp_path):
        import json

        (tmp_path / "vocoder").mkdir()
        (tmp_path / "vocoder" / "config.json").write_text(json.dumps({"upsample_rates": [8, 8]}))
        assert _detect_vocoder_output_sample_rate(str(tmp_path)) is None
