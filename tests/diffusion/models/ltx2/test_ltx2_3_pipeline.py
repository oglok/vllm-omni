# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the LTX-2.3 pipeline loading utilities.

These tests validate the pure-logic helper functions (safetensors path
resolution, metadata loading, component weight filtering, key remapping,
text-encoder path resolution) and registry integration WITHOUT requiring
a GPU or actual model weights.

The utility functions are loaded directly from the source file to avoid
triggering the heavy vllm/diffusers import chain through vllm_omni.__init__.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Load the pipeline module directly to avoid the vllm_omni.__init__ chain
# which requires vllm to be installed.
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(__file__).resolve().parents[4] / ("vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py")


def _import_ltx23_utils():
    """Extract and compile the utility functions from pipeline_ltx2_3.py.

    This reads the source file and exec's just the utility functions in a
    minimal namespace containing only stdlib + safetensors + torch.  This
    completely avoids importing vllm_omni, diffusers, transformers, or vllm.
    """
    import ast

    source = _MODULE_PATH.read_text()
    tree = ast.parse(source)

    # Names of the utility functions / constants we want to test
    _WANTED = {
        "_RAW_TRANSFORMER_PREFIX",
        "_RAW_VAE_PREFIX",
        "_RAW_AUDIO_VAE_PREFIX",
        "_RAW_VOCODER_PREFIX",
        "_RAW_CONNECTORS_PREFIX",
        "_DEFAULT_TEXT_ENCODER_REPO",
        "_resolve_safetensors_path",
        "_load_safetensors_metadata",
        "_load_component_weights_from_safetensors",
        "_get_text_encoder_path",
    }

    # Extract only the AST nodes for the functions/constants we need.
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _WANTED:
                nodes.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _WANTED:
                    nodes.append(node)

    # Build a new module from the extracted nodes and execute it.
    new_tree = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, str(_MODULE_PATH), "exec")

    import logging

    import safetensors

    ns: dict = {
        "__builtins__": __builtins__,
        "os": os,
        "json": json,
        "torch": torch,
        "safetensors": safetensors,
        "logger": logging.getLogger("test_ltx2_3"),
    }

    exec(code, ns)

    # Return as a SimpleNamespace so callers can do mod.func_name
    from types import SimpleNamespace

    return SimpleNamespace(**{k: ns[k] for k in _WANTED if k in ns})


# Try direct import first (works when vllm is installed, e.g. CI).
# Fall back to the stub-based loader for environments without vllm.
try:
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
        _DEFAULT_TEXT_ENCODER_REPO,
        _RAW_AUDIO_VAE_PREFIX,
        _RAW_CONNECTORS_PREFIX,
        _RAW_TRANSFORMER_PREFIX,
        _RAW_VAE_PREFIX,
        _RAW_VOCODER_PREFIX,
        _get_text_encoder_path,
        _load_component_weights_from_safetensors,
        _load_safetensors_metadata,
        _resolve_safetensors_path,
    )

    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _mod = _import_ltx23_utils()
    _RAW_TRANSFORMER_PREFIX = _mod._RAW_TRANSFORMER_PREFIX
    _RAW_VAE_PREFIX = _mod._RAW_VAE_PREFIX
    _RAW_AUDIO_VAE_PREFIX = _mod._RAW_AUDIO_VAE_PREFIX
    _RAW_VOCODER_PREFIX = _mod._RAW_VOCODER_PREFIX
    _RAW_CONNECTORS_PREFIX = _mod._RAW_CONNECTORS_PREFIX
    _DEFAULT_TEXT_ENCODER_REPO = _mod._DEFAULT_TEXT_ENCODER_REPO
    _resolve_safetensors_path = _mod._resolve_safetensors_path
    _load_safetensors_metadata = _mod._load_safetensors_metadata
    _load_component_weights_from_safetensors = _mod._load_component_weights_from_safetensors
    _get_text_encoder_path = _mod._get_text_encoder_path
    _HAS_VLLM = False


pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_safetensors(
    path: str,
    tensors: dict[str, torch.Tensor] | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    """Write a small safetensors file with optional metadata."""
    if tensors is None:
        tensors = {"dummy": torch.zeros(1)}
    save_file(tensors, path, metadata=metadata)
    return path


@dataclass
class _FakeODConfig:
    """Minimal stand-in for OmniDiffusionConfig used by _get_text_encoder_path."""

    model: str | None = None
    custom_pipeline_args: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# _resolve_safetensors_path
# ---------------------------------------------------------------------------


class TestResolveSafetensorsPath:
    """Tests for _resolve_safetensors_path."""

    def test_direct_file_path(self, tmp_path):
        sft = tmp_path / "model.safetensors"
        _write_safetensors(str(sft))
        assert _resolve_safetensors_path(str(sft)) == str(sft)

    def test_non_safetensors_file_returns_none(self, tmp_path):
        txt = tmp_path / "model.txt"
        txt.write_text("not a model")
        assert _resolve_safetensors_path(str(txt)) is None

    def test_empty_directory_returns_none(self, tmp_path):
        assert _resolve_safetensors_path(str(tmp_path)) is None

    def test_directory_prefers_dev_file(self, tmp_path):
        dev = tmp_path / "ltx-2.3-22b-dev.safetensors"
        distilled = tmp_path / "ltx-2.3-22b-distilled.safetensors"
        lora = tmp_path / "ltx-2.3-22b-distilled-lora-384.safetensors"
        _write_safetensors(str(dev))
        _write_safetensors(str(distilled))
        _write_safetensors(str(lora))
        assert _resolve_safetensors_path(str(tmp_path)) == str(dev)

    def test_directory_prefers_distilled_over_lora(self, tmp_path):
        distilled = tmp_path / "ltx-2.3-22b-distilled.safetensors"
        lora = tmp_path / "ltx-2.3-22b-distilled-lora-384.safetensors"
        _write_safetensors(str(distilled))
        _write_safetensors(str(lora))
        assert _resolve_safetensors_path(str(tmp_path)) == str(distilled)

    def test_directory_skips_upscaler_files(self, tmp_path):
        upscaler = tmp_path / "ltx-2.3-spatial-upscaler-x2-dev.safetensors"
        dev = tmp_path / "ltx-2.3-22b-dev.safetensors"
        _write_safetensors(str(upscaler))
        _write_safetensors(str(dev))
        assert _resolve_safetensors_path(str(tmp_path)) == str(dev)

    def test_directory_fallback_to_largest_file(self, tmp_path):
        small = tmp_path / "small.safetensors"
        big = tmp_path / "big.safetensors"
        _write_safetensors(str(small), {"a": torch.zeros(1)})
        _write_safetensors(str(big), {"a": torch.zeros(100)})
        result = _resolve_safetensors_path(str(tmp_path))
        assert result == str(big)

    def test_nonexistent_path_returns_none(self):
        assert _resolve_safetensors_path("/nonexistent/path") is None


# ---------------------------------------------------------------------------
# _load_safetensors_metadata
# ---------------------------------------------------------------------------


class TestLoadSafetensorsMetadata:
    """Tests for _load_safetensors_metadata."""

    def test_loads_config_from_metadata(self, tmp_path):
        config = {
            "transformer": {"num_layers": 52, "num_attention_heads": 32},
            "vae": {"latent_channels": 128},
        }
        sft = tmp_path / "model.safetensors"
        _write_safetensors(str(sft), metadata={"config": json.dumps(config)})
        result = _load_safetensors_metadata(str(sft))
        assert result["transformer"]["num_layers"] == 52
        assert result["vae"]["latent_channels"] == 128

    def test_returns_empty_when_no_config_key(self, tmp_path):
        sft = tmp_path / "model.safetensors"
        _write_safetensors(str(sft), metadata={"version": "1.0"})
        assert _load_safetensors_metadata(str(sft)) == {}

    def test_returns_empty_when_no_metadata(self, tmp_path):
        sft = tmp_path / "model.safetensors"
        _write_safetensors(str(sft))
        assert _load_safetensors_metadata(str(sft)) == {}

    def test_returns_empty_for_nonexistent_path(self):
        assert _load_safetensors_metadata("/nonexistent") == {}

    def test_directory_path_resolution(self, tmp_path):
        config = {"transformer": {"num_layers": 48}}
        sft = tmp_path / "ltx-2.3-22b-dev.safetensors"
        _write_safetensors(str(sft), metadata={"config": json.dumps(config)})
        result = _load_safetensors_metadata(str(tmp_path))
        assert result["transformer"]["num_layers"] == 48


# ---------------------------------------------------------------------------
# _load_component_weights_from_safetensors
# ---------------------------------------------------------------------------


class TestLoadComponentWeights:
    """Tests for _load_component_weights_from_safetensors."""

    def _make_checkpoint(self, tmp_path) -> str:
        """Create a safetensors file mimicking LTX-2.3 key structure."""
        tensors = {
            # Transformer weights
            "model.diffusion_model.proj_in.weight": torch.randn(4, 4),
            "model.diffusion_model.proj_in.bias": torch.randn(4),
            "model.diffusion_model.blocks.0.norm.weight": torch.randn(4),
            # VAE weights
            "vae.encoder.conv_in.weight": torch.randn(4, 4),
            "vae.decoder.conv_out.weight": torch.randn(4, 4),
            # Audio VAE weights
            "audio_vae.decoder.conv_in.weight": torch.randn(4, 4),
            "audio_vae.per_channel_statistics.mean": torch.randn(4),
            # Vocoder weights
            "vocoder.conv_pre.weight": torch.randn(4, 4),
            # Connectors
            "embeddings_proj.linear.weight": torch.randn(4, 4),
        }
        sft_path = str(tmp_path / "checkpoint.safetensors")
        save_file(tensors, sft_path)
        return sft_path

    def test_filter_transformer_keys(self, tmp_path):
        sft = self._make_checkpoint(tmp_path)
        sd = _load_component_weights_from_safetensors(sft, _RAW_TRANSFORMER_PREFIX)
        assert "proj_in.weight" in sd
        assert "proj_in.bias" in sd
        assert "blocks.0.norm.weight" in sd
        assert len(sd) == 3

    def test_filter_vae_keys(self, tmp_path):
        sft = self._make_checkpoint(tmp_path)
        sd = _load_component_weights_from_safetensors(sft, _RAW_VAE_PREFIX)
        assert "encoder.conv_in.weight" in sd
        assert "decoder.conv_out.weight" in sd
        assert len(sd) == 2

    def test_filter_audio_vae_keys(self, tmp_path):
        sft = self._make_checkpoint(tmp_path)
        sd = _load_component_weights_from_safetensors(sft, _RAW_AUDIO_VAE_PREFIX)
        assert "decoder.conv_in.weight" in sd
        assert "per_channel_statistics.mean" in sd
        assert len(sd) == 2

    def test_filter_vocoder_keys(self, tmp_path):
        sft = self._make_checkpoint(tmp_path)
        sd = _load_component_weights_from_safetensors(sft, _RAW_VOCODER_PREFIX)
        assert "conv_pre.weight" in sd
        assert len(sd) == 1

    def test_filter_connectors_keys(self, tmp_path):
        sft = self._make_checkpoint(tmp_path)
        sd = _load_component_weights_from_safetensors(sft, _RAW_CONNECTORS_PREFIX)
        assert "linear.weight" in sd
        assert len(sd) == 1

    def test_no_strip_prefix(self, tmp_path):
        sft = self._make_checkpoint(tmp_path)
        sd = _load_component_weights_from_safetensors(sft, _RAW_VOCODER_PREFIX, strip_prefix=False)
        assert "vocoder.conv_pre.weight" in sd

    def test_empty_result_for_unknown_prefix(self, tmp_path):
        sft = self._make_checkpoint(tmp_path)
        sd = _load_component_weights_from_safetensors(sft, "nonexistent.")
        assert len(sd) == 0

    def test_tensors_have_correct_values(self, tmp_path):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sft_path = str(tmp_path / "values.safetensors")
        save_file({"vocoder.test": t}, sft_path)
        sd = _load_component_weights_from_safetensors(sft_path, "vocoder.")
        assert torch.equal(sd["test"], t)


# ---------------------------------------------------------------------------
# _get_text_encoder_path
# ---------------------------------------------------------------------------


class TestGetTextEncoderPath:
    """Tests for _get_text_encoder_path."""

    def test_from_custom_pipeline_args(self, tmp_path):
        te_path = str(tmp_path / "gemma3")
        os.makedirs(te_path)
        cfg = _FakeODConfig(
            model="/some/model",
            custom_pipeline_args={"text_encoder_path": te_path},
        )
        assert _get_text_encoder_path(cfg) == te_path

    def test_from_text_encoder_subdirectory(self, tmp_path):
        model_dir = tmp_path / "model"
        te_dir = model_dir / "text_encoder"
        te_dir.mkdir(parents=True)
        cfg = _FakeODConfig(model=str(model_dir))
        assert _get_text_encoder_path(cfg) == str(model_dir)

    def test_auto_downloads_from_default_repo(self, tmp_path, monkeypatch):
        """When no local path is found, auto-downloads from _DEFAULT_TEXT_ENCODER_REPO."""
        downloaded_to = str(tmp_path / "downloaded")
        os.makedirs(downloaded_to)

        def fake_snapshot_download(repo_id, **kwargs):
            assert repo_id == _DEFAULT_TEXT_ENCODER_REPO
            assert "text_encoder/*" in kwargs.get("allow_patterns", [])
            assert "tokenizer/*" in kwargs.get("allow_patterns", [])
            return downloaded_to

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

        cfg = _FakeODConfig(model=str(tmp_path))
        result = _get_text_encoder_path(cfg)
        assert result == downloaded_to

    def test_auto_downloads_from_custom_repo(self, tmp_path, monkeypatch):
        """text_encoder_model overrides the default download repo."""
        downloaded_to = str(tmp_path / "downloaded")
        os.makedirs(downloaded_to)

        def fake_snapshot_download(repo_id, **kwargs):
            assert repo_id == "my-org/custom-gemma3"
            return downloaded_to

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

        cfg = _FakeODConfig(
            model=str(tmp_path),
            custom_pipeline_args={"text_encoder_model": "my-org/custom-gemma3"},
        )
        result = _get_text_encoder_path(cfg)
        assert result == downloaded_to

    def test_hf_id_in_text_encoder_path_triggers_download(self, tmp_path, monkeypatch):
        """A non-local text_encoder_path is used as the repo to download from."""
        downloaded_to = str(tmp_path / "downloaded")
        os.makedirs(downloaded_to)

        def fake_snapshot_download(repo_id, **kwargs):
            assert repo_id == "google/gemma-3-12b"
            return downloaded_to

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

        cfg = _FakeODConfig(
            model=str(tmp_path),
            custom_pipeline_args={"text_encoder_path": "google/gemma-3-12b"},
        )
        result = _get_text_encoder_path(cfg)
        assert result == downloaded_to

    def test_none_custom_args_triggers_auto_download(self, tmp_path, monkeypatch):
        downloaded_to = str(tmp_path / "downloaded")
        os.makedirs(downloaded_to)

        def fake_snapshot_download(repo_id, **kwargs):
            assert repo_id == _DEFAULT_TEXT_ENCODER_REPO
            return downloaded_to

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

        cfg = _FakeODConfig(model=str(tmp_path), custom_pipeline_args=None)
        result = _get_text_encoder_path(cfg)
        assert result == downloaded_to

    def test_default_text_encoder_repo_value(self):
        assert _DEFAULT_TEXT_ENCODER_REPO == "Lightricks/LTX-2"


# ---------------------------------------------------------------------------
# Weight key remapping in load_weights
# ---------------------------------------------------------------------------

# We test the remapping logic directly (same code as LTX23Pipeline.load_weights)
# without instantiating the full pipeline.
_TRANSFORMER_PREFIX = "model.diffusion_model."


class TestWeightKeyRemapping:
    """Tests for the key remapping logic used in LTX23Pipeline.load_weights."""

    @staticmethod
    def _remap_weights(weights):
        """Reproduces the key remapping from LTX23Pipeline.load_weights."""
        for name, tensor in weights:
            if name.startswith(_TRANSFORMER_PREFIX):
                new_name = "transformer." + name[len(_TRANSFORMER_PREFIX) :]
                yield new_name, tensor

    def test_transformer_prefix_remapped(self):
        input_weights = [
            ("model.diffusion_model.proj_in.weight", torch.zeros(2)),
            ("model.diffusion_model.blocks.0.norm.weight", torch.zeros(2)),
        ]
        remapped = list(self._remap_weights(input_weights))
        assert remapped[0][0] == "transformer.proj_in.weight"
        assert remapped[1][0] == "transformer.blocks.0.norm.weight"

    def test_non_transformer_keys_skipped(self):
        input_weights = [
            ("model.diffusion_model.proj_in.weight", torch.zeros(2)),
            ("vae.encoder.conv_in.weight", torch.zeros(2)),
            ("audio_vae.decoder.conv_in.weight", torch.zeros(2)),
            ("vocoder.conv_pre.weight", torch.zeros(2)),
        ]
        remapped = list(self._remap_weights(input_weights))
        # Only transformer key should pass through
        assert len(remapped) == 1
        assert remapped[0][0] == "transformer.proj_in.weight"

    def test_empty_input(self):
        assert list(self._remap_weights([])) == []

    def test_preserves_tensor_values(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        input_weights = [("model.diffusion_model.x", t)]
        remapped = list(self._remap_weights(input_weights))
        assert torch.equal(remapped[0][1], t)


# ---------------------------------------------------------------------------
# Registry integration (requires vllm installed)
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
# Prefix constants
# ---------------------------------------------------------------------------


class TestPrefixConstants:
    """Sanity-check the weight key prefix constants."""

    def test_transformer_prefix(self):
        assert _RAW_TRANSFORMER_PREFIX == "model.diffusion_model."

    def test_vae_prefix(self):
        assert _RAW_VAE_PREFIX == "vae."

    def test_audio_vae_prefix(self):
        assert _RAW_AUDIO_VAE_PREFIX == "audio_vae."

    def test_vocoder_prefix(self):
        assert _RAW_VOCODER_PREFIX == "vocoder."

    def test_connectors_prefix(self):
        assert _RAW_CONNECTORS_PREFIX == "embeddings_proj."

    def test_prefixes_are_disjoint(self):
        """No prefix should be a prefix of another (avoids key conflicts)."""
        prefixes = [
            _RAW_TRANSFORMER_PREFIX,
            _RAW_VAE_PREFIX,
            _RAW_AUDIO_VAE_PREFIX,
            _RAW_VOCODER_PREFIX,
            _RAW_CONNECTORS_PREFIX,
        ]
        for i, a in enumerate(prefixes):
            for j, b in enumerate(prefixes):
                if i != j:
                    assert not a.startswith(b), f"Prefix conflict: {a!r} starts with {b!r}"
