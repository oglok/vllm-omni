import os
from functools import lru_cache

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

logger = init_logger(__name__)


def load_diffusers_config(model_name) -> dict:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    config = DiffusionPipeline.load_config(model_name)
    return config


def _looks_like_bagel(model_name: str) -> bool:
    """Best-effort detection for Bagel (non-diffusers) diffusion models."""
    try:
        cfg = get_hf_file_to_dict("config.json", model_name)
        model_type = cfg.get("model_type")
        if model_type == "bagel":
            return True
        architectures = cfg.get("architectures") or []
        return "BagelForConditionalGeneration" in architectures
    except Exception:
        return False


def _looks_like_raw_diffusion_model(model_name: str) -> bool:
    """Detect raw safetensors diffusion models without diffusers/transformers config.

    Models like Lightricks/LTX-2.3 ship only ``.safetensors`` files with no
    ``config.json`` or ``model_index.json``.  We identify them by checking
    that the HuggingFace repo (or local directory) contains ``.safetensors``
    files but lacks standard config files.
    """
    # Local directory: check for .safetensors files without config.json
    if os.path.isdir(model_name):
        has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(model_name))
        has_config = os.path.exists(os.path.join(model_name, "config.json"))
        has_model_index = os.path.exists(os.path.join(model_name, "model_index.json"))
        if has_safetensors and not has_config and not has_model_index:
            logger.debug("Detected raw safetensors diffusion model (local): %s", model_name)
            return True
        return False

    # Remote HuggingFace repo: list files to check
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(model_name)
        has_safetensors = any(f.endswith(".safetensors") for f in files)
        has_config = "config.json" in files
        has_model_index = "model_index.json" in files
        if has_safetensors and not has_config and not has_model_index:
            logger.debug("Detected raw safetensors diffusion model (remote): %s", model_name)
            return True
    except Exception as e:
        logger.debug("Failed to check HF repo for raw safetensors: %s", e)

    return False


@lru_cache
def is_diffusion_model(model_name: str) -> bool:
    """Check if a model is a diffusion model.

    Uses multiple fallback strategies to detect diffusion models:
    1. Check local file system for model_index.json (fastest, no imports)
    2. Check using vllm's get_hf_file_to_dict utility
    3. Try the standard diffusers approach (may fail due to import issues)
    """
    # Strategy 1: Check local file system first (fastest, avoids import issues)
    if os.path.isdir(model_name):
        model_index_path = os.path.join(model_name, "model_index.json")
        if os.path.exists(model_index_path):
            try:
                import json

                with open(model_index_path) as f:
                    config_dict = json.load(f)
                if config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
                    logger.debug("Detected diffusion model via local model_index.json")
                    return True
            except Exception as e:
                logger.debug("Failed to read local model_index.json: %s", e)

    # Strategy 2: Check using vllm's utility (works for both local and remote models)
    try:
        config_dict = get_hf_file_to_dict("model_index.json", model_name)
        if config_dict is not None and config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
            logger.debug("Detected diffusion model via model_index.json")
            return True
    except Exception as e:
        logger.debug("Failed to check model_index.json via get_hf_file_to_dict: %s", e)

    # Strategy 3: Try the standard diffusers approach (may fail due to import issues)
    # This is last because it requires importing diffusers/xformers/flash_attn
    # which may have compatibility issues
    try:
        load_diffusers_config(model_name)
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug("Failed to import diffusers dependencies: %s", e)
        logger.debug("This may be due to flash_attn/PyTorch version mismatch")
    except Exception as e:
        logger.debug("Failed to load diffusers config via DiffusionPipeline: %s", e)

        # Bagel is not a diffusers pipeline (no model_index.json), but is still a
        # diffusion-style model in vllm-omni. Detect it via config.json.
    if _looks_like_bagel(model_name):
        return True

    # Strategy 4: Detect raw safetensors models (e.g. Lightricks/LTX-2.3)
    # that are neither diffusers format nor transformers format.
    # These are identified by having .safetensors files but no config.json
    # or model_index.json.
    return _looks_like_raw_diffusion_model(model_name)
