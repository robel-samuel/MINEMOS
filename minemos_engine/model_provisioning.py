"""First-use provisioning of the engine's local transformer models.

Both models are ordinary Hugging Face checkpoints: nothing is bundled in the
wheel.  This module resolves them on first use, in this order:

1. the standard Hugging Face cache (or ``cache_dir`` / ``MINEMOS_MODEL_CACHE``
   when configured) is read without any network access;
2. only if the model is missing is it downloaded through ``transformers``,
   which writes to the same cache, shows its normal progress bars and resumes
   an interrupted download from the partial files;
3. the in-process component cache makes a second encoder or gate share the
   already loaded tokenizer and weights instead of loading them again.

Offline operation therefore needs no configuration: a cache populated by an
earlier run is used without contacting the network.  Setting
``HF_HUB_OFFLINE=1`` or ``TRANSFORMERS_OFFLINE=1`` turns every miss into an
immediate, explicit error instead.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

_LOGGER = logging.getLogger(__name__)

CACHE_DIR_ENV = "MINEMOS_MODEL_CACHE"

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_TRUTHY = frozenset({"1", "true", "yes", "on"})

_LOAD_LOCK = threading.RLock()
_COMPONENTS: dict[tuple, Any] = {}


class ModelProvisioningError(RuntimeError):
    """A local model required for semantic functionality is unavailable."""


def resolve_cache_dir(cache_dir: Optional[str] = None) -> Optional[str]:
    """Explicit argument, then ``MINEMOS_MODEL_CACHE``, else the HF default.

    ``None`` lets ``transformers`` apply its own resolution (``HF_HUB_CACHE``,
    ``HF_HOME`` or ``~/.cache/huggingface``).
    """
    configured = cache_dir
    if configured is None:
        configured = os.environ.get(CACHE_DIR_ENV) or None
    if configured is None:
        return None
    return str(Path(configured).expanduser().resolve())


def cache_location(cache_dir: Optional[str] = None) -> str:
    """Human-readable cache path used in messages."""
    resolved = resolve_cache_dir(cache_dir)
    if resolved is not None:
        return resolved
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home).expanduser() / "hub")
    return str(Path.home() / ".cache" / "huggingface" / "hub")


def offline_mode() -> bool:
    """Whether the Hugging Face clients were told not to use the network."""
    return any(
        os.environ.get(name, "").strip().lower() in _TRUTHY
        for name in _OFFLINE_ENV_VARS
    )


def clear_loaded_components() -> None:
    """Drops the in-process component cache (used by tests and diagnostics)."""
    with _LOAD_LOCK:
        _COMPONENTS.clear()


def _load(
    kind: str,
    model_name: str,
    factory: Callable[..., Any],
    *,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    device: Optional[str] = None,
) -> Any:
    resolved_cache = resolve_cache_dir(cache_dir)
    if local_files_only is None:
        local_files_only = offline_mode()
    key = (kind, model_name, resolved_cache, device)
    with _LOAD_LOCK:
        if key in _COMPONENTS:
            return _COMPONENTS[key]

        if local_files_only:
            try:
                component = factory(
                    model_name, cache_dir=resolved_cache,
                    local_files_only=True,
                )
            except Exception as exc:  # noqa: BLE001 - reported below
                raise ModelProvisioningError(
                    _offline_message(kind, model_name, resolved_cache)
                ) from exc
        else:
            # Cached models never touch the network: the local load either
            # succeeds or tells us a download is genuinely required.
            try:
                component = factory(
                    model_name, cache_dir=resolved_cache,
                    local_files_only=True,
                )
            except Exception:  # noqa: BLE001 - expected on a cold cache
                _LOGGER.info(
                    "Downloading %s model %r on first use; cache: %s",
                    kind, model_name, cache_location(resolved_cache))
                try:
                    component = factory(
                        model_name, cache_dir=resolved_cache,
                    )
                except Exception as exc:  # noqa: BLE001 - reported below
                    raise ModelProvisioningError(
                        _download_failure_message(
                            kind, model_name, resolved_cache)
                    ) from exc

        if device is not None:
            component = component.to(device)
        if kind != "tokenizer":
            component.eval()
        _COMPONENTS[key] = component
        return component


def _offline_message(
    kind: str, model_name: str, cache_dir: Optional[str]
) -> str:
    return (
        f"The {kind} model {model_name!r} is not in the Hugging Face cache "
        f"at {cache_location(cache_dir)}, and offline mode is enabled "
        f"({', '.join(_OFFLINE_ENV_VARS)}). Run once with network access "
        f"without those variables to let MINEMOS download it, or point "
        f"HF_HOME / {CACHE_DIR_ENV} at a cache that already contains it."
    )


def _download_failure_message(
    kind: str, model_name: str, cache_dir: Optional[str]
) -> str:
    return (
        f"Could not download the {kind} model {model_name!r} into the "
        f"Hugging Face cache at {cache_location(cache_dir)}. Check network "
        f"access and free disk space, then retry; an interrupted download "
        f"resumes from its partial files. To use another location, set "
        f"HF_HOME or {CACHE_DIR_ENV}. If any of "
        f"({', '.join(_OFFLINE_ENV_VARS)}) is set, unset it for the first "
        f"download."
    )


def load_tokenizer(
    model_name: str,
    *,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
):
    from transformers import AutoTokenizer

    return _load(
        "tokenizer", model_name, AutoTokenizer.from_pretrained,
        cache_dir=cache_dir, local_files_only=local_files_only,
    )


def load_base_model(
    model_name: str,
    *,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    device: str = "cpu",
):
    from transformers import AutoModel

    return _load(
        "sentence-embedding", model_name, AutoModel.from_pretrained,
        cache_dir=cache_dir, local_files_only=local_files_only,
        device=device,
    )


def load_sequence_classification_model(
    model_name: str,
    *,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    device: str = "cpu",
):
    from transformers import AutoModelForSequenceClassification

    return _load(
        "NLI", model_name,
        AutoModelForSequenceClassification.from_pretrained,
        cache_dir=cache_dir, local_files_only=local_files_only,
        device=device,
    )


__all__ = [
    "CACHE_DIR_ENV",
    "ModelProvisioningError",
    "cache_location",
    "clear_loaded_components",
    "load_base_model",
    "load_sequence_classification_model",
    "load_tokenizer",
    "offline_mode",
    "resolve_cache_dir",
]