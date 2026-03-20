"""
lmstudio.py — LM Studio / mlx-engine integration for Flash Mode.

apply_flash_patch(config)
    Replaces mlx_lm.load with a flash-aware version.  Call once at startup.
    If config.enabled is False, the function is a no-op.

The patched load():
    1. Checks for the flash_mode key in the model directory's
       config.json (injected by LM Studio's checkbox).
    2. Falls back to the original mlx_lm.load if flash_mode is absent.
    3. If flash_mode is present and True, routes through FlashManager.

This is intentionally minimal — we do NOT subclass any mlx-engine class.
The goal is a single-file diff that can be upstreamed as a PR.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ..config import FlashConfig
from ..manager import FlashManager

_ORIGINAL_LOAD: Any | None = None
_MANAGER: FlashManager | None = None


def apply_flash_patch(config: FlashConfig | None = None) -> None:
    """
    Monkey-patch mlx_lm.load to route through FlashManager when appropriate.

    Parameters
    ----------
    config:
        If None, a default FlashConfig is used.  enabled defaults to False
        so the patch is a no-op unless explicitly activated.
    """
    global _ORIGINAL_LOAD, _MANAGER

    if config is None:
        config = FlashConfig(enabled=False)

    try:
        import mlx_lm
    except ImportError as e:
        raise ImportError(
            "mlx_lm is not installed.  "
            "Install the mlx-engine extras: pip install mlx-lm"
        ) from e

    if _ORIGINAL_LOAD is not None:
        # Already patched; update config and return
        if _MANAGER is not None:
            _MANAGER.config = config
        return

    _ORIGINAL_LOAD = mlx_lm.load
    _MANAGER = FlashManager(config)

    @functools.wraps(_ORIGINAL_LOAD)  # type: ignore
    def _patched_load(
        model: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        should_flash = _should_use_flash(model, config)
        if not should_flash:
            assert _ORIGINAL_LOAD is not None
            return _ORIGINAL_LOAD(model, *args, **kwargs)

        assert _MANAGER is not None
        if config.debug:
            import sys
            print(f"[flash] Flash Mode ACTIVE for {model}", file=sys.stderr)

        return _MANAGER.load(model, load_fn=_ORIGINAL_LOAD, **kwargs)

    mlx_lm.load = _patched_load  # type: ignore

def remove_flash_patch() -> None:
    """Restore the original mlx_lm.load (useful for tests)."""
    global _ORIGINAL_LOAD, _MANAGER
    if _ORIGINAL_LOAD is None:
        return
    try:
        import mlx_lm
        mlx_lm.load = _ORIGINAL_LOAD
    except ImportError:
        pass
    if _MANAGER is not None:
        _MANAGER.shutdown()
    _ORIGINAL_LOAD = None
    _MANAGER = None


def _should_use_flash(model_path: str, config: FlashConfig) -> bool:
    """
    Decide whether to activate flash mode for this model load.

    Priority (highest to lowest):
    1. config.enabled is explicitly True — always flash.
    2. config.enabled is explicitly False — never flash.
    3. model config.json contains "flash_mode": true (LM Studio injects this).
    4. model directory contains a Modelfile with FLASH true.
    """
    if not config.enabled:
        return False

    p = Path(model_path)
    if not p.exists():
        return True  # let FlashManager handle the error

    # Check for LM Studio injection in config.json
    cfg_path = p / "config.json"
    if cfg_path.exists():
        import json
        try:
            cfg = json.loads(cfg_path.read_text())
            if "flash_mode" in cfg:
                return bool(cfg["flash_mode"])
        except Exception:
            pass

    # Check for Modelfile directive
    for mf_name in ("Modelfile", "modelfile", "MODELFILE"):
        mf_path = p / mf_name
        if mf_path.exists():
            from .modelfile import parse_flash_directives
            mf_config = parse_flash_directives(mf_path.read_text())
            return mf_config.enabled

    return config.enabled
