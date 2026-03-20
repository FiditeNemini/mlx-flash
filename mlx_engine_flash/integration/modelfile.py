"""
modelfile.py — Parse Flash Mode directives from Ollama-style Modelfiles.

Supported directives (case-insensitive, value is the argument after the key):

    FLASH true|false|1|0|yes|no
    FLASH_RAM_GB <float>          # RAM budget for page-resident weights
    FLASH_THREADS <int>           # parallel pread thread count
    FLASH_PREFETCH_LAYERS <int>   # how many layers ahead to prefetch
    FLASH_QUANT_WARN_BELOW <int>  # warn if quant bits < this value
    FLASH_TOP_K <int>             # override MoE top-K (0 = use model default)
    FLASH_EVICTION dontneed|free|none

Unknown FLASH_* keys are silently ignored.

Example Modelfile::

    FROM /models/DeepSeek-R1-671B-Q4_K_M
    SYSTEM "You are a helpful assistant."
    FLASH true
    FLASH_RAM_GB 12
    FLASH_THREADS 6
    FLASH_PREFETCH_LAYERS 3
"""

from __future__ import annotations

import re
from typing import Optional

from ..config import FlashConfig


def parse_flash_directives(modelfile_text: str) -> FlashConfig:
    """
    Parse FLASH_* directives from a Modelfile string.

    Returns a FlashConfig; enabled=False if no FLASH directive found.
    """
    cfg_dict: dict = {}
    enabled: Optional[bool] = None

    for line in modelfile_text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue

        parts = line.split(None, 1)
        if len(parts) < 2:
            continue

        key, value = parts[0].upper(), parts[1].strip()

        if key == "FLASH":
            enabled = _parse_bool(value)
        elif key == "FLASH_RAM_GB":
            try:
                cfg_dict["ram_budget_gb"] = float(value)
            except ValueError:
                pass
        elif key == "FLASH_THREADS":
            try:
                cfg_dict["n_io_threads"] = int(value)
            except ValueError:
                pass
        elif key == "FLASH_PREFETCH_LAYERS":
            try:
                cfg_dict["prefetch_layers"] = int(value)
            except ValueError:
                pass
        elif key == "FLASH_QUANT_WARN_BELOW":
            try:
                cfg_dict["min_quant_bits"] = int(value)
            except ValueError:
                pass
        elif key == "FLASH_TOP_K":
            try:
                k = int(value)
                cfg_dict["moe_top_k_override"] = k if k > 0 else None
            except ValueError:
                pass
        elif key == "FLASH_EVICTION":
            v = value.lower()
            if v in ("dontneed", "free", "none"):
                cfg_dict["eviction_strategy"] = v
        # Silently ignore unknown FLASH_* keys

    cfg_dict["enabled"] = enabled if enabled is not None else False
    return FlashConfig.from_dict(cfg_dict)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes", "on")
