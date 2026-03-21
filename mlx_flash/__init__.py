"""
mlx-flash
=====================
Flash Weight Streaming for LM Studio / mlx-engine.

Typical usage inside mlx-engine (or any mlx-lm wrapper):

    from mlx_flash import FlashConfig, FlashManager
    from mlx_flash.integration.lmstudio import apply_flash_patch

    config = FlashConfig(enabled=True, ram_budget_gb=10.0, n_io_threads=4)
    apply_flash_patch(config)          # patches mlx_lm.load globally
    # ... rest of mlx-engine startup unchanged

Or via Modelfile:

    from mlx_flash.integration.modelfile import parse_flash_directives
    config = parse_flash_directives(open("Modelfile").read())
"""

from .config import FlashConfig
from .generation import FlashGenerationLoop, FlashLLM
from .manager import FlashManager


__all__ = [
    "FlashConfig",
    "FlashManager",
    "FlashLLM",
    "FlashGenerationLoop",
]

__version__ = "0.1.0"
