# LM Studio Integration Guide

## The Extension Point
LM Studio uses mlx-engine (github.com/lmstudio-ai/mlx-engine) as its inference
backend for Apple Silicon. The key entry point is mlx_lm.load().

We inject apply_flash_patch() into mlx-engine's startup sequence — ideally in
mlx_engine/model_pool.py or wherever mlx_lm.load() is called.

## Proposed Change to mlx-engine
```python
# mlx_engine/model_pool.py (mlx-engine upstream)

# --- EXISTING CODE ---
import mlx_lm

def load_model(model_path: str, config: dict) -> tuple:
    return mlx_lm.load(model_path)

# --- PROPOSED ADDITION (Flash Mode PR) ---
def load_model(model_path: str, config: dict) -> tuple:
    flash_enabled = config.get("flash_mode", False)
    if flash_enabled:
        from mlx_flash import FlashConfig
        from mlx_flash.integration.lmstudio import apply_flash_patch
        flash_cfg = FlashConfig(
            enabled=True,
            ram_budget_gb=config.get("flash_ram_gb", 10.0),
            n_io_threads=config.get("flash_threads", 4),
        )
        apply_flash_patch(flash_cfg)
    return mlx_lm.load(model_path)
```

## UI Hook (LM Studio Electron app)
The LM Studio UI is Electron/React. The checkbox would add a flash_mode: true
key to the inference config JSON passed to the mlx-engine subprocess.

The config key name flash_mode matches what _should_use_flash() in lmstudio.py
looks for in config.json.

## Testing without LM Studio
```bash
# Simulate LM Studio passing flash_mode=True
python -c "
import json, pathlib, sys
# Inject flash_mode into the model's config.json temporarily
p = pathlib.Path('/path/to/model/config.json')
cfg = json.loads(p.read_text())
cfg['flash_mode'] = True
p.write_text(json.dumps(cfg))
print('flash_mode injected — now load the model in LM Studio')
"
```
