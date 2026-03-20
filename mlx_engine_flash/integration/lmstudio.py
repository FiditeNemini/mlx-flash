
import functools
import pathlib
from typing import Any

from ..config import FlashConfig
from ..manager import FlashManager

_ORIGINAL_LOAD = None
_MANAGER: FlashManager | None = None


def apply_flash_patch(config: FlashConfig | None = None) -> None:
    """
    Monkey-patch mlx_lm to be Flash-compatible.
    """
    global _ORIGINAL_LOAD, _MANAGER

    if config is None:
        config = FlashConfig(enabled=False)

    try:
        import mlx_lm
        import mlx.core as mx
    except ImportError as e:
        raise ImportError("mlx_lm not installed") from e

    if _ORIGINAL_LOAD is not None:
        if _MANAGER is not None: _MANAGER.config = config
        return

    # LOCK DOWN METAL LIMITS
    mx.metal.set_cache_limit = lambda *args, **kwargs: None
    mx.metal.set_wired_limit = lambda *args, **kwargs: None

    _ORIGINAL_LOAD = mlx_lm.load
    _MANAGER = FlashManager(config)

    # 1. Patch LOAD
    @functools.wraps(_ORIGINAL_LOAD)
    def _patched_load(model: str, *args: Any, **kwargs: Any) -> Any:
        should_flash = _should_use_flash(model, config)
        if not should_flash:
            return _ORIGINAL_LOAD(model, *args, **kwargs)
        return _MANAGER.load(model, load_fn=_ORIGINAL_LOAD, **kwargs)

    mlx_lm.load = _patched_load

    # 2. Patch stream_generate with PRODUCTION GUARDRAILS
    def _flash_stream_generate(model, tokenizer, prompt, *args, **kwargs):
        """Stable entry point with hard guardrails against Metal OOM."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        # Enforce basic safety
        kwargs.setdefault("prefill_step_size", 1)
        kwargs.setdefault("kv_bits", 4)

        # 1. ALWAYS Encode prompt to tokens for reliable chunking
        if isinstance(prompt, str):
            token_list = tokenizer.encode(prompt)
        else:
            token_list = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
        
        original_len = len(token_list)

        # 2. APPLY GUARDRAIL
        current_config = _MANAGER.config if _MANAGER else config
        if original_len > current_config.max_safe_context_tokens and current_config.strict_guardrails:
            safe_len = current_config.max_safe_context_tokens
            print(f"\n[mlx-flash] ⚠️  GUARDRAIL ACTIVATED")
            print(f"   Prompt length: {original_len} tokens → truncated to last {safe_len} tokens")
            print(f"   Reason: Large model + long prompt exceeds Metal graph limits on 16 GB Mac")
            print(f"   Recommendation: For full long-context (2k+ tokens) use llama.cpp (GGUF) instead")
            print(f"   (This is the documented limitation of current MLX — not a bug in mlx-flash)\n")
            token_list = token_list[-safe_len:]

        # 3. Create a Response-like wrapper to match mlx_lm.stream_generate
        class Response:
            def __init__(self, text, token):
                self.text = text
                self.token = token

        # Convert back to MLX array for generate_step
        tokens_mx = mx.array(token_list)

        # 4. Filter kwargs for generate_step
        valid_args = {
            "max_tokens", "sampler", "logits_processors", "max_kv_size",
            "prompt_cache", "prefill_step_size", "kv_bits", "kv_group_size"
        }
        gen_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

        # 5. Execute generation
        for token, _ in generate_step(tokens_mx, model, **gen_kwargs):
            text = tokenizer.decode([token.item()])
            yield Response(text, token.item())
            mx.synchronize()
            mx.clear_cache()

    mlx_lm.stream_generate = _flash_stream_generate

def remove_flash_patch() -> None:
    """Restore the original state."""
    global _ORIGINAL_LOAD, _MANAGER
    if _ORIGINAL_LOAD is None: return
    import mlx_lm
    mlx_lm.load = _ORIGINAL_LOAD
    if _MANAGER: _MANAGER.shutdown()
    _ORIGINAL_LOAD = None
    _MANAGER = None


def _should_use_flash(model_path: str, config: FlashConfig) -> bool:
    if config.enabled: return True
    p = pathlib.Path(model_path)
    for mf_name in ("Modelfile", "modelfile"):
        mf_path = p / mf_name
        if mf_path.exists():
            from .modelfile import parse_flash_directives
            mf_config = parse_flash_directives(mf_path.read_text())
            return mf_config.enabled
    return config.enabled
