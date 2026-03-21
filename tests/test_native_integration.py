
import mlx.core as mx
import mlx_lm

from mlx_flash import FlashConfig
from mlx_flash.integration.lmstudio import apply_flash_patch, remove_flash_patch


def test_native_stream_generate_with_patch(tmp_model_dir):
    """Verify that mlx_lm.stream_generate works with patched FlashLLM."""
    # 1. Apply patch
    config = FlashConfig(enabled=True)
    apply_flash_patch(config)
    
    try:
        # 2. Load model (this will return a FlashLLM because of the patch)
        model, tokenizer = mlx_lm.load(str(tmp_model_dir))
        from mlx_flash.generation import FlashLLM
        assert isinstance(model, FlashLLM)
        
        # 3. Use standard mlx_lm.stream_generate
        prompt = "Hello"
        responses = list(mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=5))
        
        assert len(responses) > 0
        assert all(hasattr(r, "text") for r in responses)
        
        # 4. Verify memory was released (mx.eval would have happened)
        # If it wasn't synchronous, memory would likely still be 'active'
        # but here we just check it doesn't crash.
        print(f"Generated {len(responses)} tokens successfully.")
        
    finally:
        remove_flash_patch()

def test_prefill_mask_correctness(tmp_model_dir):
    """Verify that FlashLLM handles prefill (T > 1) via create_attention_mask."""
    from mlx_lm.models.cache import make_prompt_cache

    from mlx_flash.generation import FlashLLM
    model, tokenizer = mlx_lm.load(str(tmp_model_dir))
    flash_model = FlashLLM(model, FlashConfig(enabled=True))
    
    # Large prompt to trigger prefill mask
    prompt = "This is a longer prompt to ensure create_attention_mask is called. " * 5
    tokens = mx.array(tokenizer.encode(prompt))[None]
    
    # Pass a cache so the mask construction path (mask is None and cache is not None)
    # is actually exercised via create_attention_mask.
    cache = make_prompt_cache(model)
    logits = flash_model(tokens, cache=cache)
    assert logits.shape[1] == tokens.shape[1]
    assert not mx.isnan(logits).any()
