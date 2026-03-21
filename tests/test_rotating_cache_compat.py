
from mlx_flash import FlashConfig
from mlx_flash.generation import FlashGenerationLoop


def test_rotating_cache_constructor_safety(tmp_model_dir):
    """Verify that RotatingKVCache is constructed without error."""
    import mlx_lm
    model, tokenizer = mlx_lm.load(str(tmp_model_dir))
    
    # Case 1: Max size provided
    config = FlashConfig(enabled=True, max_kv_size=1024)
    loop = FlashGenerationLoop(model, tokenizer, config)
    
    assert loop._cache is not None
    assert len(loop._cache) == 2 # 2 layers in synthetic model
    
    # Verify we can run a step
    prompt = "Test"
    gen = loop.stream_generate(prompt, max_tokens=1)
    next(gen)
