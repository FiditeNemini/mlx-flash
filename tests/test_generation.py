from mlx_engine_flash.config import FlashConfig
from mlx_engine_flash.generation import FlashGenerationLoop


def test_flash_generation_non_empty(tmp_model_dir):
    config = FlashConfig(enabled=True, debug=True)
    loop = FlashGenerationLoop(tmp_model_dir, config)
    
    try:
        # Prompt "Hello" should produce some tokens
        gen = loop.stream_generate("Hello", max_tokens=5)
        tokens = list(gen)
        
        assert len(tokens) > 0, "Expected non-empty output from generator"
        assert all(isinstance(t, str) for t in tokens)
    finally:
        pass
