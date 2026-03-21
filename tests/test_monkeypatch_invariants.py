
import mlx_lm
import mlx.core as mx
import pytest
from mlx_flash import FlashConfig
from mlx_flash.integration.lmstudio import apply_flash_patch, remove_flash_patch

def test_apply_flash_patch_disabled_does_not_clobber_metal():
    """Verify that enabled=False doesn't touch mx.metal limits."""
    # Reset any existing patch
    remove_flash_patch()
    
    orig_cache_limit = mx.metal.set_cache_limit
    orig_wired_limit = mx.metal.set_wired_limit
    
    config = FlashConfig(enabled=False)
    apply_flash_patch(config)
    
    # In my current implementation, I save originals then check enabled.
    # If disabled, I DON'T override.
    assert mx.metal.set_cache_limit is orig_cache_limit
    assert mx.metal.set_wired_limit is orig_wired_limit
    
    remove_flash_patch()

def test_apply_flash_patch_enabled_clobbers_metal():
    """Verify that enabled=True DOES override mx.metal limits."""
    remove_flash_patch()
    
    orig_cache_limit = mx.metal.set_cache_limit
    
    config = FlashConfig(enabled=True)
    apply_flash_patch(config)
    
    assert mx.metal.set_cache_limit is not orig_cache_limit
    # Should be a no-op lambda
    assert mx.metal.set_cache_limit(1024) is None
    
    remove_flash_patch()

@pytest.mark.real_model
def test_integration_stream_generate_has_text_attr(tmp_model_dir):
    """Verify that patch -> stream_generate yields objects with .text."""
    from mlx_flash import FlashConfig
    from mlx_flash.integration.lmstudio import apply_flash_patch
    
    apply_flash_patch(FlashConfig(enabled=True))
    
    # Use synthetic model from conftest
    model, tokenizer = mlx_lm.load(str(tmp_model_dir))
    
    # stream_generate should now yield objects with .text
    stream = mlx_lm.stream_generate(model, tokenizer, "test")
    first_chunk = next(stream)
    
    assert hasattr(first_chunk, "text")
    assert isinstance(first_chunk.text, str)
    
    remove_flash_patch()
