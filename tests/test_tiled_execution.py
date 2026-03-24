import mlx.core as mx
import mlx.nn as nn
from mlx_flash.config import FlashConfig
from mlx_flash.tiled import apply_tiling, TiledColumnLinear, TiledRowLinear

def test_tiled_column_linear():
    mx.random.seed(42)
    original = nn.Linear(32, 64)
    tiled = TiledColumnLinear(original, tile_size=16)
    
    x = mx.random.uniform(shape=(4, 32))
    
    out_orig = original(x)
    out_tiled = tiled(x)
    
    mx.eval(out_orig, out_tiled)
    assert mx.allclose(out_orig, out_tiled, rtol=1e-5, atol=1e-5)

def test_tiled_row_linear():
    mx.random.seed(42)
    original = nn.Linear(64, 32)
    tiled = TiledRowLinear(original, tile_size=16)
    
    x = mx.random.uniform(shape=(4, 64))
    
    out_orig = original(x)
    out_tiled = tiled(x)
    
    mx.eval(out_orig, out_tiled)
    assert mx.allclose(out_orig, out_tiled, rtol=1e-5, atol=1e-5)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(32, 64)
        self.down_proj = nn.Linear(64, 32)

def test_apply_tiling():
    model = DummyModel()
    apply_tiling(model, tile_size=16)
    
    assert isinstance(model.up_proj, TiledColumnLinear)
    assert isinstance(model.down_proj, TiledRowLinear)
    
    x = mx.random.uniform(shape=(4, 32))
    out = model.down_proj(model.up_proj(x))
    mx.eval(out)
    assert out.shape == (4, 32)
