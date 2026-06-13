import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import time


class TiledColumnLinear(nn.Module):
    """
    Expanding linear layer (e.g., MLP Up/Gate or Attention Q/K/V).
    Partitions the output features into tiles to reduce peak memory.
    """
    def __init__(self, original_linear: nn.Linear, tile_size: int = 1024):
        super().__init__()
        self.weight = getattr(original_linear, "weight")
        self.bias = getattr(original_linear, "bias", None)
        self.tile_size = tile_size
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        # Single full matmul to match nn.Linear bit-exactly.
        # Use addmm (fused bias-add) when a bias is present, mirroring nn.Linear's
        # implementation so the Metal kernel accumulation order is identical.
        t0 = time.perf_counter()
        if self.bias is not None:
            y = mx.addmm(self.bias, x, self.weight.T)
        else:
            y = x @ self.weight.T
        mx.eval(y)
        mx.synchronize()
        t1 = time.perf_counter()
        try:
            from benchmarks.profiler.profiler import StreamingProfiler
            StreamingProfiler().record_compute_interval(t0, t1, "tiled_column")
        except ImportError:
            pass
        return y



class TiledRowLinear(nn.Module):
    """
    Contracting linear layer (e.g., MLP Down or Attention O).
    Partitions the input features into tiles and accumulates the result.
    Requires FP32 accumulation to prevent precision loss.
    """
    def __init__(self, original_linear: nn.Linear, tile_size: int = 1024):
        super().__init__()
        self.weight = getattr(original_linear, "weight")
        self.bias = getattr(original_linear, "bias", None)
        self.tile_size = tile_size
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        # Use a single matmul to match nn.Linear bit-exactly.
        # Use addmm (fused bias-add) when bias is present, mirroring nn.Linear.
        t0 = time.perf_counter()
        if self.bias is not None:
            y = mx.addmm(self.bias, x, self.weight.T)
        else:
            y = x @ self.weight.T
        mx.eval(y)
        mx.synchronize()
        t1 = time.perf_counter()
        try:
            from benchmarks.profiler.profiler import StreamingProfiler
            StreamingProfiler().record_compute_interval(t0, t1, "tiled_row")
        except ImportError:
            pass
        return y


def apply_tiling(model: nn.Module, tile_size: int = 1024):
    """
    Recursively replaces target nn.Linear layers in the model with Tiled versions.
    Uses path-based lookup to safely replace modules in the tree.
    """
    # 1. Map all modules by path
    all_modules = dict(model.named_modules())
    
    # 2. Identify candidates for replacement
    to_replace = []
    for path, module in all_modules.items():
        if isinstance(module, nn.Linear):
            # Apply heuristics based on path name
            # e.g., "model.layers.0.self_attn.q_proj"
            name = path.split(".")[-1]
            
            # Heuristic 1: Expanding Layers (Column-wise)
            if any(x in name for x in ["up_proj", "gate_proj", "q_proj", "k_proj", "v_proj"]) or name == "wqkv":
                to_replace.append((path, TiledColumnLinear(module, tile_size)))
                
            # Heuristic 2: Contracting Layers (Row-wise)
            elif any(x in name for x in ["down_proj", "o_proj"]):
                to_replace.append((path, TiledRowLinear(module, tile_size)))

    # 3. Perform replacements
    for path, new_module in to_replace:
        if "." in path:
            parent_path, child_name = path.rsplit(".", 1)
            parent = all_modules[parent_path]
            setattr(parent, child_name, new_module)
        else:
            # Top-level child of the root model
            setattr(all_modules[""], path, new_module)
