from mlx_flash import FlashConfig, FlashModelLoader
from pathlib import Path

def check_names():
    model_path = "/Users/granite/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit"
    config = FlashConfig(enabled=True)
    loader = FlashModelLoader(Path(model_path), config)
    loader.open()
    
    names = loader._streamer.index.layer_tensor_names(0)
    print(f"Sample names for layer 0: {names[:3]}")
    
    loader.close()

if __name__ == "__main__":
    check_names()
