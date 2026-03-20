#!/usr/bin/env bash
# download_test_model.sh — Download the smallest recommended test model
# Requires: huggingface-hub installed
set -euo pipefail
DEST="${1:-./test_models/Qwen2.5-3B}"
echo "Downloading Qwen2.5-3B-Instruct-4bit (recommended test model)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
'mlx-community/Qwen2.5-3B-Instruct-4bit',
local_dir='$DEST',
ignore_patterns=['.md', 'original/'],
)
print('Downloaded to $DEST')
"
echo ""
echo "✅ Model ready at: $DEST"
echo ""
echo "Run tests:"
echo "  pytest tests/ -v"
echo "  python examples/quick_start.py --model $DEST --flash"
