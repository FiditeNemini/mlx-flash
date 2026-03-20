#!/usr/bin/env bash
# uninstall.sh — Cleanly remove mlx-flash and its artifacts
set -euo pipefail

echo "🗑️  mlx-flash uninstaller"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Uninstall python package
if command -v pip &>/dev/null; then
    echo "Uninstalling mlx-flash package..."
    pip uninstall -y mlx-flash || true
fi

# 2. Remove virtual environment
if [[ -d ".venv" ]]; then
    echo "Removing virtualenv .venv..."
    rm -rf .venv
fi

# 3. Clean up Metal kernels
echo "Cleaning up compiled Metal kernels..."
find mlx_engine_flash/kernels -name "*.metallib" -delete

# 4. Clean up Python cache
echo "Cleaning up Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf *.egg-info .pytest_cache .mypy_cache

echo ""
echo "✅  Uninstallation complete!"
