#!/usr/bin/env bash
# install.sh — Set up mlx-flash development environment
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV=".venv"

echo "🔧  mlx-flash installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: Flash Mode requires macOS + Apple Silicon."
    echo "   Non-macOS install proceeds without page-cache optimisations."
fi

# Check Python version
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.11"
if [[ "$(printf '%s\n' "$REQUIRED" "$PYVER" | sort -V | head -n1)" != "$REQUIRED" ]]; then
    echo "❌  Python $REQUIRED+ required; found $PYVER"
    exit 1
fi
echo "✓  Python $PYVER"

# Create venv
if [[ ! -d "$VENV" ]]; then
    "$PYTHON" -m venv "$VENV"
    echo "✓  Created virtualenv $VENV"
fi
source "$VENV/bin/activate"
echo "✓  Activated $VENV"

# Install
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet
echo "✓  Installed mlx-flash[dev]"

# Optional: compile Metal kernels
if command -v xcrun &>/dev/null; then
    echo ""
    echo "Xcode CLT found — compiling Metal kernels..."
    python mlx_flash/kernels/compile_kernels.py && echo "✓  Metal kernels compiled"
else
    echo "⚠️  Xcode CLT not found — Metal kernels will use MLX fallbacks."
    echo "   Install with: xcode-select --install"
fi

echo ""
echo "✅  Installation complete!"
echo ""
echo "Next steps:"
echo "  source $VENV/bin/activate"
echo "  pytest tests/ -v"
echo "  python examples/quick_start.py --model /path/to/model --flash"
