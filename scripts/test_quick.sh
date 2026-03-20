#!/usr/bin/env bash
# test_quick.sh — Run the fast test suite + optional real-model smoke test
# Usage: scripts/test_quick.sh [MODEL_PATH]
set -euo pipefail
MODEL="${1:-}"
echo "⚡ Flash Mode — Quick Test Runner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Unit + integration tests (no model needed)
echo ""
echo "Running unit tests..."
pytest tests/test_config.py tests/test_streamer.py tests/test_moe.py \
tests/test_loader.py -v --tb=short

if [[ -n "$MODEL" ]]; then
echo ""
echo "Running smoke test against: $MODEL"
python examples/quick_start.py \
    --model "$MODEL" \
    --flash \
    --max-tokens 5 \
    --benchmark
else
echo ""
echo "💡  Pass a model path to run a real-model smoke test:"
echo "    $0 /path/to/your/model"
fi

echo ""
echo "✅  All done!"
