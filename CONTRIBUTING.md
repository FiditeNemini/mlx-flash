# Contributing to mlx-flash

Thank you for your interest! This project targets Apple Silicon Macs running
macOS 13+. Here is how to get started.

## Development setup
```bash
# 1. Fork & clone
git clone https://github.com/YOUR_USERNAME/mlx-flash
cd mlx-flash

# 2. Create a virtual environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"

# 4. Run the test suite against a small 4B model
scripts/test_quick.sh
```

## Pull-request guidelines

* Target the `main` branch.
* Keep each PR focused; one feature / fix per PR.
* All public functions need docstrings and type annotations.
* Run `ruff check . && mypy mlx_engine_flash` before opening a PR.
* Add or update tests for every behavioural change.
* For Metal kernel changes, include before/after `bench_flash.py` numbers.

## Architecture overview

See `docs/architecture.md` for a deep dive into the streaming pipeline.

## Reporting bugs

Open a GitHub Issue with:
1. macOS + chip version (e.g. macOS 14.5, M4 Air 16 GB)
2. Model name & quantisation level
3. Full traceback
4. Output of `python -c "import mlx; print(mlx.__version__)"`
