---
name: Bug report
about: Something isn't working
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what went wrong.

**Model & Hardware**
- Model: (e.g. `mlx-community/Meta-Llama-3-70B-Instruct-4bit`)
- Mac: (e.g. MacBook Air M4, 16 GB)
- macOS version: (e.g. 15.2)
- MLX version: `python -c "import mlx.core as mx; print(mx.__version__)"`
- mlx-lm version: `python -c "import mlx_lm; print(mlx_lm.__version__)"`
- mlx-flash version: `python -c "import mlx_flash; print(mlx_flash.__version__)"`

**Minimal reproduction**
```python
# paste the smallest code that triggers the bug
```

**Full traceback**
```
paste traceback here
```

**Flash config used**
```python
FlashConfig(
    enabled=True,
    ram_budget_gb=...,
    # etc
)
```

**Expected behavior**
What did you expect to happen?
