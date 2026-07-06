# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

MoTorch is a small, educational deep learning library built on NumPy and
inspired by PyTorch. It currently provides:

- A `Tensor` wrapper with NumPy interoperability and automatic differentiation.
- Autograd helpers under `src/motorch/autograd/`.
- Neural network layers, activations, loss functions, and parameters under
  `src/motorch/nn/`.
- Optimizers under `src/motorch/optim/`.
- Tests under `tests/`, grouped roughly by subsystem.
- Example notebooks under `examples/`.

## Environment and Tooling

This project uses `uv`.

Common commands:

```bash
uv run pytest
uv run pytest tests/path/to/test_file.py
uv run ruff check .
uv run ruff format .
```

Notes:

- Use Ruff for both linting and formatting when making changes.
- Development dependencies are declared in `pyproject.toml` under
  `[dependency-groups].dev`.

## Testing Expectations

- Only use the public API in tests if possible; do not test internals unless necessary
- Prefer `np.testing.assert_array_equal` or `np.testing.assert_allclose` for
  tensor data comparisons.

## Repository Hygiene

- Do not overwrite unrelated local changes.
- Check `git status --short` before and after edits when changing files.
- Keep changes scoped to the request.
- If a command fails because dependencies are missing, prefer the `uv` workflow
  rather than invoking tools globally.
