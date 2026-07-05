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

The project is still evolving, so prefer focused, understandable changes over
large framework-style abstractions.

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

- Add or update pytest tests for behavioral changes.
- Prefer `np.testing.assert_array_equal` or `np.testing.assert_allclose` for
  tensor data comparisons.
- Run the narrowest relevant pytest target first, then broaden if the change
  touches shared behavior.

## Code Style and Design Preferences

- Follow the existing PyTorch-inspired public API where it is already
  established.
- Keep implementation simple and readable; this is an educational library.
- Prefer NumPy operations and structured Tensor/autograd helpers over ad hoc
  special cases.
- Preserve existing module boundaries:
  - Core tensor behavior belongs in `src/motorch/tensor.py`.
  - Gradient graph construction and propagation belongs in
    `src/motorch/autograd/`.
  - Layers, activations, losses, and parameters belong in `src/motorch/nn/`.
  - Optimizers belong in `src/motorch/optim/`.
- Avoid broad refactors unless they are directly needed for the task.
- Keep comments useful and sparse; explain non-obvious math or graph behavior,
  not routine assignments.

## Repository Hygiene

- Do not overwrite unrelated local changes.
- Check `git status --short` before and after edits when changing files.
- Keep changes scoped to the request.
- If a command fails because dependencies are missing, prefer the `uv` workflow
  rather than invoking tools globally.
