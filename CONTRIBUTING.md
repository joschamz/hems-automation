# Contributing

Thanks for your interest in improving this project.

## How to contribute

1. Open an issue describing the bug or proposed enhancement.
2. Create a branch from `main`.
3. Keep changes focused and small.
4. Run tests and checks locally before opening a pull request.
5. Open a pull request with a clear description and rationale.

## Local setup

```bash
./scripts/setup.sh
source .venv/bin/activate
pip install -e ".[dev]"
```

## Test expectations

Run the existing test workflow locally before creating a PR:

```bash
pytest ./.github/workflows/testing/test_import_libraries.py -q
```

If your change touches core logic in `utils/` or `src/`, add or update tests accordingly.

## Pull request checklist

- [ ] Scope is limited to the intended change.
- [ ] Documentation is updated when behavior changes.
- [ ] No secrets are committed.
- [ ] CI is expected to pass on all target platforms.
