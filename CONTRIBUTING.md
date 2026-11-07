# Contributing

Thanks for taking the time to improve rag-eval-lab.

## Development setup

```bash
git clone https://github.com/s78519300-creator/rag-eval-lab.git
cd rag-eval-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
pytest
rag-lab eval --docs examples/docs --evalset examples/evalset.jsonl
```

- Keep the core free of third-party runtime dependencies.
- Add or update tests for any behaviour change.
- If a change affects retrieval or generation, include the before/after
  `rag-lab eval` summary in the PR description. A metric that drops should be
  explained, not hidden.
- Add a line to `CHANGELOG.md` under "Unreleased".

## Reporting bugs

Open an issue with the command you ran, the output you got, and what you
expected. A minimal document and question that reproduce the problem help a lot.
