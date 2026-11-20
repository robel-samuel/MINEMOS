# Contributing to MINEMOS

MINEMOS is an open-source memory engine. Contributions that fix lifecycle
defects, improve retrieval, add test coverage, or document existing behavior are
welcome.

## Before you start

Open an issue describing the change before writing code when it affects:

- memory decisions (consolidation, contradiction, supersession, retention);
- the verification gate or retrieval pipeline;
- checkpoint format or compatibility;
- public API behavior.

Small bug fixes and documentation improvements do not need an issue.

## Development setup

```bash
git clone <repository-url>
cd MINEMOS

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
```

Use `pip install -e ".[qdrant,test]"` when working on the Qdrant backend, or
`pip install -r requirements.txt` for the exact pinned CI environment.

Download the two models once while online:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base")
AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

## Running tests

```bash
python -m pytest tests/ -q
```

The Qdrant integration tests use an embedded database by default. To run them
against a server:

```bash
MINEMOS_QDRANT_URL=http://127.0.0.1:6333 \
  python -m pytest tests/test_qdrant_integration.py -q
```

A change is ready when the full suite passes. If a test fails only under heavy
load, reproduce it in isolation and report the conditions instead of weakening
the assertion.

## Engineering guidelines

### Memory decisions are governed behavior

Consolidation bounds, anchor weights, verification thresholds, and fusion
weights are part of the engine's behavior. Do not change them as a side effect
of an unrelated change. If a change intentionally alters memory decisions, add
tests that pin the new behavior and explain the trade-off in the pull request.

### Correctness before speed

Retrieval and lifecycle paths are concurrent. Keep the snapshot and
cache-invalidation contracts intact: mutate canonical state through engine
methods, never by writing `MemorySlot` fields or `memory.slots` directly.

### Results need context

If a pull request includes measurements, state the dataset, capacity, and seed,
and report negative results as well as positive ones. Do not present a result
from one corpus as a general property of the engine.

### Code comments

Preserve comments that explain a non-obvious constraint, invariant, or
workaround. Do not add comments that restate what the code does.

## Pull request checklist

- [ ] Tests pass (`python -m pytest tests/ -q`)
- [ ] New behavior has a regression test
- [ ] No checkpoint-format change without a compatibility note
- [ ] No secrets, API keys, or personal paths committed
- [ ] Public documentation updated when the API changes
- [ ] Generated artifacts and local data directories are not committed

## License

By contributing, you agree that your contributions are licensed under the
Apache License, Version 2.0, the license of this repository.