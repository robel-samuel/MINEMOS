# Installation

MINEMOS installs as a Python package from the repository. `pip install -e .`
registers both the public `minemos` interface and the `minemos_engine` engine
package in editable mode, so edits in the checkout take effect immediately.

## Requirements

- Python 3.12 (the version used by CI and the pinned dependencies).
- CPU-only PyTorch is sufficient; no GPU is required.
- Approximately 2 GB of free space for the virtual environment and models.
- Optional: a Qdrant server for the `production` retrieval profile. The
  canonical engine does not need it.

Installation and the full suite are verified on Windows 10/11 with Python
3.12.10. The code paths are platform-neutral and the CI workflow targets Ubuntu;
run the suite on your platform before relying on it there.

## 1. Clone and create a virtual environment

```bash
git clone https://github.com/robel-samuel/MINEMOS.git
cd MINEMOS

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

## 2. Install the package

Install CPU-only PyTorch first from the PyTorch index, then install MINEMOS in
editable mode:

```bash
pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

`pyproject.toml` declares the runtime dependencies (`numpy`, `scikit-learn`,
`torch`, `transformers`, `certifi`). The canonical engine does not import or
require the Qdrant client.

Optional extras:

```bash
pip install -e ".[qdrant]"    # Qdrant backend
pip install -e ".[test]"      # pytest
pip install -e ".[dev]"       # pytest + pip-audit
```

`requirements.txt` pins the complete development and CI environment, including
transitive packages, and can still be used when an exact environment is
required. Installing `qdrant-client` does not start or connect to a Qdrant
server.

## 3. Download the two models

MINEMOS uses:

- `cross-encoder/nli-distilroberta-base` for the verification gate;
- `sentence-transformers/all-MiniLM-L6-v2` for dense addressing and retrieval.

Download them once while online:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base")
AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("Models ready.")
PY
```

Models are cached under `~/.cache/huggingface` by default. Set `HF_HOME` before
this step to place them elsewhere. For offline runs after caching, set
`TRANSFORMERS_OFFLINE=1`.

## 4. Smoke test

```python
from minemos import BlackHoleMemory, Candidate

mem = BlackHoleMemory(capacity=50)
result = mem.absorb(Candidate("alice", "works_at", "university",
                              timestamp=1.0))
print(result["action"])

rows = mem.query("where does alice work", top_k=3)
print(rows[0]["text"], rows[0]["similarity"])
```

The first run loads both models and may take a few seconds. If the models are
missing, run step 3 again or check the Hugging Face cache location.

## 5. Run the tests

Install the test extra first if it was skipped above, then run the suite from
the repository root:

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

The suite uses the model cache from step 3 and creates temporary Qdrant embedded
databases for the integration tests. To run the Qdrant integration tests against
a server instead:

```bash
MINEMOS_QDRANT_URL=http://127.0.0.1:6333 \
  python -m pytest tests/test_qdrant_integration.py -q
```

## Optional Qdrant setup

The `production` retrieval profile requires server-mode Qdrant. Start one with
the official container image, for example:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Then set `QDRANT_URL` (and `QDRANT_API_KEY` for a secured remote host) before
starting MINEMOS. Configuration details are in
[Configuration](configuration.md); the index contract is in
[Qdrant Integration](../architecture/qdrant-integration.md).

Embedded mode (`QDRANT_PATH`) is supported for development and tests. It is
rejected by `retrieval_mode="production"`.

## Next steps

- [Getting Started](../getting-started.md)
- [Configuration](configuration.md)
- [Troubleshooting](../operations/troubleshooting.md)
