# Installation

MINEMOS runs from the repository checkout. It is not published as an installable
package, so examples assume the repository root is the working directory.

## Requirements

- Python 3.12 (the version used by CI and the pinned dependencies).
- CPU-only PyTorch is sufficient; no GPU is required.
- Approximately 2 GB of free space for the virtual environment and models.
- Optional: a Qdrant server for the `production` retrieval profile. The
  canonical engine does not need it.

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

## 2. Install Python dependencies

Install CPU-only PyTorch first from the PyTorch index, then the remaining pinned
requirements:

```bash
pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` also pins `qdrant-client`; installing it does not start or
connect to a Qdrant server. The canonical backend never imports the client.

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

From the repository root:

```python
from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate

mem = BlackHoleMemory(capacity=50)
result = mem.absorb(Candidate("alice", "works_at", "university", timestamp=1.0))
print(result["action"])

rows = mem.query("where does alice work", top_k=3)
print(rows[0]["text"], rows[0]["similarity"])
```

The first run loads both models and may take a few seconds. If the models are
missing, run step 3 again or check the Hugging Face cache location.

## 5. Run the tests

```bash
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
[Configuration](configuration.md); deployment rules are in
[Deployment](deployment.md).

Embedded mode (`QDRANT_PATH`) is supported for development and tests. It is
rejected by `retrieval_mode="production"`.

## Next steps

- [Configuration](configuration.md)
- [Quick start in the README](../../README.md#quick-start)
- [Troubleshooting](../operations/troubleshooting.md)
