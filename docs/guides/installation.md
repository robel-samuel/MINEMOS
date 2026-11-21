# Installation

MINEMOS is published on PyPI as `minemos`. `pip install minemos` installs the
public `minemos` interface and the `minemos_engine` engine package as a normal
non-editable wheel, which is the recommended path for applications. For
development, a source checkout installed in editable mode is described in
[Install from source](#install-from-source).

## Requirements

- Python 3.12 (the version used by CI and the pinned dependencies).
- CPU-only PyTorch is sufficient; no GPU is required.
- Approximately 2 GB of free space for the virtual environment and models.
- Optional: a Qdrant server for the `production` retrieval profile. The
  canonical engine does not need it.

Installation and the full suite are verified on Windows 10/11 with Python
3.12.10. The code paths are platform-neutral and the CI workflow targets Ubuntu;
run the suite on your platform before relying on it there.

## 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

## 2. Install the package

```bash
pip install minemos
```

This installs the runtime dependencies declared in `pyproject.toml` (`numpy`,
`scikit-learn`, `torch`, `transformers`, `certifi`). The canonical engine does
not import or require the Qdrant client.

The default PyPI PyTorch wheel is CPU-capable on Windows and macOS. On Linux it
also bundles the CUDA runtime; to use the smaller CPU-only build instead,
install it first and then MINEMOS:

```bash
pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install minemos
```

This preinstall cannot be automated from the package metadata: a PEP 508
requirement cannot point pip at an additional package index, and requiring
`torch==2.14.0+cpu` would make `pip install minemos` unresolvable on platforms
where only the plain PyPI wheel exists. Because `torch==2.14.0` accepts the
`2.14.0+cpu` local version, the second command keeps the CPU build and works
with the same requirement on every platform.

Optional extras:

```bash
pip install "minemos[qdrant]"    # Qdrant backend
pip install "minemos[test]"      # pytest
pip install "minemos[dev]"       # pytest + pip-audit
```

`requirements.txt` pins the complete development and CI environment, including
transitive packages, and can still be used when an exact environment is
required. Installing `qdrant-client` does not start or connect to a Qdrant
server.

## 3. Models are provisioned on first use

MINEMOS uses two local checkpoints:

- `cross-encoder/nli-distilroberta-base` for the verification gate (~330 MB);
- `sentence-transformers/all-MiniLM-L6-v2` for dense addressing and retrieval
  (~90 MB).

Neither is bundled in the wheel. They are downloaded automatically the first
time an operation needs them, through the normal `transformers` download path,
with its progress bars and resumable files. The download goes to the standard
Hugging Face cache (`~/.cache/huggingface`).

- Importing the package, creating a `BlackHoleMemory`, restoring a checkpoint
  and using `MemoryState`/`Candidate` do not download anything. A first
  single-fact `absorb` loads only the sentence encoder; the NLI model is loaded
  when the first semantic comparison runs (`absorb` with existing slots, or a
  query that needs the dense encoder).
- Later processes read the cached models without contacting the network.
- `HF_HOME` selects a different cache root. `MINEMOS_MODEL_CACHE` (environment)
  or `BlackHoleMemory(model_cache_dir=...)` selects the model directory
  explicitly.
- With `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1` set, a missing model
  raises `minemos.ModelProvisioningError` naming the cache path instead of
  attempting the network. This is the correct setting for air-gapped hosts
  once the cache has been populated elsewhere and copied in.

If a download is interrupted, run the same operation again: the partial files
are resumed rather than downloaded from scratch.

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

On the first run this provisions the sentence encoder (and, from the second
absorb onward, the NLI model). Later runs load them from the cache. If the
download fails, the error names the model, the cache location and the
environment variables that control it.

## 5. Run the tests

The test suite lives in the source repository. Clone it, install the test extra
in editable mode, and run pytest from the repository root:

```bash
git clone https://github.com/robel-samuel/MINEMOS.git
cd MINEMOS
pip install -e ".[test]"
python -m pytest tests/ -q
```

The suite provisions the two models automatically on its first run (or reuses
the cache from earlier runs) and creates temporary Qdrant embedded databases
for the integration tests. To run the Qdrant integration tests against a server
instead:

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

## Install from source

For development, install the checkout in editable mode so changes take effect
without reinstalling:

```bash
git clone https://github.com/robel-samuel/MINEMOS.git
cd MINEMOS
pip install -e ".[qdrant,test]"
```

Add the CPU-only PyTorch wheel first if you are on Linux and do not want the
CUDA runtime that the default PyPI wheel pulls in.

`pip install -e .` registers the public `minemos` interface and the
`minemos_engine` engine package in editable mode.

## Next steps

- [Getting Started](../getting-started.md)
- [Configuration](configuration.md)
- [Troubleshooting](../operations/troubleshooting.md)
