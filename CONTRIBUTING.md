# Contributing to MINEMOS

Thank you for your interest in contributing. MINEMOS is a research codebase. Contributing here is different from contributing to a production library — the primary concern is **research integrity**, not feature velocity.

---

## Before You Start

Open an issue describing what you intend to do before writing code. This is especially important for:

- Changes to any frozen baseline parameter (these are experimental controls)
- New experiment phases
- Changes to the core update equation
- Changes to the NLI gate or retrieval pipeline

For small bug fixes or documentation improvements, an issue is optional.

---

## Development Setup

```bash
git clone <repository-url>
cd MINEMOS

# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate

pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the required models once:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base")
AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 177 tests must pass before submitting a pull request.

---

## Research Integrity Guidelines

### Frozen Baselines

Parameters marked as "FROZEN" in the source code are experimental controls established in earlier phases. They must not be changed without:

1. Creating a new experiment phase
2. Running against the frozen baseline as a control
3. Documenting both outcomes

Currently frozen (as of Phase 23):

```
w_anchor = 0.15
decay_lambda = 0.005
max_anchors = 3
tau_min = 0.40
tau_high = 0.75
delta_margin = 0.10
k_max = 3
tau_anchor = 0.70
MATCH_THRESHOLD = 0.75
```

### Reporting Results

Any pull request that includes quantitative results must:

- State the dataset size, capacity, and random seed
- Report both positive and negative findings
- Distinguish experimental results from production claims
- Not selectively report results that make the system look better than it is

### Code Comments and Docstrings

Preserve existing comments. They explain design decisions that are not obvious from the code alone — particularly the known limitations documented at the module level.

---

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] No frozen baseline was silently modified
- [ ] Existing docstrings and comments are preserved
- [ ] No secrets, API keys, or personal paths committed
- [ ] `scratch/` output files are not committed
- [ ] If reporting new numbers: dataset, capacity, and seed are documented

---

## What We Are Not Looking For

- Performance optimizations that hide failures
- Silent modifications to research behavior to make tests pass
- Marketing-style descriptions of results
- New dependencies unless clearly necessary
