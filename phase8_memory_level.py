"""
benchmarks/phase8_memory_level.py

System A: atomic encoder + Black Hole (control)
System B: semantic embedding + Black Hole (full update mechanism)
System C: semantic embedding + NO UPDATE (isolates whether any benefit
          comes from the embedding itself vs. the update mechanism)

Same candidates, same ordering, same capacity, same threshold (frozen,
predefined 0.75) for all three.
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState
from phase2.memory_update import absorb, cosine_similarity, MATCH_THRESHOLD
from phase2.checkpoint import save_checkpoint
from phase2.encoders import AtomicEncoder, SemanticEmbeddingEncoder
from benchmarks.phase8_dataset import PAIRS

D = 4096  # atomic encoder's dimension
SEMANTIC_DIM = 300  # fixed by the pretrained model -- not a free parameter


def _dim_for(encoder) -> int:
    return SEMANTIC_DIM if isinstance(encoder, SemanticEmbeddingEncoder) else D


def merged_status(pair, encoder, threshold):
    dim = _dim_for(encoder)
    m = MemoryState(capacity=10, dim=dim)
    absorb(m, pair.c1, threshold=threshold, encoder=encoder)
    r2 = absorb(m, pair.c2, threshold=threshold, encoder=encoder)
    return r2["action"] == "update"


def run_system(name: str, encoder, use_update: bool, out_dir: str) -> dict:
    dim = _dim_for(encoder)
    para_pairs = [p for p in PAIRS if p.category == "paraphrase"]
    contra_pairs = [p for p in PAIRS if p.category == "contradiction"]
    unrel_pairs = [p for p in PAIRS if p.category == "unrelated"]
    exact_pairs = [p for p in PAIRS if p.category == "exact_duplicate"]

    def merged(pair):
        if use_update:
            return merged_status(pair, encoder, MATCH_THRESHOLD)
        return False  # no-update never merges, by definition

    exact_rate = sum(1 for p in exact_pairs if merged(p)) / len(exact_pairs)
    para_rate = sum(1 for p in para_pairs if merged(p)) / len(para_pairs)
    contra_rate = sum(1 for p in contra_pairs if merged(p)) / len(contra_pairs)
    unrel_rate = sum(1 for p in unrel_pairs if merged(p)) / len(unrel_pairs)

    # full stream through one memory
    mem = MemoryState(capacity=300, dim=dim)
    t0 = time.perf_counter()
    n_updates = n_inserts = 0
    for pair in PAIRS:
        for c in (pair.c1, pair.c2):
            if use_update:
                r = absorb(mem, c, threshold=MATCH_THRESHOLD, encoder=encoder)
                if r["action"] == "update":
                    n_updates += 1
                else:
                    n_inserts += 1
            else:
                x_t = encoder.encode_full(c, dim=dim)
                mem.add(x_t, x_t.copy(), c.confidence, c.timestamp,
                         debug_subject=c.subject, debug_predicate=c.predicate)
                n_inserts += 1
    absorb_ms = (time.perf_counter() - t0) / (2 * len(PAIRS)) * 1000

    ckpt = os.path.join(out_dir, f"phase8_{name}.npz")
    save_checkpoint(mem, ckpt)
    state_bytes = os.path.getsize(ckpt)

    return dict(system=name,
                exact_duplicate_consolidation=round(exact_rate, 4),
                paraphrase_consolidation=round(para_rate, 4),
                contradiction_consolidation=round(contra_rate, 4),
                unrelated_false_consolidation=round(unrel_rate, 4),
                slots_used=len(mem.slots), n_updates=n_updates, n_inserts=n_inserts,
                state_bytes=state_bytes, avg_absorb_latency_ms=round(absorb_ms, 5))


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "_phase8_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    atomic = AtomicEncoder()
    semantic = SemanticEmbeddingEncoder()

    systems = [
        ("A_atomic_plus_blackhole", atomic, True),
        ("B_semantic_plus_blackhole", semantic, True),
        ("C_semantic_NO_update", semantic, False),
    ]

    results = []
    for name, enc, use_update in systems:
        r = run_system(name, enc, use_update, out_dir)
        results.append(r)
        print(f"{name}:")
        for k, v in r.items():
            if k != "system":
                print(f"    {k}: {v}")
        print()

    print("=" * 100)
    print("CRITICAL ABLATION: does System B (semantic + update) beat System C (semantic + no-update)?")
    print("=" * 100)
    b, c = results[1], results[2]
    print(f"  paraphrase_consolidation:  B={b['paraphrase_consolidation']}  C={c['paraphrase_consolidation']}")
    print(f"  contradiction_consolidation: B={b['contradiction_consolidation']}  C={c['contradiction_consolidation']}")
    print(f"  slots_used: B={b['slots_used']}  C={c['slots_used']}")
    print(f"  state_bytes: B={b['state_bytes']}  C={c['state_bytes']}")

    return results


if __name__ == "__main__":
    main()
