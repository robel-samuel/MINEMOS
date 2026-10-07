"""
benchmarks/phase7_memory_level.py

Runs the exact Phase 6 consolidation dataset (A/B/C pairs) through the
UNCHANGED Black Hole memory mechanism, with only the encoder swapped.
MATCH_THRESHOLD frozen at 0.75 (the Phase 6 default). Compares Encoder A
(control) vs Encoder B (field-aware lexical) on:
  - exact duplicate / paraphrase / false consolidation rates
  - full-mechanism vs no-update ablation
  - slots used, state bytes, latency
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.memory_state import MemoryState
from phase2.memory_update import absorb
from phase2.checkpoint import save_checkpoint
from phase2.encoders import AtomicEncoder, FieldAwareLexicalEncoder
from benchmarks.threshold_sweep import A_PAIRS, B_PAIRS, C_PAIRS, ALL_PAIRS

D = 4096
FROZEN_THRESHOLD = 0.75  # Phase 6 default, frozen per Phase 7 spec


def merged_status(pair, encoder, threshold: float) -> bool:
    m = MemoryState(capacity=10, dim=D)
    absorb(m, pair.c1, threshold=threshold, encoder=encoder)
    r2 = absorb(m, pair.c2, threshold=threshold, encoder=encoder)
    return r2["action"] == "update"


def run_encoder(name: str, encoder, out_dir: str) -> dict:
    a_merged = sum(1 for p in A_PAIRS if merged_status(p, encoder, FROZEN_THRESHOLD))
    b_merged = sum(1 for p in B_PAIRS if merged_status(p, encoder, FROZEN_THRESHOLD))
    c_merged = sum(1 for p in C_PAIRS if merged_status(p, encoder, FROZEN_THRESHOLD))

    # full mechanism, combined stream
    mem_full = MemoryState(capacity=200, dim=D)
    t0 = time.perf_counter()
    for pair in ALL_PAIRS:
        absorb(mem_full, pair.c1, threshold=FROZEN_THRESHOLD, encoder=encoder)
        absorb(mem_full, pair.c2, threshold=FROZEN_THRESHOLD, encoder=encoder)
    absorb_ms = (time.perf_counter() - t0) / (2 * len(ALL_PAIRS)) * 1000

    # no-update baseline, same stream
    mem_none = MemoryState(capacity=200, dim=D)
    for pair in ALL_PAIRS:
        for c in (pair.c1, pair.c2):
            from phase2.structured_candidate import Candidate
            x_t = encoder.encode_full(c, dim=D)
            mem_none.add(x_t, x_t.copy(), c.confidence, c.timestamp,
                         debug_subject=c.subject, debug_predicate=c.predicate)

    ckpt_path = os.path.join(out_dir, f"phase7_{name}.npz")
    save_checkpoint(mem_full, ckpt_path)
    state_bytes = os.path.getsize(ckpt_path)

    return dict(
        encoder=name,
        exact_duplicate_consolidation_rate=round(a_merged / len(A_PAIRS), 4),
        paraphrase_consolidation_rate=round(b_merged / len(B_PAIRS), 4),
        false_consolidation_rate=round(c_merged / len(C_PAIRS), 4),
        full_slots_used=len(mem_full.slots),
        no_update_slots_used=len(mem_none.slots),
        slot_reduction=len(mem_none.slots) - len(mem_full.slots),
        state_bytes=state_bytes,
        avg_absorb_latency_ms=round(absorb_ms, 5),
    )


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "_phase7_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reusing Phase 6 dataset: {len(A_PAIRS)} exact-dup, {len(B_PAIRS)} paraphrase, "
          f"{len(C_PAIRS)} hard-negative pairs. MATCH_THRESHOLD frozen at {FROZEN_THRESHOLD}.")
    print()

    results = []
    for name, enc in [("A_atomic_control", AtomicEncoder()),
                       ("B_field_aware_lexical", FieldAwareLexicalEncoder())]:
        r = run_encoder(name, enc, out_dir)
        results.append(r)
        print(f"{name}:")
        for k, v in r.items():
            if k != "encoder":
                print(f"    {k}: {v}")
        print()

    return results


if __name__ == "__main__":
    main()
