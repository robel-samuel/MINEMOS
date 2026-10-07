"""
Phase 2 — memory update mechanism.

Implements exactly:

  s_i = sim(x_t, k_i)                         (cosine similarity)
  j = argmax_i(s_i);  max_similarity = s_j
  N_t = 1 - max_similarity   (N_t = 1 if memory empty)
  alpha_t = N_t                                (documented simple mapping)
  v_j(t+1) = (1 - alpha_t) v_j(t) + alpha_t x_t

ADDRESSING / MERGE-VS-NEW-SLOT RULE (the "central part of the experiment"
the task calls out):

x_t is the FULL candidate encoding (subject+predicate+object tokens,
see structured_candidate.py). Because it's a bag-of-tokens vector, two
candidates sharing the same (subject, predicate) but a DIFFERENT object
will have PARTIAL token overlap -- similarity somewhere between "totally
different fact" and "identical fact", not a clean 0 or 1. A single fixed
threshold on this similarity is therefore what decides merge vs.
new-slot:

    MATCH_THRESHOLD = 0.75   (fixed BEFORE running any benchmark below --
                               not tuned after seeing results)

  s_j >= MATCH_THRESHOLD  -> same slot: blend value via the update
                             equation (this is the "same fact,
                             reinforce/paraphrase" path)
  s_j <  MATCH_THRESHOLD  -> new slot (if capacity allows), or FIFO
                             eviction of the oldest slot if at capacity

This is a real, known limitation, stated up front rather than discovered
later: a single content-similarity threshold cannot simultaneously (a)
merge true paraphrases of the same fact and (b) keep genuinely
contradictory facts (same subject+predicate, different object) apart,
because both cases differ from the "identical fact" case by the same
kind of partial token overlap. Which behavior the threshold produces for
a given contradiction depends on how many tokens the subject+predicate
share relative to the differing object token -- see memory_phase2_report.md,
Test E, for the measured outcome, not an assumed one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phase2.structured_candidate import Candidate, encode_full
from phase2.memory_state import MemoryState, MemorySlot, CapacityExceeded

MATCH_THRESHOLD = 0.75  # fixed before any benchmark run; see module docstring


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def address(memory: MemoryState, x_t: np.ndarray) -> tuple[Optional[int], float]:
    """Returns (best_slot_index_or_None, max_similarity). Empty memory ->
    (None, 0.0), which the caller must map to N_t = 1 explicitly."""
    if not memory.slots:
        return None, 0.0
    sims = [cosine_similarity(x_t, slot.key) for slot in memory.slots]
    best_idx = int(np.argmax(sims))
    return best_idx, sims[best_idx]


def novelty(max_similarity: float, memory_empty: bool) -> float:
    if memory_empty:
        return 1.0
    return 1.0 - max_similarity


def absorb(memory: MemoryState, candidate: Candidate,
           eviction: str = "fifo", threshold: float = None,
           encoder=None) -> dict:
    """
    Full absorb pipeline for one candidate. Returns a dict describing what
    happened (useful for tests/diagnostics): action in {"update","insert",
    "evict_insert","reject"}, slot_index, similarity, novelty, alpha.

    threshold: override for MATCH_THRESHOLD (Phase 6). Defaults to the
    module-level MATCH_THRESHOLD so every existing call site/test is
    unaffected.
    encoder: override for how x_t = E(C_t) is computed (Phase 7). Must be
    an object with .encode_full(candidate, dim) -> np.ndarray. Defaults to
    the Phase 2B atomic encoder (structured_candidate.encode_full), so
    every existing call site/test is unaffected. This is the ONLY change
    Phase 7 needed at the memory-mechanism layer -- similarity, novelty,
    the update equation, addressing, and thresholding are all untouched.
    """
    if threshold is None:
        threshold = MATCH_THRESHOLD
    if encoder is None:
        x_t = encode_full(candidate, dim=memory.dim)
    else:
        x_t = encoder.encode_full(candidate, dim=memory.dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t  # documented simple mapping: alpha_t = N_t

    if best_idx is not None and max_sim >= threshold:
        slot = memory.slots[best_idx]
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        return dict(action="update", slot_index=best_idx, similarity=max_sim,
                    novelty=n_t, alpha=alpha_t)

    # no sufficiently similar slot -- needs a new one
    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot_index=len(memory.slots) - 1,
                    similarity=max_sim, novelty=n_t, alpha=alpha_t)

    if eviction == "fifo":
        memory.evict_oldest()
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="evict_insert", slot_index=len(memory.slots) - 1,
                    similarity=max_sim, novelty=n_t, alpha=alpha_t)

    return dict(action="reject", slot_index=None, similarity=max_sim,
                novelty=n_t, alpha=alpha_t)
