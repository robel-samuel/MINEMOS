"""Self-contained fixtures for the optional retrieval backend tests."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate

WIDTH = 384
QUERY = "q"
_Q = np.random.default_rng(7).standard_normal(WIDTH).astype(np.float32)
Q_VECTOR = (_Q / np.linalg.norm(_Q)).astype(np.float32)


class FixedEncoder:
    """Deterministic 384-d encoder; unknown text hashes to a unit vector."""

    def __init__(self, mapping: Optional[Dict[str, np.ndarray]] = None,
                 width: int = WIDTH):
        self.mapping = dict(mapping or {QUERY: Q_VECTOR})
        self.width = int(width)

    def encode(self, text: str) -> np.ndarray:
        if text in self.mapping:
            return np.asarray(self.mapping[text], dtype=np.float32).copy()
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(self.width)
        v = v.astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v


class RuleVerifier:
    """First matching (slot substring, candidate substring) rule wins."""

    def __init__(self, rules: Sequence[tuple] = (), default: str = "NEUTRAL"):
        self.rules = list(rules)
        self.default = default

    def _decide(self, slot_text: str, cand_text: str) -> str:
        for slot_marker, cand_marker, decision in self.rules:
            if slot_marker in slot_text and cand_marker in cand_text:
                return decision
        return self.default

    def verify_batch(self, pairs, **kwargs):
        out = []
        for slot_text, cand_text in pairs:
            decision = self._decide(slot_text, cand_text)
            contra = decision == "CONTRADICTION"
            same = decision == "SAME"
            out.append({
                "decision": decision,
                "confidence": 0.99,
                "p_entail": 0.99 if same else 0.01,
                "p_contra": 0.99 if contra else 0.01,
                "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01,
                "fwd": {"p_contra": 0.99 if contra else 0.01,
                        "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01},
                "rev": {"p_contra": 0.99 if contra else 0.01,
                        "p_neutral": 0.98 if decision == "NEUTRAL" else 0.01},
            })
        return out


class StubNLI:
    pass


def make_memory(capacity: int = 100, verifier=None, backend: str = "canonical",
                encoder=None, **kwargs) -> BlackHoleMemory:
    return BlackHoleMemory(
        capacity=capacity,
        dim=4096,
        dense_encoder=encoder or FixedEncoder(),
        nli_gate=StubNLI(),
        verifier=verifier or RuleVerifier(),
        retrieval_backend=backend,
        **kwargs,
    )


def add_facts(mem: BlackHoleMemory, facts) -> None:
    for subject, predicate, obj, timestamp in facts:
        mem.absorb(Candidate(subject, predicate, obj, timestamp=timestamp))


def unit_vector(seed: int, width: int = WIDTH) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(width).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def rows_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if set(ra) != set(rb):
            return False
        for key in ra:
            va, vb = ra[key], rb[key]
            if isinstance(va, float) and isinstance(vb, float):
                if math.isnan(va) and math.isnan(vb):
                    continue
            if va != vb:
                return False
    return True


class TimedRLock:
    """RLock proxy that records top-level acquisition wait time."""

    def __init__(self, lock):
        self._lock = lock
        self._guard = threading.Lock()
        self._held = threading.local()
        self.samples = []

    def acquire(self, blocking: bool = True, timeout: float = -1):
        t0 = time.perf_counter()
        if timeout is None or timeout < 0:
            ok = self._lock.acquire(blocking)
        else:
            ok = self._lock.acquire(blocking, timeout)
        if ok:
            depth = getattr(self._held, "depth", 0)
            if depth == 0:
                with self._guard:
                    self.samples.append(time.perf_counter() - t0)
            self._held.depth = depth + 1
        return ok

    def release(self):
        depth = getattr(self._held, "depth", 1)
        self._held.depth = max(0, depth - 1)
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def percentile(values, p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]
