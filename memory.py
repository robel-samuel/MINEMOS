"""
Black Hole Memory — Phase 1 core (v2: grouped storage).

A minimal, inspectable implementation of the Update(M(t), X(t)) function
described in the design doc (Section 11), covering:
  - absorption of new facts            (Section 9)
  - surprise-driven reinforcement       (Section 13)
  - contradiction / temporal handling   (Sections 17-18)
  - confidence tracking                 (Section 20)
  - consolidation via grouped storage   (Section 14)
  - checkpoint / restore                (Section 21)

v2 change from the first prototype: facts are grouped under one record per
subject (attribute -> list of values, oldest to newest) instead of one
standalone record per (subject, predicate, object) triple. This removes
per-fact UUIDs and duplicated subject/predicate strings, which were
dominating memory size in the v1 benchmark (compression ratio was ~0.1x —
worse than storing raw text). Lookup is O(1) by subject via an index,
replacing the earlier linear scan.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AttributeValue:
    """One value an attribute has held, with its validity window."""
    o: str                     # object / value
    c: float = 0.5              # confidence
    sc: int = 1                 # source_count (times reinforced)
    t: float = 0.0              # last-updated timestamp
    vf: float = 0.0             # valid_from
    vu: Optional[float] = None  # valid_until (None = still active)

    @property
    def active(self) -> bool:
        return self.vu is None

    # Kept as full names for readability where used outside this file.
    @property
    def object(self) -> str: return self.o
    @property
    def confidence(self) -> float: return self.c
    @property
    def source_count(self) -> int: return self.sc
    @property
    def timestamp(self) -> float: return self.t
    @property
    def valid_from(self) -> float: return self.vf
    @property
    def valid_until(self) -> Optional[float]: return self.vu

    def to_dict(self) -> dict:
        """
        Sparse serialization. Two fixes vs. the first version:
          1. Timestamps stored as whole seconds (int), not microsecond
             floats -- fact-level memory doesn't need sub-second precision,
             and each float was costing ~18 chars vs ~10 for an int.
          2. Fields are only written when they carry information: "t" is
             omitted when it equals "vf" (true until a fact is reinforced
             at least once), and "vu" is omitted when None (true for every
             still-active fact, i.e. most of them). This removes redundant
             '"vu":null' pairs and duplicate timestamp pairs that were
             ~38% of a serialized fact's bytes.
        """
        d = {"o": self.o, "c": round(self.c, 2), "sc": self.sc, "vf": int(self.vf)}
        t_int = int(self.t)
        if t_int != int(self.vf):
            d["t"] = t_int
        if self.vu is not None:
            d["vu"] = int(self.vu)
        return d

    @staticmethod
    def from_dict(d: dict) -> "AttributeValue":
        vf = d["vf"]
        return AttributeValue(
            o=d["o"], c=d["c"], sc=d["sc"],
            vf=vf, t=d.get("t", vf), vu=d.get("vu"),
        )


@dataclass
class SubjectRecord:
    """
    All known attributes for one subject, e.g. person_3 ->
      {uses_id: [AttributeValue(...)], lives_in_id: [AttributeValue(...)]}
    Each attribute's value list is history-ordered; only the last entry
    can be active (vu is None).

    Keyed by predicate ID (int), not predicate string -- see predicate
    interning in MemoryStore. The predicate string itself is stored once
    in a shared table rather than once per subject that uses it.
    """
    subject: str
    attrs: dict[int, list[AttributeValue]] = field(default_factory=dict)

    def active(self, predicate_id: int) -> Optional[AttributeValue]:
        values = self.attrs.get(predicate_id)
        if not values:
            return None
        return values[-1] if values[-1].active else None

    def to_dict(self) -> dict:
        return {
            "s": self.subject,
            # JSON object keys must be strings; predicate_id stays a small
            # int-as-string ("0", "1", ...) which is still far shorter than
            # repeating a predicate name like "lives_in" at every subject.
            "a": {str(pid): [v.to_dict() for v in vs] for pid, vs in self.attrs.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "SubjectRecord":
        rec = SubjectRecord(subject=d["s"])
        rec.attrs = {
            int(pid): [AttributeValue.from_dict(v) for v in vs]
            for pid, vs in d["a"].items()
        }
        return rec


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Holds SubjectRecords and implements Update(M(t), X(t)).
    O(1) lookup by subject via self.subjects (a dict), so absorb/recall no
    longer scan the whole memory state.
    """

    BASE_CONFIDENCE = 0.5
    MAX_CONFIDENCE = 0.99
    REINFORCE_STEP = 0.15

    def __init__(self):
        self.subjects: dict[str, SubjectRecord] = {}
        # Predicate interning: predicate strings ("uses", "lives_in", ...)
        # were previously repeated as a dict key once per subject that had
        # that attribute. At 10K facts that's thousands of repeated copies
        # of a handful of distinct strings. Store each predicate string
        # once in a shared table and reference it by small int ID instead.
        self._pred_to_id: dict[str, int] = {}
        self._id_to_pred: list[str] = []

    def _intern(self, predicate: str) -> int:
        pid = self._pred_to_id.get(predicate)
        if pid is None:
            pid = len(self._id_to_pred)
            self._pred_to_id[predicate] = pid
            self._id_to_pred.append(predicate)
        return pid

    def _predicate_name(self, pid: int) -> str:
        return self._id_to_pred[pid]

    # -- absorption / update --------------------------------------------

    def absorb(self, subject: str, predicate: str, object_: str,
               base_confidence: Optional[float] = None,
               dedupe: bool = True,
               at_time: Optional[float] = None) -> AttributeValue:
        """
        dedupe=True (default, unchanged from before): repeated statements
        about the same (subject, predicate) reinforce a single record --
        this is the consolidation mechanism being tested in the realistic-
        conversation benchmark.
        dedupe=False: always append a new record, never reinforce. Used
        only to build an "unconsolidated" baseline store for comparison --
        i.e. what memory would cost if every absorbed statement were kept
        as its own record, the way the very first prototype worked.
        at_time: override for "now", so benchmarks can simulate facts
        arriving on specific days instead of wall-clock time.
        """
        now = at_time if at_time is not None else time.time()
        record = self.subjects.get(subject)
        if record is None:
            record = SubjectRecord(subject=subject)
            self.subjects[subject] = record

        pid = self._intern(predicate)
        existing_active = record.active(pid) if dedupe else None

        # Case 1: no prior value for this attribute -> insert new
        if existing_active is None:
            value = AttributeValue(
                o=object_,
                c=base_confidence or self.BASE_CONFIDENCE,
                t=now, vf=now,
            )
            record.attrs.setdefault(pid, []).append(value)
            return value

        # Case 2: same object -> reinforcement (surprise decreases)
        if existing_active.o == object_:
            decay = 1.0 / (1 + existing_active.sc)
            existing_active.c = min(
                self.MAX_CONFIDENCE,
                existing_active.c + self.REINFORCE_STEP * decay,
            )
            existing_active.sc += 1
            existing_active.t = now
            return existing_active

        # Case 3: different object -> contradiction / temporal update.
        # Close the old value's validity window; append the new one.
        existing_active.vu = now
        new_value = AttributeValue(
            o=object_,
            c=base_confidence or self.BASE_CONFIDENCE,
            t=now, vf=now,
        )
        record.attrs[pid].append(new_value)
        return new_value

    # -- recall -----------------------------------------------------------

    def recall(self, subject: str, predicate: Optional[str] = None,
               as_of: Optional[float] = None, include_history: bool = False):
        record = self.subjects.get(subject)
        if record is None:
            return []

        if predicate is not None:
            pid = self._pred_to_id.get(predicate)
            if pid is None:
                return []  # this predicate was never absorbed by anyone
            pids = [pid]
        else:
            pids = list(record.attrs.keys())

        results = []
        for pid in pids:
            values = record.attrs.get(pid, [])
            for v in values:
                if as_of is not None:
                    if v.vf > as_of:
                        continue
                    if v.vu is not None and v.vu <= as_of:
                        continue
                else:
                    if not include_history and not v.active:
                        continue
                results.append(_ResultView(subject, self._predicate_name(pid), v))
        results.sort(key=lambda r: r.valid_from)
        return results

    # -- consolidation ------------------------------------------------------

    def consolidate(self, min_source_count: int = 3):
        """
        Facts reinforced enough times get a stable-confidence floor.
        Grouped storage already provides the main compression win (Section
        14); this pass adds the confidence-stability behavior from v1.
        """
        promoted = []
        for subject, record in self.subjects.items():
            for pid, values in record.attrs.items():
                v = values[-1]
                if v.active and v.sc >= min_source_count:
                    v.c = max(v.c, 0.9)
                    promoted.append((subject, self._predicate_name(pid)))
        return promoted

    # -- persistence --------------------------------------------------------

    def checkpoint(self, path: str):
        data = {
            "p": self._id_to_pred,
            "d": [r.to_dict() for r in self.subjects.values()],
        }
        with open(path, "w") as fh:
            json.dump(data, fh, separators=(",", ":"))  # no indent: compact

    @staticmethod
    def restore(path: str) -> "MemoryStore":
        store = MemoryStore()
        with open(path) as fh:
            payload = json.load(fh)
        store._id_to_pred = payload["p"]
        store._pred_to_id = {name: i for i, name in enumerate(store._id_to_pred)}
        for d in payload["d"]:
            rec = SubjectRecord.from_dict(d)
            store.subjects[rec.subject] = rec
        return store

    # -- stats ----------------------------------------------------------

    def stats(self) -> dict:
        active_count = 0
        historical_count = 0
        confidences = []
        for record in self.subjects.values():
            for values in record.attrs.values():
                for v in values:
                    if v.active:
                        active_count += 1
                        confidences.append(v.c)
                    else:
                        historical_count += 1
        return {
            "subjects": len(self.subjects),
            "active_facts": active_count,
            "historical_facts": historical_count,
            "avg_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        }


class _ResultView:
    """Read-only convenience wrapper so recall() results keep the same
    .object / .confidence / .subject / .predicate attribute names v1 used."""

    def __init__(self, subject: str, predicate: str, value: AttributeValue):
        self.subject = subject
        self.predicate = predicate
        self._value = value

    @property
    def object(self) -> str: return self._value.o
    @property
    def confidence(self) -> float: return self._value.c
    @property
    def source_count(self) -> int: return self._value.sc
    @property
    def timestamp(self) -> float: return self._value.t
    @property
    def valid_from(self) -> float: return self._value.vf
    @property
    def valid_until(self) -> Optional[float]: return self._value.vu
    @property
    def active(self) -> bool: return self._value.active

    def __repr__(self):
        return f"Fact({self.subject}.{self.predicate}={self.object!r}, conf={self.confidence:.2f})"
