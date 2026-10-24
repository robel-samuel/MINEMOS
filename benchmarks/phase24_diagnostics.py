"""
benchmarks/phase24_diagnostics.py

Phase 24 Failure Attribution & Lifecycle Diagnostic Instrumentation.

Classifies all evaluated probes into precise causal categories:
  A. Retrieval failure (Target was in memory, but not in candidate beam).
  B. Semantic verification rejection (Target was in beam, but NLI rejected valid paraphrase).
  C. Consolidation or representation drift failure (NLI accepted, but consolidation failed or drifted).
  D. Retention or eviction failure (Target memory was evicted before probe arrived).
  E. False consolidation / False merge (Contradiction falsely merged into target slot).
  SUCCESS. Verified correct decision and memory preservation.
  UNKNOWN. Unclassified or unregistered case.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import numpy as np

from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity


@dataclass
class Phase24TargetRecord:
    target_id: str
    concept_id: str
    target_slot_id: Optional[int] = None
    created_at_step: int = -1
    canonical_text: str = ""
    initial_value: Optional[np.ndarray] = None
    initial_dense: Optional[np.ndarray] = None
    update_history: list[dict] = field(default_factory=list)
    evicted: bool = False
    evicted_at_step: Optional[int] = None
    probes_evaluated: list[str] = field(default_factory=list)


@dataclass
class Phase24AttributionRecord:
    target_id: str
    probe_id: str
    target_slot_id: Optional[int]
    slot_exists_at_probe: bool
    slot_evicted: bool
    correct_slot_retrieved: bool
    retrieval_rank: Optional[int]
    nli_called: bool
    nli_score: Optional[float]
    nli_accepted: bool
    consolidation_attempted: bool
    consolidation_succeeded: bool
    slot_update_count: int
    cos_to_initial: Optional[float]
    cos_to_latest: Optional[float]
    final_correct: bool
    failure_category: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class Phase24DiagnosticTracker:
    """Tracks memory lifecycles and classifies probe results."""

    def __init__(self):
        self.targets: dict[str, Phase24TargetRecord] = {}
        self.slot_to_target: dict[int, str] = {}
        self.attributions: list[Phase24AttributionRecord] = []

    def register_target(
        self,
        target_id: str,
        concept_id: str,
        slot: MemorySlot,
        step: int,
        canonical_text: str,
        dense_emb: Optional[np.ndarray] = None
    ):
        slot_id = id(slot)
        val = slot.value.copy() if hasattr(slot, "value") and slot.value is not None else None
        rec = Phase24TargetRecord(
            target_id=target_id,
            concept_id=concept_id,
            target_slot_id=slot_id,
            created_at_step=step,
            canonical_text=canonical_text,
            initial_value=val,
            initial_dense=dense_emb.copy() if dense_emb is not None else None,
            update_history=[{
                "step": step,
                "text": canonical_text,
                "update_count": slot.update_count,
                "cosine_to_initial": 1.0,
                "cosine_to_latest": 1.0
            }]
        )
        self.targets[target_id] = rec
        self.slot_to_target[slot_id] = target_id

    def record_update(
        self,
        slot: MemorySlot,
        step: int,
        update_text: str,
        cand_lex: Optional[np.ndarray] = None
    ):
        slot_id = id(slot)
        target_id = self.slot_to_target.get(slot_id)
        if not target_id or target_id not in self.targets:
            return
        rec = self.targets[target_id]
        cos_initial = 1.0
        cos_latest = 1.0
        if rec.initial_value is not None and slot.value is not None:
            cos_initial = float(cosine_similarity(rec.initial_value, slot.value))
        if cand_lex is not None and slot.value is not None:
            cos_latest = float(cosine_similarity(cand_lex, slot.value))

        rec.update_history.append({
            "step": step,
            "text": update_text,
            "update_count": slot.update_count,
            "cosine_to_initial": round(cos_initial, 4),
            "cosine_to_latest": round(cos_latest, 4)
        })

    def record_eviction(self, evicted_slot: MemorySlot, step: int):
        slot_id = id(evicted_slot)
        target_id = self.slot_to_target.get(slot_id)
        if target_id and target_id in self.targets:
            rec = self.targets[target_id]
            rec.evicted = True
            rec.evicted_at_step = step

    def evaluate_probe_step(
        self,
        probe_id: str,
        target_id: str,
        category: str,
        memory: MemoryState,
        step_result: dict,
        cand_lex: Optional[np.ndarray] = None
    ) -> Phase24AttributionRecord:
        target_rec = self.targets.get(target_id)
        if target_rec is None:
            rec = Phase24AttributionRecord(
                target_id=target_id,
                probe_id=probe_id,
                target_slot_id=None,
                slot_exists_at_probe=False,
                slot_evicted=False,
                correct_slot_retrieved=False,
                retrieval_rank=None,
                nli_called=False,
                nli_score=None,
                nli_accepted=False,
                consolidation_attempted=False,
                consolidation_succeeded=False,
                slot_update_count=0,
                cos_to_initial=None,
                cos_to_latest=None,
                final_correct=False,
                failure_category="UNKNOWN",
                details={"reason": "target_not_registered"}
            )
            self.attributions.append(rec)
            return rec

        target_slot_id = target_rec.target_slot_id
        alive_slots_by_id = {id(s): (idx, s) for idx, s in enumerate(memory.slots)}
        slot_exists = target_slot_id in alive_slots_by_id
        target_slot = alive_slots_by_id[target_slot_id][1] if slot_exists else None
        target_curr_idx = alive_slots_by_id[target_slot_id][0] if slot_exists else None

        fused_indices = step_result.get("top_k_indices", [])
        correct_slot_retrieved = False
        retrieval_rank = None
        if slot_exists and target_curr_idx is not None:
            if target_curr_idx in fused_indices:
                correct_slot_retrieved = True
                retrieval_rank = fused_indices.index(target_curr_idx)

        action = step_result.get("action", "")
        gate = step_result.get("gate", "")
        sem_pred = step_result.get("sem_pred") or {}
        nli_called = (step_result.get("candidates_examined", 0) > 0)
        nli_score = sem_pred.get("p_entail") if sem_pred else None
        nli_accepted = (gate == "nli_accept")
        is_merge = (action == "update")
        consolidation_attempted = is_merge
        consolidation_succeeded = is_merge and (step_result.get("slot_index") == target_curr_idx)

        is_para = category in ("paraphrase", "delayed_paraphrase")
        is_contra = category in ("contradiction", "drift_contra")

        if is_para:
            final_correct = is_merge and (step_result.get("slot_index") == target_curr_idx)
        elif is_contra:
            target_merged = is_merge and (step_result.get("slot_index") == target_curr_idx)
            final_correct = not target_merged
        else:
            final_correct = True

        # Failure Category Attribution
        if final_correct:
            category_code = "SUCCESS"
        elif target_rec.evicted or not slot_exists:
            category_code = "D"  # Retention / eviction failure
        elif is_para:
            if not correct_slot_retrieved:
                category_code = "A"  # Retrieval failure
            elif not nli_accepted:
                category_code = "B"  # Semantic verification rejection
            elif not consolidation_succeeded:
                category_code = "C"  # Consolidation or drift failure
            else:
                category_code = "E"  # Mis-adjudicated / False consolidation
        elif is_contra:
            if correct_slot_retrieved and is_merge and (step_result.get("slot_index") == target_curr_idx):
                category_code = "E"  # False consolidation (contradiction erroneously merged!)
            else:
                category_code = "UNKNOWN"
        else:
            category_code = "UNKNOWN"

        cos_initial = None
        cos_latest = None
        if target_slot is not None and target_rec.initial_value is not None:
            cos_initial = round(float(cosine_similarity(target_rec.initial_value, target_slot.value)), 4)
        if target_slot is not None and cand_lex is not None:
            cos_latest = round(float(cosine_similarity(cand_lex, target_slot.value)), 4)

        rec = Phase24AttributionRecord(
            target_id=target_id,
            probe_id=probe_id,
            target_slot_id=target_slot_id,
            slot_exists_at_probe=slot_exists,
            slot_evicted=target_rec.evicted,
            correct_slot_retrieved=correct_slot_retrieved,
            retrieval_rank=retrieval_rank,
            nli_called=nli_called,
            nli_score=nli_score,
            nli_accepted=nli_accepted,
            consolidation_attempted=consolidation_attempted,
            consolidation_succeeded=consolidation_succeeded,
            slot_update_count=target_slot.update_count if target_slot else 0,
            cos_to_initial=cos_initial,
            cos_to_latest=cos_latest,
            final_correct=final_correct,
            failure_category=category_code,
            details={
                "category": category,
                "action": action,
                "gate": gate,
                "candidates_examined": step_result.get("candidates_examined", 0)
            }
        )
        self.attributions.append(rec)
        target_rec.probes_evaluated.append(probe_id)
        return rec

    def compute_summary_distribution(self) -> dict:
        total = len(self.attributions)
        if total == 0:
            return {"total": 0, "counts": {}, "percentages": {}}
        counts = {}
        for r in self.attributions:
            counts[r.failure_category] = counts.get(r.failure_category, 0) + 1
        pcts = {cat: round(cnt / total * 100.0, 2) for cat, cnt in counts.items()}
        return {"total": total, "counts": counts, "percentages": pcts}
