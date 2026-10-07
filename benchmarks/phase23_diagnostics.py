"""
benchmarks/phase23_diagnostics.py

Phase 23 Failure Attribution and Diagnostic Instrumentation for Black Hole Memory.

Diagnoses the exact point of failure when a memory candidate is processed:
  A = Correct target was not retrieved
  B = Correct target was retrieved but NLI rejected it
  C = NLI accepted the target but consolidation/update failed
  D = Correct memory was consolidated but target slot was later evicted
  E = Target survived but final probe retrieval failed
  F = Probe retrieved correct slot but final evaluation/consolidation was incorrect
  SUCCESS = Target correctly consolidated and verified by probe
  UNKNOWN = Insufficient evidence to classify
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import numpy as np

from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import cosine_similarity


@dataclass
class TargetTrackingRecord:
    target_id: str
    concept_id: str
    target_slot_id: Optional[int] = None
    initial_slot_index: Optional[int] = None
    created_at_step: int = -1
    initial_value: Optional[np.ndarray] = None
    initial_dense: Optional[np.ndarray] = None
    canonical_text: str = ""
    # Drift tracking across sequential updates
    update_history: list[dict] = field(default_factory=list)
    # Eviction tracking
    evicted: bool = False
    evicted_at_step: Optional[int] = None
    evicted_before_probe: bool = False
    # Probes associated with this target
    probes_evaluated: list[str] = field(default_factory=list)


@dataclass
class FailureAttributionRecord:
    target_id: str
    probe_id: str
    target_slot_id: Optional[int]
    slot_exists_at_probe: bool
    slot_evicted: bool
    retrieved_slot_ids: list[int]
    correct_slot_retrieved: bool
    retrieval_rank: Optional[int]
    nli_called: bool
    nli_score: Optional[float]
    nli_accepted: bool
    consolidation_attempted: bool
    consolidation_succeeded: bool
    slot_update_count: int
    slot_embedding_before: Optional[list[float]] = None
    slot_embedding_after: Optional[list[float]] = None
    probe_retrieval_rank: Optional[int] = None
    final_correct: bool = False
    failure_category: str = "UNKNOWN"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # convert numpy / float objects
        if d["slot_embedding_before"] is not None:
            d["slot_embedding_before"] = [round(float(x), 4) for x in d["slot_embedding_before"][:5]]
        if d["slot_embedding_after"] is not None:
            d["slot_embedding_after"] = [round(float(x), 4) for x in d["slot_embedding_after"][:5]]
        return d


class Phase23DiagnosticTracker:
    """
    Tracks target memory lifecycles and classifies probe failures into
    categories A, B, C, D, E, F, or SUCCESS.
    """
    def __init__(self):
        self.targets: dict[str, TargetTrackingRecord] = {}
        self.slot_to_target: dict[int, str] = {}
        self.attributions: list[FailureAttributionRecord] = []
        self.drift_records: list[dict] = []

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
        dense = dense_emb.copy() if dense_emb is not None else None
        rec = TargetTrackingRecord(
            target_id=target_id,
            concept_id=concept_id,
            target_slot_id=slot_id,
            initial_slot_index=None,
            created_at_step=step,
            initial_value=val,
            initial_dense=dense,
            canonical_text=canonical_text,
            update_history=[{
                "step": step,
                "text": canonical_text,
                "update_count": slot.update_count,
                "cosine_to_initial": 1.0
            }]
        )
        self.targets[target_id] = rec
        self.slot_to_target[slot_id] = target_id

    def record_update(
        self,
        slot: MemorySlot,
        step: int,
        update_text: str,
        dense_emb: Optional[np.ndarray] = None
    ):
        slot_id = id(slot)
        target_id = self.slot_to_target.get(slot_id)
        if not target_id or target_id not in self.targets:
            return
        rec = self.targets[target_id]
        cos_initial = 1.0
        if rec.initial_value is not None and slot.value is not None:
            cos_initial = float(cosine_similarity(rec.initial_value, slot.value))
        rec.update_history.append({
            "step": step,
            "text": update_text,
            "update_count": slot.update_count,
            "cosine_to_initial": round(cos_initial, 4)
        })

    def record_eviction(self, evicted_slot: MemorySlot, step: int):
        slot_id = id(evicted_slot)
        target_id = self.slot_to_target.get(slot_id)
        if target_id and target_id in self.targets:
            rec = self.targets[target_id]
            rec.evicted = True
            rec.evicted_at_step = step
            rec.evicted_before_probe = True

    def evaluate_probe_step(
        self,
        probe_id: str,
        target_id: str,
        category: str,  # 'paraphrase', 'contradiction', etc.
        memory: MemoryState,
        step_result: dict,
        cand_dense: Optional[np.ndarray] = None,
        cand_lex: Optional[np.ndarray] = None
    ) -> FailureAttributionRecord:
        """
        Classifies the outcome of a probe into failure category:
          A: Correct target not retrieved
          B: Correct target retrieved but NLI rejected
          C: NLI accepted target but consolidation failed
          D: Target consolidated earlier but slot was evicted
          E: Target survived but probe retrieval failed
          F: Probe retrieved slot but evaluation/consolidation incorrect
          SUCCESS: Successful correct decision and action
        """
        target_rec = self.targets.get(target_id)
        if target_rec is None:
            # Target was never registered
            rec = FailureAttributionRecord(
                target_id=target_id,
                probe_id=probe_id,
                target_slot_id=None,
                slot_exists_at_probe=False,
                slot_evicted=False,
                retrieved_slot_ids=[],
                correct_slot_retrieved=False,
                retrieval_rank=None,
                nli_called=False,
                nli_score=None,
                nli_accepted=False,
                consolidation_attempted=False,
                consolidation_succeeded=False,
                slot_update_count=0,
                final_correct=False,
                failure_category="UNKNOWN",
                details={"reason": "target_not_registered"}
            )
            self.attributions.append(rec)
            return rec

        target_slot_id = target_rec.target_slot_id

        # 1. Check if target slot currently exists in memory
        alive_slots_by_id = {id(s): (idx, s) for idx, s in enumerate(memory.slots)}
        slot_exists = target_slot_id in alive_slots_by_id
        target_slot = alive_slots_by_id[target_slot_id][1] if slot_exists else None
        target_curr_idx = alive_slots_by_id[target_slot_id][0] if slot_exists else None

        # 2. Extract candidate information
        fused_indices = step_result.get("top_k_indices", [])
        retrieved_slot_ids = []
        for idx in fused_indices:
            if idx < len(memory.slots):
                retrieved_slot_ids.append(id(memory.slots[idx]))

        correct_slot_retrieved = False
        retrieval_rank = None
        if slot_exists and target_curr_idx is not None:
            if target_curr_idx in fused_indices:
                correct_slot_retrieved = True
                retrieval_rank = fused_indices.index(target_curr_idx)

        # 3. NLI and action data
        action = step_result.get("action", "")
        gate = step_result.get("gate", "")
        sem_pred = step_result.get("sem_pred") or {}
        nli_called = (step_result.get("candidates_examined", 0) > 0)
        nli_score = sem_pred.get("p_entail") if sem_pred else None
        nli_accepted = (gate == "nli_accept")
        is_merge = (action == "update")
        consolidation_attempted = is_merge
        consolidation_succeeded = is_merge and (step_result.get("slot_index") == target_curr_idx)

        # 4. Correctness definition
        is_para = category in ("paraphrase", "delayed_paraphrase", "pre_drift_probe", "post_drift_probe")
        is_contra = category in ("contradiction", "drift_contra")
        is_unrelated = category in ("distractor", "unrelated")

        if is_para:
            final_correct = is_merge and (step_result.get("slot_index") == target_curr_idx)
        elif is_contra or is_unrelated:
            # Contradictions / distractors must NOT merge into target slot
            target_merged = is_merge and (step_result.get("slot_index") == target_curr_idx)
            final_correct = not target_merged
        else:
            final_correct = True

        # 5. Classify Failure Category
        if final_correct:
            category_code = "SUCCESS"
        elif target_rec.evicted or not slot_exists:
            # Slot was evicted prior to or at probe time
            category_code = "D"
        elif is_para:
            # Paraphrase expected to merge into target slot
            if not correct_slot_retrieved:
                # Slot is alive, but was not in candidate beam
                category_code = "A"  # or E
            else:
                # Correct slot WAS retrieved in candidate set
                if not nli_accepted:
                    # NLI rejected the correct candidate
                    category_code = "B"
                else:
                    # NLI accepted, but consolidation failed
                    if not consolidation_succeeded:
                        category_code = "C"
                    else:
                        category_code = "F"
        elif is_contra or is_unrelated:
            # Contradiction / distractor falsely merged into target slot
            if correct_slot_retrieved and is_merge and (step_result.get("slot_index") == target_curr_idx):
                # NLI falsely accepted contradiction/distractor
                category_code = "F"
            else:
                category_code = "UNKNOWN"
        else:
            category_code = "UNKNOWN"

        # Drift metrics if slot is alive
        emb_before = None
        emb_after = None
        if target_slot is not None and target_slot.value is not None:
            emb_after = target_slot.value.tolist()[:10]

        rec = FailureAttributionRecord(
            target_id=target_id,
            probe_id=probe_id,
            target_slot_id=target_slot_id,
            slot_exists_at_probe=slot_exists,
            slot_evicted=target_rec.evicted,
            retrieved_slot_ids=retrieved_slot_ids,
            correct_slot_retrieved=correct_slot_retrieved,
            retrieval_rank=retrieval_rank,
            nli_called=nli_called,
            nli_score=nli_score,
            nli_accepted=nli_accepted,
            consolidation_attempted=consolidation_attempted,
            consolidation_succeeded=consolidation_succeeded,
            slot_update_count=target_slot.update_count if target_slot else 0,
            slot_embedding_before=emb_before,
            slot_embedding_after=emb_after,
            probe_retrieval_rank=retrieval_rank,
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
            cat = r.failure_category
            counts[cat] = counts.get(cat, 0) + 1
        pcts = {cat: round(cnt / total * 100.0, 2) for cat, cnt in counts.items()}
        return {
            "total": total,
            "counts": counts,
            "percentages": pcts
        }
