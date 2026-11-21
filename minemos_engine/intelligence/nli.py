"""
Natural-language-inference verification gate.

Decides whether a new observation restates an existing memory, revises it, or
describes something unrelated, and exposes the bidirectional comparison used
by the memory decision pipeline.
- A: Lexical baseline (FieldAwareLexicalEncoder, MATCH_THRESHOLD=0.75)
- B: NLI semantic gate (lexical candidate retrieval + CrossEncoder NLI verification)
- C: Oracle gate (ground-truth concept & category matching control)
- D: Diagnostic exhaustive NLI gate (all-slot NLI evaluation control)

The update equation's single blend line:
    slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
is identical across all configurations.
"""

from __future__ import annotations
import os, sys, time, threading
from typing import Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from minemos_engine.structured_candidate import Candidate
from minemos_engine.memory_state import MemoryState, MemorySlot
from minemos_engine.memory_update import address, novelty, cosine_similarity, MATCH_THRESHOLD
from minemos_engine.encoders import FieldAwareLexicalEncoder
from minemos_engine.model_provisioning import (
    load_sequence_classification_model,
    load_tokenizer,
)

D = 4096
ENCODER = FieldAwareLexicalEncoder()
MODEL_NAME = "cross-encoder/nli-distilroberta-base"


class NliSemanticGate:
    """Verified pretrained cross-encoder NLI classifier."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu",
                 cache_dir: Optional[str] = None,
                 local_files_only: Optional[bool] = None):
        self.device = device
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = load_tokenizer(
            model_name, cache_dir=cache_dir,
            local_files_only=local_files_only)
        self.model = load_sequence_classification_model(
            model_name, cache_dir=cache_dir,
            local_files_only=local_files_only, device=device)
        # Verified configuration indices: 0: contradiction, 1: entailment, 2: neutral
        self.id2label = self.model.config.id2label

    def predict_pair(self, premise: str, hypothesis: str,
                     entail_threshold: float = 0.50,
                     contra_threshold: float = 0.50,
                     abstain_min_confidence: float = 0.35) -> dict:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        p_contra, p_entail, p_neutral = probs[0], probs[1], probs[2]
        max_prob = max(probs)

        if max_prob < abstain_min_confidence:
            decision = "ABSTAIN"
        elif p_entail >= entail_threshold:
            decision = "SAME"
        elif p_contra >= contra_threshold:
            decision = "CONTRADICTION"
        else:
            decision = "NEUTRAL"

        return dict(
            decision=decision,
            p_entail=p_entail,
            p_contra=p_contra,
            p_neutral=p_neutral,
            confidence=max_prob
        )

    def predict_batch(self, pairs: list[tuple[str, str]],
                      entail_threshold: float = 0.50,
                      contra_threshold: float = 0.50,
                      abstain_min_confidence: float = 0.35) -> list[dict]:
        if not pairs:
            return []
        premises = [p[0] for p in pairs]
        hypotheses = [p[1] for p in pairs]
        inputs = self.tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().tolist()

        results = []
        for p in probs:
            p_contra, p_entail, p_neutral = p[0], p[1], p[2]
            max_prob = max(p)
            if max_prob < abstain_min_confidence:
                decision = "ABSTAIN"
            elif p_entail >= entail_threshold:
                decision = "SAME"
            elif p_contra >= contra_threshold:
                decision = "CONTRADICTION"
            else:
                decision = "NEUTRAL"
            results.append(dict(
                decision=decision,
                p_entail=p_entail,
                p_contra=p_contra,
                p_neutral=p_neutral,
                confidence=max_prob
            ))
        return results


class LazyNliSemanticGate:
    """A :class:`NliSemanticGate` that loads its weights on first use.

    Creating a memory or restoring a checkpoint does not need the NLI model;
    it is downloaded and loaded only when the first pair is actually verified.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu",
                 cache_dir: Optional[str] = None,
                 local_files_only: Optional[bool] = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._gate: Optional[NliSemanticGate] = None
        self._load_lock = threading.Lock()

    def _resolve(self) -> NliSemanticGate:
        if self._gate is None:
            with self._load_lock:
                if self._gate is None:
                    self._gate = NliSemanticGate(
                        self.model_name, device=self.device,
                        cache_dir=self.cache_dir,
                        local_files_only=self.local_files_only)
        return self._gate

    def predict_pair(self, premise: str, hypothesis: str, **kwargs) -> dict:
        return self._resolve().predict_pair(premise, hypothesis, **kwargs)

    def predict_batch(self, pairs: list[tuple[str, str]], **kwargs) -> list[dict]:
        return self._resolve().predict_batch(pairs, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)


def lexical_absorb(memory: MemoryState, candidate: Candidate,
                   threshold: float = MATCH_THRESHOLD,
                   encoder=ENCODER, dim: int = D) -> dict:
    """A: pure lexical baseline."""
    x_t = encoder.encode_full(candidate, dim=dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t

    if best_idx is not None and max_sim >= threshold:
        slot = memory.slots[best_idx]
        # --- identical update line to memory_update.absorb() ---
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        return dict(action="update", slot_index=best_idx, similarity=max_sim, novelty=n_t, gate="lexical_accept")

    if not memory.is_full():
        memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                   debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        return dict(action="insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate="new_slot")

    memory.evict_oldest()
    memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
               debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate="evict_insert")


def nli_gated_absorb(memory: MemoryState, candidate: Candidate,
                     slot_texts: dict[int, str],
                     semantic_gate: NliSemanticGate,
                     entail_threshold: float = 0.50,
                     encoder=ENCODER, dim: int = D) -> dict:
    """B: lexical retrieval + NLI verification gate."""
    x_t = encoder.encode_full(candidate, dim=dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    target_idx = None
    target_pred = None
    gate_reason = "new_slot"

    if best_idx is not None:
        slot = memory.slots[best_idx]
        premise_text = slot_texts[id(slot)]
        pred = semantic_gate.predict_pair(premise_text, cand_text, entail_threshold=entail_threshold)
        if pred["decision"] == "SAME":
            target_idx = best_idx
            target_pred = pred
        else:
            gate_reason = "rejected_contra" if pred["decision"] == "CONTRADICTION" else "rejected_diff"

    if target_idx is not None:
        slot = memory.slots[target_idx]
        # --- identical update line to memory_update.absorb() ---
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        return dict(action="update", slot_index=target_idx, similarity=max_sim, novelty=n_t,
                    gate="nli_accept", sem_pred=target_pred)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        return dict(action="insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate=gate_reason)

    evicted = memory.evict_oldest()
    if id(evicted) in slot_texts:
        del slot_texts[id(evicted)]
    slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_texts[id(slot)] = cand_text
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate=gate_reason)


def oracle_absorb(memory: MemoryState, candidate: Candidate,
                  slot_concept_ids: dict[int, str],
                  concept_id: str, category: str,
                  encoder=ENCODER, dim: int = D) -> dict:
    """C: oracle upper-bound control."""
    x_t = encoder.encode_full(candidate, dim=dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t

    target_idx = None
    if category not in ("hard_negative", "unrelated", "contradiction"):
        for idx, slot in enumerate(memory.slots):
            if slot_concept_ids.get(id(slot)) == concept_id:
                target_idx = idx
                break

    if target_idx is not None:
        slot = memory.slots[target_idx]
        # --- identical update line to memory_update.absorb() ---
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        return dict(action="update", slot_index=target_idx, similarity=max_sim, novelty=n_t, gate="oracle_accept")

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_concept_ids[id(slot)] = concept_id
        return dict(action="insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate="oracle_new_slot")

    evicted = memory.evict_oldest()
    if id(evicted) in slot_concept_ids:
        del slot_concept_ids[id(evicted)]
    slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_concept_ids[id(slot)] = concept_id
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate="oracle_evict_insert")


def exhaustive_nli_absorb(memory: MemoryState, candidate: Candidate,
                          slot_texts: dict[int, str],
                          semantic_gate: NliSemanticGate,
                          entail_threshold: float = 0.50,
                          encoder=ENCODER, dim: int = D) -> dict:
    """D (diagnostic control): NLI evaluated against all existing slots."""
    x_t = encoder.encode_full(candidate, dim=dim)
    best_idx, max_sim = address(memory, x_t)
    n_t = novelty(max_sim, memory_empty=(best_idx is None))
    alpha_t = n_t
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    target_idx = None
    target_pred = None
    if memory.slots:
        # Evaluate against all slots
        pairs = [(slot_texts[id(s)], cand_text) for s in memory.slots]
        preds = semantic_gate.predict_batch(pairs, entail_threshold=entail_threshold)
        best_entail = -1.0
        for idx, pred in enumerate(preds):
            if pred["decision"] == "SAME" and pred["p_entail"] > best_entail:
                best_entail = pred["p_entail"]
                target_idx = idx
                target_pred = pred

    if target_idx is not None:
        slot = memory.slots[target_idx]
        # --- identical update line to memory_update.absorb() ---
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        # ---------------------------------------------------------------
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = candidate.timestamp
        slot.update_count += 1
        return dict(action="update", slot_index=target_idx, similarity=max_sim, novelty=n_t,
                    gate="exhaustive_accept", sem_pred=target_pred)

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        return dict(action="insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate="new_slot")

    evicted = memory.evict_oldest()
    if id(evicted) in slot_texts:
        del slot_texts[id(evicted)]
    slot = memory.add(x_t, x_t.copy(), candidate.confidence, candidate.timestamp,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_texts[id(slot)] = cand_text
    return dict(action="evict_insert", slot_index=len(memory.slots) - 1, similarity=max_sim, novelty=n_t, gate="evict_insert")


def run_system(observations, capacity: int, system: str,
                       semantic_gate: NliSemanticGate = None,
                       entail_threshold: float = 0.50,
                       dim: int = D) -> dict:
    mem = MemoryState(capacity=capacity, dim=dim)
    slot_texts = {}
    slot_concept_ids = {}

    n_updates = n_inserts = n_evicts = 0
    gate_counts = {}
    t0 = time.perf_counter()

    for obs in observations:
        candidate = obs.candidate if hasattr(obs, "candidate") else obs
        concept_id = getattr(obs, "concept_id", None)
        category = getattr(obs, "category", None)

        if system == "A_lexical":
            r = lexical_absorb(mem, candidate, threshold=MATCH_THRESHOLD, encoder=ENCODER, dim=dim)
        elif system == "B_nli_gate":
            r = nli_gated_absorb(mem, candidate, slot_texts, semantic_gate, entail_threshold=entail_threshold, dim=dim)
        elif system == "C_oracle":
            r = oracle_absorb(mem, candidate, slot_concept_ids, concept_id, category, dim=dim)
        elif system == "D_exhaustive_nli":
            r = exhaustive_nli_absorb(mem, candidate, slot_texts, semantic_gate, entail_threshold=entail_threshold, dim=dim)
        else:
            raise ValueError(f"Unknown system: {system}")

        if r["action"] == "update":
            n_updates += 1
        elif r["action"] == "insert":
            n_inserts += 1
        elif r["action"] == "evict_insert":
            n_evicts += 1

        gate = r.get("gate", "unknown")
        gate_counts[gate] = gate_counts.get(gate, 0) + 1

    elapsed = time.perf_counter() - t0
    return dict(
        system=system, memory=mem, n_slots=len(mem.slots),
        n_updates=n_updates, n_inserts=n_inserts, n_evicts=n_evicts,
        gate_counts=gate_counts, wall_seconds=elapsed
    )
