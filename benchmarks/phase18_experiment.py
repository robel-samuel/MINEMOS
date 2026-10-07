"""
benchmarks/phase18_experiment.py

Phase 18 Experiment Architecture:
Semantic Dense Retrieval + Utility-Aware Eviction.

2x2 Factorial Systems:
  System A: Lexical Retrieval + FIFO Eviction (Phase 17 baseline reproduction)
  System B: Dense Semantic Retrieval + FIFO Eviction
  System C: Lexical Retrieval + Utility-Aware Eviction
  System D: Dense Semantic Retrieval + Utility-Aware Eviction

Core Invariants (STRICTLY PRESERVED):
  - NLI Model: cross-encoder/nli-distilroberta-base (local_files_only=True)
  - NLI Policy: entailment >= 0.50 -> SAME; contradiction >= 0.50 -> CONTRADICTION; abstain < 0.35
  - MATCH_THRESHOLD: 0.75
  - Core Update Equation: slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
  - Online Metadata Only: Eviction uses ONLY online slot metadata (update_count, timestamp, confidence)
"""

from __future__ import annotations
import os, sys, time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from phase2.structured_candidate import Candidate
from phase2.memory_state import MemoryState, MemorySlot
from phase2.memory_update import address, novelty, cosine_similarity, MATCH_THRESHOLD
from phase2.encoders import FieldAwareLexicalEncoder
from benchmarks.phase16_experiment import NliSemanticGate

D_LEX = 4096
D_DENSE = 384
LEXICAL_ENCODER = FieldAwareLexicalEncoder()
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ──────────────────────────────────────────────
#  Dense Bi-Encoder Retrieval Layer
# ──────────────────────────────────────────────

class DenseSemanticEncoder:
    """
    Genuine dense bi-encoder sentence embedding model.
    Model: sentence-transformers/all-MiniLM-L6-v2
    Embedding Dimension: 384
    Pooling: Mean pooling with attention mask
    Normalization: L2 normalization
    Execution: Local PyTorch on CPU (local_files_only=True)
    """

    def __init__(self, model_name: str = DENSE_MODEL_NAME, device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        self.model.eval()
        self.dim = D_DENSE

    def encode(self, text: str) -> np.ndarray:
        """Encodes a single sentence into a normalized 384-dim numpy array."""
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized[0].cpu().numpy().astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encodes a batch of sentences into normalized (N, 384) numpy array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32)


def dense_address(slot_embeddings: list[np.ndarray], cand_embedding: np.ndarray,
                  top_k: int = 1) -> list[tuple[int, float]]:
    """
    Ranks memory slots by cosine similarity against candidate embedding.
    Because embeddings are L2 normalized, cosine similarity is the dot product.
    Returns list of (slot_index, similarity) sorted descending by similarity.
    """
    if not slot_embeddings:
        return []
    matrix = np.stack(slot_embeddings)  # (N, 384)
    sims = np.dot(matrix, cand_embedding)  # (N,)
    k = min(top_k, len(slot_embeddings))
    top_indices = np.argsort(-sims)[:k]
    return [(int(idx), float(sims[idx])) for idx in top_indices]


# ──────────────────────────────────────────────
#  Utility-Aware Eviction Function
# ──────────────────────────────────────────────

def utility_score(slot: MemorySlot, current_time: float, tau: float = 50.0) -> float:
    """
    Online utility function using ONLY slot metadata available online:
      - consolidation / update_count: min(update_count / 5.0, 1.0) (weight = 0.50)
      - recency: exp(-max(0, current_time - slot.timestamp) / tau) (weight = 0.30)
      - confidence: slot.confidence (weight = 0.20)
    
    Zero future knowledge, zero oracle concept IDs, zero test labels.
    """
    u_consolidation = min(float(slot.update_count) / 5.0, 1.0) * 0.50
    dt = max(0.0, float(current_time) - float(slot.timestamp))
    u_recency = float(np.exp(-dt / tau)) * 0.30
    u_confidence = float(slot.confidence) * 0.20
    return u_consolidation + u_recency + u_confidence


def evict_slot(memory: MemoryState, eviction_policy: str, current_time: float,
               slot_texts: dict[int, str],
               slot_dense_embs: list[np.ndarray],
               slot_concept_ids: dict[int, str] = None) -> tuple[int, MemorySlot]:
    """
    Evicts a single slot according to the chosen policy:
      - 'fifo': oldest created_at timestamp
      - 'utility': lowest utility score U(s, t)
    Synchronizes memory.slots, slot_texts, slot_dense_embs, and slot_concept_ids.
    """
    if not memory.slots:
        raise RuntimeError("Cannot evict from empty memory")

    if eviction_policy == "fifo":
        evict_idx = int(np.argmin([s.created_at for s in memory.slots]))
    elif eviction_policy == "utility":
        scores = [utility_score(s, current_time) for s in memory.slots]
        evict_idx = int(np.argmin(scores))
    else:
        raise ValueError(f"Unknown eviction policy: {eviction_policy}")

    evicted_slot = memory.slots.pop(evict_idx)
    if id(evicted_slot) in slot_texts:
        del slot_texts[id(evicted_slot)]
    if slot_dense_embs and evict_idx < len(slot_dense_embs):
        slot_dense_embs.pop(evict_idx)
    if slot_concept_ids and id(evicted_slot) in slot_concept_ids:
        del slot_concept_ids[id(evicted_slot)]

    return evict_idx, evicted_slot


# ──────────────────────────────────────────────
#  2x2 Factorial Absorb Pipeline
# ──────────────────────────────────────────────

def absorb_observation(
    memory: MemoryState,
    candidate: Candidate,
    system: str,  # 'A', 'B', 'C', 'D'
    slot_texts: dict[int, str],
    slot_dense_embs: list[np.ndarray],
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder = None,
    top_k: int = 1,
    current_time: float = 0.0,
    entail_threshold: float = 0.50,
    lex_encoder: FieldAwareLexicalEncoder = LEXICAL_ENCODER,
    dim: int = D_LEX
) -> dict:
    """
    Unified 2x2 Factorial Absorb Pipeline:
      System A: Lexical Retrieval + FIFO Eviction
      System B: Dense Semantic Retrieval + FIFO Eviction
      System C: Lexical Retrieval + Utility-Aware Eviction
      System D: Dense Semantic Retrieval + Utility-Aware Eviction

    Enforces:
      - Core update equation is identical for all systems:
          slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
      - Dense similarity NEVER directly authorizes a merge.
        NLI semantic gate is ALWAYS the final decision maker.
    """
    # 1. Lexical encoding for core memory value
    x_t = lex_encoder.encode_full(candidate, dim=dim)
    cand_text = f"{candidate.subject} {candidate.predicate} {candidate.object}."

    is_dense_retrieval = (system in ("B", "D"))
    eviction_policy = "utility" if (system in ("C", "D")) else "fifo"

    t_emb = 0.0
    t_retr = 0.0
    t_nli = 0.0

    target_idx = None
    target_pred = None
    gate_reason = "new_slot"
    candidates_examined = 0
    top_k_indices = []

    # 2. Candidate Retrieval (Stage 1)
    if memory.slots:
        if is_dense_retrieval:
            # Dense bi-encoder candidate generator
            t0_emb = time.perf_counter()
            cand_dense = dense_encoder.encode(cand_text)
            t_emb += (time.perf_counter() - t0_emb)

            t0_retr = time.perf_counter()
            ranked_candidates = dense_address(slot_dense_embs, cand_dense, top_k=top_k)
            t_retr += (time.perf_counter() - t0_retr)

            top_k_indices = [idx for idx, _ in ranked_candidates]

            # Stage 2: NLI Verification across top-k candidates
            t0_nli = time.perf_counter()
            for cand_idx, sim in ranked_candidates:
                candidates_examined += 1
                slot = memory.slots[cand_idx]
                premise_text = slot_texts[id(slot)]
                pred = nli_gate.predict_pair(premise_text, cand_text, entail_threshold=entail_threshold)
                if pred["decision"] == "SAME":
                    target_idx = cand_idx
                    target_pred = pred
                    gate_reason = "nli_accept"
                    break
                elif pred["decision"] == "CONTRADICTION" and gate_reason == "new_slot":
                    gate_reason = "rejected_contra"
                elif gate_reason == "new_slot":
                    gate_reason = "rejected_neutral"
            t_nli += (time.perf_counter() - t0_nli)

            # Novelty based on best dense similarity
            best_sim = ranked_candidates[0][1] if ranked_candidates else 0.0
            n_t = novelty(best_sim, memory_empty=False)
            best_retrieved_idx = ranked_candidates[0][0] if ranked_candidates else None

        else:
            # Lexical sparse candidate generator
            t0_retr = time.perf_counter()
            best_idx, max_sim = address(memory, x_t)
            t_retr += (time.perf_counter() - t0_retr)

            best_retrieved_idx = best_idx
            best_sim = max_sim
            n_t = novelty(max_sim, memory_empty=(best_idx is None))
            top_k_indices = [best_idx] if best_idx is not None else []

            if best_idx is not None:
                candidates_examined = 1
                slot = memory.slots[best_idx]
                premise_text = slot_texts[id(slot)]
                t0_nli = time.perf_counter()
                pred = nli_gate.predict_pair(premise_text, cand_text, entail_threshold=entail_threshold)
                t_nli += (time.perf_counter() - t0_nli)
                if pred["decision"] == "SAME":
                    target_idx = best_idx
                    target_pred = pred
                    gate_reason = "nli_accept"
                else:
                    gate_reason = "rejected_contra" if pred["decision"] == "CONTRADICTION" else "rejected_neutral"
    else:
        best_retrieved_idx = None
        best_sim = 0.0
        n_t = 1.0

    alpha_t = n_t

    # 3. Memory Update / Insertion Path
    if target_idx is not None:
        slot = memory.slots[target_idx]
        # Core update equation: identical across all systems
        slot.value = (1 - alpha_t) * slot.value + alpha_t * x_t
        slot.confidence = max(slot.confidence, candidate.confidence)
        slot.timestamp = current_time
        slot.update_count += 1
        return dict(
            action="update", slot_index=target_idx, similarity=best_sim, novelty=n_t,
            gate="nli_accept", sem_pred=target_pred, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
            candidates_examined=candidates_examined,
            t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
        )

    # 4. Allocate New Slot (or Evict if Full)
    if is_dense_retrieval and dense_encoder is not None:
        if t_emb == 0.0:
            t0_emb = time.perf_counter()
            cand_dense = dense_encoder.encode(cand_text)
            t_emb += (time.perf_counter() - t0_emb)
    else:
        cand_dense = None

    if not memory.is_full():
        slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                          debug_subject=candidate.subject, debug_predicate=candidate.predicate)
        slot_texts[id(slot)] = cand_text
        if is_dense_retrieval and cand_dense is not None:
            slot_dense_embs.append(cand_dense)
        return dict(
            action="insert", slot_index=len(memory.slots) - 1, similarity=best_sim,
            novelty=n_t, gate=gate_reason, cand_text=cand_text,
            retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
            candidates_examined=candidates_examined,
            t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
        )

    # Eviction required
    evict_idx, evicted_slot = evict_slot(
        memory, eviction_policy, current_time, slot_texts, slot_dense_embs
    )
    slot = memory.add(x_t, x_t.copy(), candidate.confidence, current_time,
                      debug_subject=candidate.subject, debug_predicate=candidate.predicate)
    slot_texts[id(slot)] = cand_text
    if is_dense_retrieval and cand_dense is not None:
        slot_dense_embs.append(cand_dense)

    return dict(
        action="evict_insert", slot_index=len(memory.slots) - 1, similarity=best_sim,
        novelty=n_t, gate=gate_reason, cand_text=cand_text,
        retrieved_idx=best_retrieved_idx, top_k_indices=top_k_indices,
        candidates_examined=candidates_examined, evicted_idx=evict_idx,
        t_emb=t_emb, t_retr=t_retr, t_nli=t_nli
    )


# ──────────────────────────────────────────────
#  Stream Execution Harness
# ──────────────────────────────────────────────

def run_phase18_stream(
    observations,
    capacity: int,
    system: str,  # 'A', 'B', 'C', 'D'
    nli_gate: NliSemanticGate,
    dense_encoder: DenseSemanticEncoder = None,
    top_k: int = 1,
    record_diagnostics: bool = True,
    dim: int = D_LEX
) -> dict:
    """
    Runs a stream of Phase18Obs through the chosen factorial system.
    Records comprehensive diagnostic traces for decomposition analysis.
    """
    memory = MemoryState(capacity=capacity, dim=dim)
    slot_texts: dict[int, str] = {}
    slot_dense_embs: list[np.ndarray] = []

    n_updates = n_inserts = n_evicts = 0
    nli_calls = 0
    gate_counts: dict[str, int] = {}
    trace = []
    total_t_emb = total_t_retr = total_t_nli = 0.0

    t_start = time.perf_counter()

    for idx, obs in enumerate(observations):
        current_time = float(getattr(obs.candidate, "timestamp", idx))

        r = absorb_observation(
            memory=memory,
            candidate=obs.candidate,
            system=system,
            slot_texts=slot_texts,
            slot_dense_embs=slot_dense_embs,
            nli_gate=nli_gate,
            dense_encoder=dense_encoder,
            top_k=top_k,
            current_time=current_time,
            dim=dim
        )

        if r["action"] == "update":
            n_updates += 1
        elif r["action"] == "insert":
            n_inserts += 1
        elif r["action"] == "evict_insert":
            n_evicts += 1

        g = r.get("gate", "unknown")
        gate_counts[g] = gate_counts.get(g, 0) + 1
        nli_calls += r.get("candidates_examined", 0)

        total_t_emb += r.get("t_emb", 0.0)
        total_t_retr += r.get("t_retr", 0.0)
        total_t_nli += r.get("t_nli", 0.0)

        if record_diagnostics:
            trace.append({
                "step": idx,
                "concept_id": obs.concept_id,
                "category": obs.category,
                "scenario": getattr(obs, "scenario", ""),
                "is_probe": obs.is_probe,
                "action": r["action"],
                "gate": g,
                "slot_index": r.get("slot_index"),
                "retrieved_idx": r.get("retrieved_idx"),
                "top_k_indices": r.get("top_k_indices", []),
                "similarity": r.get("similarity"),
                "candidates_examined": r.get("candidates_examined", 0),
                "sem_pred": r.get("sem_pred")
            })

    total_time = time.perf_counter() - t_start

    return dict(
        system=system,
        capacity=capacity,
        top_k=top_k,
        n_obs=len(observations),
        final_slots=len(memory.slots),
        n_updates=n_updates,
        n_inserts=n_inserts,
        n_evicts=n_evicts,
        nli_calls=nli_calls,
        gate_counts=gate_counts,
        wall_seconds=total_time,
        obs_per_second=round(len(observations) / total_time, 2) if total_time > 0 else 0.0,
        time_embedding=round(total_t_emb, 3),
        time_retrieval=round(total_t_retr, 3),
        time_nli=round(total_t_nli, 3),
        memory=memory,
        trace=trace if record_diagnostics else None
    )
