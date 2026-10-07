"""
Realistic-conversation benchmark (Tasks 1-14 from the task spec).

Three parts, run separately and reported separately:

  1. run_scale_benchmark(n) -- ingest n conversations, measure compression,
     recall accuracy, distractor resistance, false-memory rate. Builds a
     CONSOLIDATED store (normal absorb, paraphrases reinforce one record)
     and an UNCONSOLIDATED store (dedupe=False, every statement is its own
     record) side by side, so the consolidation benefit (Task 4/13) is an
     isolated, measured comparison rather than an assumption.

  2. run_temporal_test() -- Task 3's Day 1/5/10/20 preference-change
     sequence. Verifies current vs. historical recall.

  3. main() -- runs both, prints a before/after report against the
     predicate-interned synthetic-fact baseline (0.333x @ 10K), and does
     NOT touch that existing benchmark or its code.

Run with: python benchmarks/conversation_benchmark.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory import MemoryStore
from benchmarks.conversation_generator import generate_dataset
from benchmarks.conversation_extractor import ConversationExtractor


# ---------------------------------------------------------------------------
# Part 1: scale benchmark
# ---------------------------------------------------------------------------

def run_scale_benchmark(n: int, out_dir: str, extractor=None, label: str = "rule") -> dict:
    dataset = generate_dataset(n)
    if extractor is None:
        extractor = ConversationExtractor()

    consolidated = MemoryStore()
    unconsolidated = MemoryStore()

    raw_input_bytes = 0
    n_raw_statements_extracted = 0
    n_distractors_total = 0
    n_distractors_wrongly_extracted = 0

    t_absorb_start = time.perf_counter()
    for convo in dataset:
        raw_input_bytes += len(convo["text"].encode("utf-8"))
        n_distractors_total += convo["ground_truth"]["n_distractors"]
        default_subject = convo["ground_truth"]["subject"]

        for sentence in convo["sentences"]:
            fact = extractor.extract_sentence(sentence)
            is_distractor = sentence in _DISTRACTOR_SET
            if fact is None:
                continue
            if is_distractor:
                n_distractors_wrongly_extracted += 1  # false positive
            n_raw_statements_extracted += 1
            subject = getattr(fact, "subject", None) or default_subject
            confidence = getattr(fact, "confidence", None)
            consolidated.absorb(subject, fact.predicate, fact.object,
                                 base_confidence=confidence)
            unconsolidated.absorb(subject, fact.predicate, fact.object,
                                   base_confidence=confidence, dedupe=False)
    absorb_elapsed = time.perf_counter() - t_absorb_start
    avg_absorb_latency_ms = (absorb_elapsed / max(n_raw_statements_extracted, 1)) * 1000

    # -- checkpoint both stores, measure real bytes --
    cons_path = os.path.join(out_dir, f"consolidated_{label}_{n}.json")
    unc_path = os.path.join(out_dir, f"unconsolidated_{label}_{n}.json")
    consolidated.checkpoint(cons_path)
    unconsolidated.checkpoint(unc_path)
    consolidated_bytes = os.path.getsize(cons_path)
    unconsolidated_bytes = os.path.getsize(unc_path)

    # -- recall accuracy against hidden ground truth --
    fields_to_check = ["backend_language", "framework", "database",
                        "deploy_platform", "known_failure", "lesson", "depends_on"]
    correct = 0
    total = 0
    query_times = []
    for convo in dataset:
        gt = convo["ground_truth"]
        subject = gt["subject"]
        for field in fields_to_check:
            t0 = time.perf_counter()
            result = consolidated.recall(subject, field)
            query_times.append(time.perf_counter() - t0)
            total += 1
            if result and result[0].object == gt[field]:
                correct += 1
    recall_accuracy = correct / total
    avg_recall_latency_ms = (sum(query_times) / len(query_times)) * 1000

    # -- false memory: query a predicate & subject that never existed --
    rng = random.Random(555)
    false_hits = 0
    false_queries = min(200, n)
    for i in range(false_queries):
        result = consolidated.recall(f"ghost_person_{i}", "favorite_color")
        if result:
            false_hits += 1
    false_memory_rate = false_hits / false_queries

    distractor_leak_rate = (n_distractors_wrongly_extracted / n_distractors_total
                             if n_distractors_total else 0.0)

    compression_ratio = raw_input_bytes / consolidated_bytes
    consolidation_gain = unconsolidated_bytes / consolidated_bytes  # >1 means consolidation helped
    knowledge_retention = recall_accuracy  # correct useful memories / useful memories present
    memory_efficiency = knowledge_retention / consolidated_bytes

    return {
        "n_conversations": n,
        "raw_input_bytes": raw_input_bytes,
        "consolidated_memory_bytes": consolidated_bytes,
        "unconsolidated_memory_bytes": unconsolidated_bytes,
        "compression_ratio": round(compression_ratio, 4),
        "consolidation_gain_vs_unconsolidated": round(consolidation_gain, 4),
        "n_raw_statements_extracted": n_raw_statements_extracted,
        "n_consolidated_active_facts": consolidated.stats()["active_facts"],
        "recall_accuracy": round(recall_accuracy, 4),
        "false_memory_rate": round(false_memory_rate, 4),
        "distractor_leak_rate": round(distractor_leak_rate, 4),
        "avg_absorb_latency_ms": round(avg_absorb_latency_ms, 4),
        "avg_recall_latency_ms": round(avg_recall_latency_ms, 4),
        "knowledge_retention": round(knowledge_retention, 4),
        "memory_efficiency": memory_efficiency,
    }


_DISTRACTOR_SET = None  # populated in main() once generator module is available


# ---------------------------------------------------------------------------
# Part 2: temporal / continual-learning test (Task 3)
# ---------------------------------------------------------------------------

def run_temporal_test() -> dict:
    store = MemoryStore()
    subject = "temporal_test_person"

    day = lambda n: n * 86400.0  # simulated seconds-since-epoch by "day"

    store.absorb(subject, "prefers_language", "Python", at_time=day(1))
    store.absorb(subject, "prefers_language", "Python", at_time=day(5))   # reinforce
    store.absorb(subject, "prefers_language", "Rust", at_time=day(10))   # contradiction / switch
    store.absorb(subject, "prefers_language", "Rust", at_time=day(20))   # reinforce new pref

    current = store.recall(subject, "prefers_language")
    historical = store.recall(subject, "prefers_language", as_of=day(7))
    full_history = store.recall(subject, "prefers_language", include_history=True)

    current_correct = bool(current) and current[0].object == "Rust"
    historical_correct = bool(historical) and historical[0].object == "Python"
    history_preserved = len(full_history) == 2 and {f.object for f in full_history} == {"Python", "Rust"}

    return {
        "current_preference": current[0].object if current else None,
        "current_correct": current_correct,
        "historical_preference_as_of_day7": historical[0].object if historical else None,
        "historical_correct": historical_correct,
        "both_current_and_historical_preserved": history_preserved,
        "current_source_count": current[0].source_count if current else None,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global _DISTRACTOR_SET
    from benchmarks.conversation_generator import DISTRACTORS
    _DISTRACTOR_SET = set(DISTRACTORS)

    out_dir = os.path.join(os.path.dirname(__file__), "_conv_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("PART 1: SCALE BENCHMARK (realistic conversations)")
    print("=" * 70)
    scales = [100, 1_000, 10_000]
    results = []
    for n in scales:
        print(f"Running n={n} conversations ...")
        r = run_scale_benchmark(n, out_dir)
        results.append(r)

    cols = ["n_conversations", "raw_input_bytes", "consolidated_memory_bytes",
            "compression_ratio", "consolidation_gain_vs_unconsolidated",
            "recall_accuracy", "false_memory_rate", "distractor_leak_rate",
            "avg_absorb_latency_ms", "avg_recall_latency_ms"]
    w = 20
    print("\n" + "".join(c.ljust(w) for c in cols))
    for r in results:
        print("".join(str(r[c]).ljust(w) for c in cols))

    print("\n" + "=" * 70)
    print("PART 2: TEMPORAL / CONTINUAL-LEARNING TEST (Task 3)")
    print("=" * 70)
    temporal = run_temporal_test()
    for k, v in temporal.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("PART 3: BEFORE / AFTER REPORT")
    print("=" * 70)
    print("Predicate-interned synthetic-fact benchmark (previous phase):")
    print("  100 facts  = 0.313x")
    print("  1K facts   = 0.325x")
    print("  10K facts  = 0.333x")
    print("  accuracy   = 1.0")
    print("  false-memory = 0.0")
    print()
    print("Realistic-conversation benchmark (this phase, actual measured):")
    for r in results:
        print(f"  {r['n_conversations']} conversations:")
        print(f"    compression_ratio (raw text -> memory)        = {r['compression_ratio']}x")
        print(f"    consolidation_gain (vs. one-record-per-stmt)  = {r['consolidation_gain_vs_unconsolidated']}x")
        print(f"    recall_accuracy                                = {r['recall_accuracy']}")
        print(f"    false_memory_rate                              = {r['false_memory_rate']}")
        print(f"    distractor_leak_rate                           = {r['distractor_leak_rate']}")
    print()
    print("Temporal accuracy (Task 3 dedicated test):")
    print(f"  current preference correct    = {temporal['current_correct']}")
    print(f"  historical preference correct = {temporal['historical_correct']}")
    print(f"  both preserved simultaneously = {temporal['both_current_and_historical_preserved']}")

    return results, temporal


if __name__ == "__main__":
    main()
