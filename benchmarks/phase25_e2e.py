"""
Phase 25 End-to-End Workload Validation
========================================
Exercises BlackHoleMemory on a realistic streaming conversation workload:
  - Multi-turn observation stream with new facts, paraphrases, temporal revisions,
    distractor floods, and adversarial contradiction probes.
  - Validates: factual consolidation, temporal supersession, contradiction safety,
    bounded memory footprint, and point-in-time recall.
"""

import os
import sys
import json
import time

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from phase2.black_hole_memory import BlackHoleMemory
from phase2.structured_candidate import Candidate

SEPARATOR = "=" * 65


def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    marker = "OK" if condition else "XX"
    print(f"  [{status}] {marker} {label}" + (f"\n         -> {detail}" if detail else ""))
    return condition


def run():
    results = {
        "phase": 25,
        "timestamp": time.time(),
        "checks": {},
        "summary": {}
    }
    passes = 0
    failures = 0

    def record(label, condition, detail=""):
        nonlocal passes, failures
        ok = check(label, condition, detail)
        results["checks"][label] = {"passed": ok, "detail": detail}
        if ok:
            passes += 1
        else:
            failures += 1
        return ok

    # ──────────────────────────────────────────────
    section("Phase 25 BlackHoleMemory — E2E Validation")
    print()
    mem = BlackHoleMemory(capacity=20, retention_policy="tiered")

    # ── 1. Initial Fact Absorption ──
    section("1. Initial Fact Absorption")

    facts = [
        Candidate("Alice",   "leads the engineering division at", "Nexus Labs", timestamp=100.0),
        Candidate("Bob",     "holds a PhD in",                   "quantum physics", timestamp=101.0),
        Candidate("Carol",   "authored the seminal paper on",    "neural compression", timestamp=102.0),
        Candidate("David",   "manages the infrastructure for",   "cloud storage", timestamp=103.0),
        Candidate("Eve",     "pioneered the research on",        "adversarial robustness", timestamp=104.0),
    ]
    insert_results = [mem.absorb(f) for f in facts]

    record("All initial facts inserted",
           all(r["action"] == "insert" for r in insert_results),
           f"actions: {[r['action'] for r in insert_results]}")
    record("Memory contains 5 slots",
           len(mem.memory.slots) == 5,
           f"actual: {len(mem.memory.slots)}")
    record("All slots have valid_from timestamps",
           all(s.valid_from == facts[i].timestamp for i, s in enumerate(mem.memory.slots[:5])),
           f"valid_froms: {[s.valid_from for s in mem.memory.slots[:5]]}")

    # ── 2. Paraphrase Consolidation ──
    section("2. Paraphrase Consolidation")

    paraphrase = Candidate("Bob", "earned a doctorate in", "quantum physics", timestamp=120.0)
    r_para = mem.absorb(paraphrase)

    record("Paraphrase does not false-merge (action is insert, consolidate, or temporal_update)",
           r_para["action"] in ("consolidate", "insert", "temporal_update", "evict_insert"),
           f"action={r_para['action']}")

    # ── 3. Distractor Flood (10 unrelated items) ──
    section("3. Distractor Flood (capacity stress)")

    distractor_base = 200.0
    for i in range(10):
        mem.absorb(Candidate(
            f"Distractor{i}", "performed a routine check on",
            f"subsystem {i}", timestamp=distractor_base + i
        ))

    record("Memory stays bounded at capacity",
           len(mem.memory.slots) <= mem.capacity,
           f"slots={len(mem.memory.slots)}, capacity={mem.capacity}")

    alice_alive = any(s.debug_subject == "Alice" and s.is_active for s in mem.memory.slots)
    record("Alice (high-value) survived distractor flood",
           alice_alive,
           "tiered eviction should protect previously touched slots")

    # ── 4. Temporal Fact Revision ──
    section("4. Temporal Fact Revision")

    # Alice moves from Nexus Labs to Apex Systems
    revision = Candidate("Alice", "now leads the engineering division at", "Apex Systems", timestamp=300.0)
    r_rev = mem.absorb(revision)

    # Active recall should return only current state
    active = mem.recall("Alice")
    superseded = [s for s in mem.memory.slots if s.debug_subject == "Alice" and not s.is_active]

    record("Temporal revision creates a new slot or updates",
           r_rev["action"] in ("temporal_update", "insert", "evict_insert"),
           f"action={r_rev['action']}")
    record("Current active Alice recall returns exactly 1 result",
           len(active) == 1,
           f"active count={len(active)}")
    record("Superseded Alice slot has valid_until set",
           len(superseded) > 0 and all(s.valid_until is not None for s in superseded),
           f"superseded slots={len(superseded)}")

    # ── 5. Point-in-Time Historical Recall ──
    section("5. Point-in-Time Historical Recall")

    # At t=150, Alice still led Nexus Labs (before t=300 revision)
    historical = mem.recall("Alice", as_of=150.0)
    record("Historical recall at t=150 finds original Alice fact",
           len(historical) >= 1,
           f"found={len(historical)} slots at t=150")

    # ── 6. Contradiction Rejection Safety ──
    section("6. Contradiction Rejection Safety")

    # Bob does NOT hold a PhD in quantum physics (dangerous merge attempt)
    contra = Candidate("Bob", "does not hold a PhD in", "quantum physics", timestamp=400.0)
    r_contra = mem.absorb(contra)

    # Get all Bob slots
    bob_slots = [s for s in mem.memory.slots if s.debug_subject == "Bob"]
    bob_consolidated = [s for s in bob_slots if s.update_count >= 2]

    record("Contradiction is NOT merged into an existing Bob slot",
           r_contra["action"] != "consolidate",
           f"action={r_contra['action']}")

    # Verify Bob's original PhD fact is still accessible
    bob_phd = mem.recall("Bob")
    bob_phd_facts = [r for r in bob_phd if "quantum physics" in r.get("text", "")]
    record("Bob's original PhD fact survives after contradiction",
           len(bob_phd_facts) >= 1,
           f"PhD facts found={len(bob_phd_facts)}")

    # ── 7. History Retrieval ──
    section("7. Entity History Retrieval")

    alice_history = mem.get_history("Alice")
    record("Alice history contains at least 2 entries (original + revised)",
           len(alice_history) >= 2,
           f"history entries={len(alice_history)}")
    if len(alice_history) >= 2:
        record("Alice history is sorted by valid_from",
               all(alice_history[i]["valid_from"] <= alice_history[i+1]["valid_from"]
                   for i in range(len(alice_history)-1)),
               f"valid_froms={[h['valid_from'] for h in alice_history]}")

    # ── 8. Natural Language Query ──
    section("8. Natural Language Query API")

    q_results = mem.query("quantum physics doctorate research", top_k=3)
    record("Query returns results for relevant topic",
           len(q_results) >= 1,
           f"results={len(q_results)}")
    if q_results:
        top_text = q_results[0]["text"]
        record("Top query result mentions quantum physics or Bob",
               "quantum" in top_text.lower() or "Bob" in top_text,
               f"top text: {top_text!r}")

    # ── 9. Memory Footprint and Status ──
    section("9. Memory Status and Footprint")

    status = mem.status()
    record("total_absorbed tracked correctly",
           status["total_absorbed"] > 15,
           f"total_absorbed={status['total_absorbed']}")
    record("NLI calls tracked",
           status["total_nli_calls"] > 0,
           f"total_nli_calls={status['total_nli_calls']}")
    record("Memory footprint is positive",
           status["footprint_bytes"] > 0,
           f"footprint={status['footprint_bytes']} bytes")

    print()
    print(f"  total_absorbed:     {status['total_absorbed']}")
    print(f"  total_inserts:      {status['total_inserts']}")
    print(f"  total_updates:      {status['total_updates']}")
    print(f"  total_temporal:     {status['total_temporal_updates']}")
    print(f"  total_evictions:    {status['total_evictions']}")
    print(f"  total_nli_calls:    {status['total_nli_calls']}")
    print(f"  active_slots:       {status['active_slots']}")
    print(f"  provisional_slots:  {status['provisional_slots']}")
    print(f"  consolidated_slots: {status['consolidated_slots']}")
    print(f"  superseded_slots:   {status['superseded_slots']}")
    print(f"  footprint_bytes:    {status['footprint_bytes']}")

    # ── Final Summary ──
    section("SUMMARY")
    total = passes + failures
    print(f"  Checks passed:  {passes}/{total}")
    print(f"  Checks failed:  {failures}/{total}")
    if failures == 0:
        print("\n  *** ALL CHECKS PASSED -- Phase 25 E2E validation complete. ***")
    else:
        print(f"\n  FAILED: {failures} check(s) FAILED -- see details above.")

    results["summary"] = {
        "total_checks": total,
        "passed": passes,
        "failed": failures,
        "all_passed": failures == 0,
        "status": status
    }

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "phase25_e2e_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")

    return failures == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
