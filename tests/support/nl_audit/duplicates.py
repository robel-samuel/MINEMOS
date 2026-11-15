"""Explicit duplicate / temporal / distinct-fact benchmark.

This suite exists to measure one narrow question: when a second observation
describes the same fact as an existing active memory, does the engine
consolidate it, revise it as a temporal update, or preserve it as a distinct
fact -- and does it say which?

The scenarios are synthetic and hand-authored.  Every scenario declares its
expected final action, active/total slot counts, consolidation class and, for
near-duplicate cases, which field carries the surface difference.  A scenario
only passes when every declared expectation holds; suppressing a memory is
never rewarded on its own.

Groups
------
``genuine_duplicate``
    Unambiguous restatements (exact, case/space noise, alias expansion,
    repeated evidence, predicate synonym).  Expected: consolidate.
``ambiguous_restatement``
    A single-character difference in the subject, predicate or object.  The
    same surface shape is produced by distinct entities and by genuine
    relation/temporal revisions, so these must be preserved and reported as
    unresolved near-duplicates.  See the module note in
    ``minemos_engine.black_hole_memory._find_near_duplicate_slot``.
``distinct_fact``
    Cases that must never be merged: near-miss names, near-miss values,
    similar wording, different relations and different scopes.
``temporal_update``
    Legitimate revisions, corrections, negations and historical statements.
    Expected: close the prior interval and keep one active state.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional

from minemos_engine.black_hole_memory import BlackHoleMemory, Candidate

GROUPS = (
    "genuine_duplicate",
    "ambiguous_restatement",
    "distinct_fact",
    "temporal_update",
)

BASE_TS = 5_000_000.0


def _obs(obs_id: str, subject: str, predicate: str, obj: str,
         timestamp: float) -> Dict[str, Any]:
    return {
        "id": obs_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "timestamp": timestamp,
        "text": f"{subject} {predicate} {obj}.",
    }


def _scenario(scenario_id: str, group: str, subcategory: str,
              observations: List[Dict[str, Any]],
              expected_action: str, expected_active: int,
              expected_total: int, expected_class: str,
              near_field: Optional[str] = None,
              distinct_facts: Optional[List[Dict[str, str]]] = None,
              expected_update_count: Optional[int] = None,
              historical_fact: Optional[Dict[str, str]] = None,
              query: str = "") -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "group": group,
        "subcategory": subcategory,
        "observations": observations,
        "query": query or observations[0]["text"],
        "expected": {
            "final_action": expected_action,
            "active_count": expected_active,
            "total_count": expected_total,
            "consolidation_class": expected_class,
            "near_field": near_field,
            "distinct_facts": distinct_facts or [],
            "update_count": expected_update_count,
            "historical_fact": historical_fact,
        },
    }


def generate_duplicate_scenarios() -> List[Dict[str, Any]]:
    """The deterministic benchmark.  No RNG, no private data."""
    scenarios: List[Dict[str, Any]] = []
    t = BASE_TS

    # ── Genuine duplicates: unambiguous restatements ───────────────────
    scenarios.append(_scenario(
        "D001", "genuine_duplicate", "exact_restatement",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "lives in", "Lisbon", t + 10)],
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D002", "genuine_duplicate", "case_space_punctuation",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "  mira   solano ", "Lives In", "lisbon.", t + 10)],
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D003", "genuine_duplicate", "unambiguous_abbreviation",
        [_obs("a", "Mira Solano", "lives in", "New York City", t),
         _obs("b", "Mira Solano", "lives in", "NYC", t + 10)],
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D004", "genuine_duplicate", "repeated_evidence",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "lives in", "Lisbon", t + 10),
         _obs("c", "Mira Solano", "lives in", "Lisbon", t + 20)],
        "consolidate", 1, 1, "duplicate", expected_update_count=3))
    scenarios.append(_scenario(
        "D005", "genuine_duplicate", "exclusive_synonym",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "resides in", "Lisbon", t + 10)],
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D006", "genuine_duplicate", "state_neutral_adverb",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "currently lives in", "Lisbon", t + 10)],
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D007", "genuine_duplicate", "different_structure",
        [_obs("a", "Priya Thorne", "volunteers at", "Northwind Labs", t),
         _obs("b", "Northwind Labs", "is where",
              "Priya Thorne volunteers", t + 10)],
        # Known limitation: role-reordered restatements are not recognised.
        # The engine currently preserves both as a distinct fact.
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D008", "genuine_duplicate", "additional_word_detail",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "lives in", "Lisbon, Portugal", t + 10)],
        # Known limitation: an added detail word under an exclusive
        # attribute is treated as a state revision.
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D009", "genuine_duplicate", "missing_article",
        [_obs("a", "Priya Thorne", "plays", "the mandolin", t),
         _obs("b", "Priya Thorne", "plays", "mandolin", t + 10)],
        # Known limitation: a dropped article is preserved as ambiguous.
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D010", "genuine_duplicate", "informal_verb_form",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "is living in", "Lisbon", t + 10)],
        "consolidate", 1, 1, "duplicate", expected_update_count=2))
    scenarios.append(_scenario(
        "D011", "genuine_duplicate", "repeated_noisy_restatement",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "lives in", "Lisbon", t + 10),
         _obs("c", "Mira Solnao", "lives in", "Lisbon", t + 20)],
        # Known limitation: repeated noisy evidence is preserved as an
        # unresolved near-duplicate instead of counting toward the fact.
        "consolidate", 1, 1, "duplicate", expected_update_count=3))

    # ── Ambiguous restatements: preserve + report ──────────────────────
    scenarios.append(_scenario(
        "D101", "ambiguous_restatement", "subject_transposition",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solnao", "lives in", "Lisbon", t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="subject",
        distinct_facts=[
            {"subject": "Mira Solano", "predicate": "lives in",
             "object": "Lisbon"},
            {"subject": "Mira Solnao", "predicate": "lives in",
             "object": "Lisbon"}]))
    scenarios.append(_scenario(
        "D102", "ambiguous_restatement", "subject_substitution",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Soleno", "lives in", "Lisbon", t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="subject",
        distinct_facts=[
            {"subject": "Mira Solano", "predicate": "lives in",
             "object": "Lisbon"},
            {"subject": "Mira Soleno", "predicate": "lives in",
             "object": "Lisbon"}]))
    scenarios.append(_scenario(
        "D103", "ambiguous_restatement", "object_substitution",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "lives in", "Lisbom", t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="object",
        distinct_facts=[
            {"subject": "Mira Solano", "predicate": "lives in",
             "object": "Lisbon"},
            {"subject": "Mira Solano", "predicate": "lives in",
             "object": "Lisbom"}]))
    scenarios.append(_scenario(
        "D104", "ambiguous_restatement", "object_deletion",
        [_obs("a", "Mira Solano", "drinks", "cranberry juice", t),
         _obs("b", "Mira Solano", "drinks", "cranberry juce", t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="object",
        distinct_facts=[
            {"subject": "Mira Solano", "predicate": "drinks",
             "object": "cranberry juice"},
            {"subject": "Mira Solano", "predicate": "drinks",
             "object": "cranberry juce"}]))
    scenarios.append(_scenario(
        "D105", "ambiguous_restatement", "predicate_substitution",
        [_obs("a", "Mira Solano", "drinks", "cranberry juice", t),
         _obs("b", "Mira Solano", "dranks", "cranberry juice", t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="predicate",
        distinct_facts=[
            {"subject": "Mira Solano", "predicate": "drinks",
             "object": "cranberry juice"},
            {"subject": "Mira Solano", "predicate": "dranks",
             "object": "cranberry juice"}]))
    scenarios.append(_scenario(
        "D106", "ambiguous_restatement", "predicate_tense",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "lived in", "Lisbon", t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="predicate",
        distinct_facts=[
            {"subject": "Mira Solano", "predicate": "lives in",
             "object": "Lisbon"},
            {"subject": "Mira Solano", "predicate": "lived in",
             "object": "Lisbon"}]))
    scenarios.append(_scenario(
        "D107", "ambiguous_restatement", "significance_dropping_word",
        [_obs("a", "Mira Solano", "drinks", "cranberry juice", t),
         _obs("b", "Mira Solano", "drinks", "juice", t + 10)],
        # Dropping a content word changes the fact's granularity; the
        # engine must preserve both rather than guess.
        "insert", 2, 2, "unresolved_near_duplicate", near_field="object",
        distinct_facts=[
            {"subject": "Mira Solano", "predicate": "drinks",
             "object": "cranberry juice"},
            {"subject": "Mira Solano", "predicate": "drinks",
             "object": "juice"}]))

    # ── Must remain distinct ───────────────────────────────────────────
    scenarios.append(_scenario(
        "D201", "distinct_fact", "near_miss_entity_transposition",
        [_obs("a", "Mira Ashford", "lives in", "Toronto", t),
         _obs("b", "Mira Ahsford", "lives in", "Toronto", t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="subject",
        distinct_facts=[
            {"subject": "Mira Ashford", "predicate": "lives in",
             "object": "Toronto"},
            {"subject": "Mira Ahsford", "predicate": "lives in",
             "object": "Toronto"}]))
    scenarios.append(_scenario(
        "D202", "distinct_fact", "near_miss_entity_substitution",
        [_obs("a", "Rowan Bellamy", "works at", "Foxglove Publishing", t),
         _obs("b", "Rowan Bellany", "works at", "Foxglove Publishing",
              t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="subject",
        distinct_facts=[
            {"subject": "Rowan Bellamy", "predicate": "works at",
             "object": "Foxglove Publishing"},
            {"subject": "Rowan Bellany", "predicate": "works at",
             "object": "Foxglove Publishing"}]))
    scenarios.append(_scenario(
        "D203", "distinct_fact", "similar_wording_object_deletion",
        [_obs("a", "Priya Thorne", "volunteers at", "Northwind Labs", t),
         _obs("b", "Priya Thorne", "volunteers at", "Northwind Lab",
              t + 10)],
        "insert", 2, 2, "unresolved_near_duplicate", near_field="object",
        distinct_facts=[
            {"subject": "Priya Thorne", "predicate": "volunteers at",
             "object": "Northwind Labs"},
            {"subject": "Priya Thorne", "predicate": "volunteers at",
             "object": "Northwind Lab"}]))
    scenarios.append(_scenario(
        "D204", "distinct_fact", "similar_wording_distinct",
        [_obs("a", "Priya Thorne", "loves", "the Roman Republic", t),
         _obs("b", "Priya Thorne", "loves", "the Roman Empire", t + 10)],
        "insert", 2, 2, "distinct_fact",
        distinct_facts=[
            {"subject": "Priya Thorne", "predicate": "loves",
             "object": "the Roman Republic"},
            {"subject": "Priya Thorne", "predicate": "loves",
             "object": "the Roman Empire"}]))
    scenarios.append(_scenario(
        "D205", "distinct_fact", "different_relation_same_object",
        [_obs("a", "Priya Thorne", "plays", "the mandolin", t),
         _obs("b", "Priya Thorne", "owns", "the mandolin", t + 10)],
        "insert", 2, 2, "distinct_fact",
        distinct_facts=[
            {"subject": "Priya Thorne", "predicate": "plays",
             "object": "the mandolin"},
            {"subject": "Priya Thorne", "predicate": "owns",
             "object": "the mandolin"}]))
    scenarios.append(_scenario(
        "D206", "distinct_fact", "conversation_scope",
        [_obs("a", "conversation:alpha", "knows", "Mira Solano", t),
         _obs("b", "conversation:beta", "knows", "Mira Solano", t + 10)],
        "insert", 2, 2, "distinct_fact",
        distinct_facts=[
            {"subject": "conversation:alpha", "predicate": "knows",
             "object": "Mira Solano"},
            {"subject": "conversation:beta", "predicate": "knows",
             "object": "Mira Solano"}]))
    scenarios.append(_scenario(
        "D207", "distinct_fact", "document_scope",
        [_obs("a", "document:alpha", "mentions", "Mira Solano", t),
         _obs("b", "document:beta", "mentions", "Mira Solano", t + 10)],
        "insert", 2, 2, "distinct_fact",
        distinct_facts=[
            {"subject": "document:alpha", "predicate": "mentions",
             "object": "Mira Solano"},
            {"subject": "document:beta", "predicate": "mentions",
             "object": "Mira Solano"}]))

    # ── Temporal updates ───────────────────────────────────────────────
    scenarios.append(_scenario(
        "D301", "temporal_update", "exclusive_revision",
        [_obs("a", "Mira Solano", "lives in", "Toronto", t),
         _obs("b", "Mira Solano", "lives in", "Lisbon", t + 100)],
        "temporal_update", 1, 2, "temporal_update",
        historical_fact={"subject": "Mira Solano", "predicate": "lives in",
                         "object": "Toronto"}))
    scenarios.append(_scenario(
        "D302", "temporal_update", "numeric_correction",
        [_obs("a", "Mira Solano", "salary is", "90000", t),
         _obs("b", "Mira Solano", "salary is", "95000", t + 100)],
        "temporal_update", 1, 2, "temporal_update",
        historical_fact={"subject": "Mira Solano", "predicate": "salary is",
                         "object": "90000"}))
    scenarios.append(_scenario(
        "D303", "temporal_update", "negation_change",
        [_obs("a", "Mira Solano", "is allergic to", "walnuts", t),
         _obs("b", "Mira Solano", "is not allergic to", "walnuts",
              t + 100)],
        "temporal_update", 1, 2, "temporal_update",
        historical_fact={"subject": "Mira Solano",
                         "predicate": "is allergic to",
                         "object": "walnuts"}))
    scenarios.append(_scenario(
        "D304", "temporal_update", "historical_statement",
        [_obs("a", "Mira Solano", "lives in", "Lisbon", t),
         _obs("b", "Mira Solano", "used to live in", "Toronto", t + 100)],
        "historical_observation", 1, 2, "historical_observation",
        historical_fact={"subject": "Mira Solano",
                         "predicate": "used to live in",
                         "object": "Toronto"}))
    scenarios.append(_scenario(
        "D305", "temporal_update", "ownership_transition",
        [_obs("a", "Mira Solano", "owns the majority share in",
              "Apex Robotics", t),
         _obs("b", "Mira Solano", "sold all equity in",
              "Apex Robotics", t + 100)],
        "temporal_update", 1, 2, "temporal_update",
        historical_fact={"subject": "Mira Solano",
                         "predicate": "owns the majority share in",
                         "object": "Apex Robotics"}))
    scenarios.append(_scenario(
        "D306", "temporal_update", "explicit_now_marker",
        [_obs("a", "Mira Solano", "works at", "Foxglove Publishing", t),
         _obs("b", "Mira Solano", "now works at", "Meridian Robotics",
              t + 100)],
        "temporal_update", 1, 2, "temporal_update",
        historical_fact={"subject": "Mira Solano", "predicate": "works at",
                         "object": "Foxglove Publishing"}))
    scenarios.append(_scenario(
        "D307", "temporal_update", "retraction",
        [_obs("a", "Mira Solano", "reads", "fantasy", t),
         _obs("b", "Mira Solano", "does not read", "fantasy", t + 100)],
        "temporal_update", 1, 2, "temporal_update",
        historical_fact={"subject": "Mira Solano", "predicate": "reads",
                         "object": "fantasy"}))

    return scenarios


def _norm(value: Optional[str]) -> str:
    return BlackHoleMemory._normalize_text(value)


def _slot_facts(memory: BlackHoleMemory) -> List[Dict[str, str]]:
    facts = []
    for slot in memory.memory.slots:
        facts.append({
            "subject": _norm(slot.debug_subject),
            "predicate": _norm(slot.latest_predicate or slot.debug_predicate),
            "object": _norm(slot.latest_object
                            or memory.slot_texts.get(id(slot), slot.text)),
            "active": bool(slot.is_active),
            "update_count": int(slot.update_count),
            "valid_from": slot.valid_from,
            "valid_until": slot.valid_until,
        })
    return facts


def _fact_present(facts: Iterable[Dict[str, Any]], fact: Dict[str, str],
                  active: bool = True) -> bool:
    wanted = {
        "subject": _norm(fact.get("subject")),
        "predicate": _norm(fact.get("predicate")),
        "object": _norm(fact.get("object")),
    }
    for item in facts:
        if active and not item["active"]:
            continue
        if all(item[key] == wanted[key]
               for key in ("subject", "predicate", "object")):
            return True
    return False


def run_duplicate_benchmark(
    components: Optional[Dict[str, Any]] = None,
    stub: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    """Run every scenario and return records plus aggregate metrics."""
    if components is None:
        from tests.support.nl_audit.harness import build_components
        components = build_components(stub=stub)
    records: List[Dict[str, Any]] = []
    for scenario in generate_duplicate_scenarios():
        memory = BlackHoleMemory(
            capacity=20,
            dim=4096,
            dense_encoder=components["encoder"],
            nli_gate=components["gate"],
            verifier=components["verifier"],
            retrieval_backend="canonical",
            strict_entity_identity=strict,
        )
        results = []
        for observation in scenario["observations"]:
            result = memory.absorb(
                Candidate(observation["subject"], observation["predicate"],
                          observation["object"],
                          timestamp=float(observation["timestamp"])),
                timestamp=float(observation["timestamp"]),
            )
            results.append({
                "action": result.get("action"),
                "consolidation_class": result.get("consolidation_class"),
                "near_duplicate_of": result.get("near_duplicate_of"),
                "near_duplicate_field": result.get("near_duplicate_field"),
                "near_duplicate_edit": result.get("near_duplicate_edit"),
                "update_count": result.get("update_count"),
            })
        facts = _slot_facts(memory)
        expected = scenario["expected"]
        active = [item for item in facts if item["active"]]
        final = results[-1]
        checks = {
            "action_ok": final["action"] == expected["final_action"],
            "active_count_ok": len(active) == expected["active_count"],
            "total_count_ok": len(facts) == expected["total_count"],
            "classification_ok": (
                final["consolidation_class"]
                == expected["consolidation_class"]),
            "near_field_ok": (
                final["near_duplicate_field"] == expected["near_field"]),
            "facts_present_ok": all(
                _fact_present(facts, fact)
                for fact in expected["distinct_facts"]),
            "historical_fact_ok": (
                expected["historical_fact"] is None
                or _fact_present(facts, expected["historical_fact"],
                                 active=False)),
            "update_count_ok": (
                expected["update_count"] is None
                or final["update_count"] == expected["update_count"]),
        }
        records.append({
            "record_type": "duplicate_lifecycle",
            "scenario_id": scenario["scenario_id"],
            "group": scenario["group"],
            "subcategory": scenario["subcategory"],
            "expected": expected,
            "results": results,
            "active_count": len(active),
            "total_count": len(facts),
            "duplicate": (
                expected["active_count"] == 1 and len(active) > 1),
            "false_merge": (
                expected["active_count"] > 1
                and len(active) < expected["active_count"]),
            "checks": checks,
            "pass": all(checks.values()),
            "facts": facts,
        })
        memory.close()

    summary: Dict[str, Any] = {"groups": {}, "overall": {}}
    for group in GROUPS:
        members = [r for r in records if r["group"] == group]
        if not members:
            continue
        summary["groups"][group] = {
            "scenarios": len(members),
            "passed": sum(1 for r in members if r["pass"]),
            "action_accuracy": _rate(members, "action_ok"),
            "active_count_accuracy": _rate(members, "active_count_ok"),
            "classification_accuracy": _rate(members, "classification_ok"),
            "nearduplicate_field_accuracy": _rate(members, "near_field_ok"),
            "duplicate_rate": _rate(members, "duplicate"),
            "false_merge_rate": _rate(members, "false_merge"),
        }
    summary["overall"] = {
        "scenarios": len(records),
        "passed": sum(1 for r in records if r["pass"]),
        "action_accuracy": _rate(records, "action_ok"),
        "active_count_accuracy": _rate(records, "active_count_ok"),
        "classification_accuracy": _rate(records, "classification_ok"),
        "nearduplicate_field_accuracy": _rate(records, "near_field_ok"),
        "duplicate_rate": _rate(records, "duplicate"),
        "false_merge_rate": _rate(records, "false_merge"),
        "environment": {
            "stub": bool(stub),
            "strict_entity_identity": bool(strict),
            "dense_model": components.get("dense_model"),
            "nli_model": components.get("nli_model"),
        },
    }
    return {"summary": summary, "records": records}


def _rate(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [
        record["checks"][key] if key in record["checks"] else record.get(key)
        for record in records
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the duplicate-consolidation benchmark.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    result = run_duplicate_benchmark(
        stub=args.stub, strict=not args.no_strict)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
