"""R5 behavioral-integrity tests for F1, F2 and F4.

F1: a predicate containing "is" is not automatically a state attribute, and a
state conflict is never resolved by semantic similarity alone.  Copular and
role-state forms that assert one current value expose an unresolved conflict
while every competing observation and its interval remain retrievable.

F2: an ownership change is recognized only from positive evidence: the later
statement must concern the same subject and the same asset, after recipient
phrases and quantity words are removed, and must not be negated or partial.
Recipient names and similar organisation names are never identity proof.  Any
transition that cannot be proven leaves both observations active and reports
an unresolved correction with its transition kind.

F4: a whole-statement SAME verdict never authorizes destroying a represented
value the lexical gate proves incompatible.  Equivalent paraphrases still
consolidate in both strict and non-strict modes and in both ingestion orders.

The deterministic engine tests use a high-similarity encoder so the dense
gate cannot hide a verification-path defect: every statement is a candidate
for consolidation.  A separate real-local-model test confirms the same
behavior with the installed MiniLM encoder and NLI gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from minemos_engine.black_hole_memory import (
    UNRESOLVED_STATE_CONFLICT,
    UNRESOLVED_TEMPORAL_CORRECTION,
)
from tests._retrieval_helpers import RuleVerifier, make_memory, unit_vector

HIGH_SIM = unit_vector(20261108)


class HighSimEncoder:
    """Every statement maps to one vector so the dense gate always passes."""

    def encode(self, text: str) -> np.ndarray:
        return HIGH_SIM.copy()


def _engine(verifier=None, strict: bool = False, encoder=None):
    return make_memory(
        capacity=32,
        verifier=verifier or RuleVerifier(),
        encoder=encoder or HighSimEncoder(),
        strict_entity_identity=strict,
    )


def _absorb(mem, subject, predicate, obj, timestamp):
    return mem.absorb((subject, predicate, obj),
                      timestamp=timestamp, confidence=1.0)


def _active_texts(mem):
    return [
        mem.slot_texts.get(id(slot), slot.text or "")
        for slot in mem.memory.slots if slot.is_active
    ]


def _corrections(mem):
    return mem.unresolved_temporal_corrections()


def _correction_kinds(mem):
    return [
        (finding["reason"], finding.get("transition_kind"))
        for finding in _corrections(mem)
    ]


# ── F1: state semantics of "is" forms ─────────────────────────────────────


def test_state_forms_are_detected_individually_not_by_the_word_is():
    mem = _engine()
    # Copulas, attribute-copulas and state markers are time-indexed states.
    for predicate in ("is", "phase is", "build status"):
        assert mem._is_state_predicate(predicate) is True, predicate
    # Role verbs are state-attribute forms but are not copular predicates;
    # conflating the two would make "works as" and "is" one attribute.
    assert mem._is_state_predicate("works as") is False
    # Unrelated or merely "is"-containing words are not state predicates.
    for predicate in ("uses", "advises", "analysis", "plays"):
        assert mem._is_state_predicate(predicate) is False, predicate

    assert mem._state_attribute_key("is") == "__copula__"
    assert mem._state_attribute_key("goal is") == "goal"
    assert mem._state_attribute_key("website is") == "website"
    assert mem._state_attribute_key("works as") == "role"
    assert mem._state_attribute_key("serves as") == "role"
    assert mem._state_attribute_key("review stage") == "review stage"
    # "status" already has an explicit exclusive policy, so it is reported by
    # the exclusive-attribute detector, never reclassified as a state form.
    assert mem.classify_predicate("build status") == "exclusive"
    assert mem._state_attribute_key("build status") is None
    assert mem._state_attribute_key("uses") is None
    assert mem._state_attribute_key("goal is") != mem._state_attribute_key(
        "website is")


def test_conflicting_copula_values_are_preserved_and_exposed():
    mem = _engine()
    _absorb(mem, "Alice", "is", "tired", 1.0)
    _absorb(mem, "Alice", "is", "awake", 2.0)

    assert set(_active_texts(mem)) == {"Alice is tired.", "Alice is awake."}
    outcome = mem.eligible_query("What is Alice?")
    assert outcome["status"] == "answered"
    assert len(outcome["rows"]) == 2
    temporal = outcome["temporal"]
    assert temporal["conflict_status"] == "unresolved"
    assert temporal["answer_is_settled"] is False
    finding = temporal["temporal_conflicts"][0]
    assert finding["reason"] == UNRESOLVED_STATE_CONFLICT
    assert finding["confidence_scope"] == "state_predicate_form_only"
    assert finding["state_attribute"] == "__copula__"

    # Both competing observations keep their own temporal evidence.  Neither
    # observation supersedes the other, so both intervals stay open: at 2.5
    # the two values are still competing and are both reported.
    assert [row["text"] for row in mem.recall("Alice", as_of=1.5)] == [
        "Alice is tired."]
    assert {row["text"] for row in mem.recall("Alice", as_of=2.5)} == {
        "Alice is tired.", "Alice is awake."}


def test_semantic_same_never_supersedes_incompatible_state_values():
    mem = _engine(verifier=RuleVerifier(default="SAME"))
    first = _absorb(mem, "Saturn project", "phase is", "design", 1.0)
    second = _absorb(mem, "Saturn project", "phase is", "build", 2.0)

    assert (first["action"], second["action"]) == ("insert", "insert")
    assert set(_active_texts(mem)) == {
        "Saturn project phase is design.",
        "Saturn project phase is build.",
    }
    assert mem.total_incompatible_value_separations >= 1
    outcome = mem.eligible_query("What is Saturn project's phase?")
    assert outcome["temporal"]["answer_is_settled"] is False
    assert outcome["temporal"]["temporal_conflicts"][0]["state_attribute"] == (
        "phase")


def test_different_state_attributes_do_not_conflict():
    mem = _engine(verifier=RuleVerifier(default="SAME"))
    _absorb(mem, "Saturn project", "goal is", "ship by June", 1.0)
    _absorb(mem, "Saturn project", "website is", "example.com", 2.0)

    assert len(_active_texts(mem)) == 2
    assert mem.exclusive_temporal_conflicts() == []
    outcome = mem.eligible_query("What is Saturn project's goal?")
    assert outcome["temporal"]["answer_is_settled"] is True
    assert outcome["temporal"]["temporal_conflicts"] == []


def test_role_state_predicates_share_one_state_attribute():
    mem = _engine()
    _absorb(mem, "Saturn project", "works as", "pilot", 1.0)
    _absorb(mem, "Saturn project", "serves as", "prototype", 2.0)

    assert len(_active_texts(mem)) == 2
    outcome = mem.eligible_query("What does the Saturn project serve as?")
    temporal = outcome["temporal"]
    assert temporal["answer_is_settled"] is False
    finding = temporal["temporal_conflicts"][0]
    assert finding["reason"] == UNRESOLVED_STATE_CONFLICT
    assert finding["state_attribute"] == "role"


def test_identical_state_value_still_consolidates():
    mem = _engine(verifier=RuleVerifier(default="SAME"))
    _absorb(mem, "Alice", "is", "tired", 1.0)
    second = _absorb(mem, "Alice", "is", "tired", 2.0)

    assert second["action"] == "consolidate"
    assert _active_texts(mem) == ["Alice is tired."]
    assert mem.memory.slots[0].update_count == 2


# ── F2: ownership transitions and identity ────────────────────────────────


def test_full_disposal_of_same_entity_corrects_ownership():
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "owns the majority share in", "Acme", 100.0)
    result = _absorb(
        mem, "Ada", "sold all equity in", "Acme to Bob", 150.0)

    assert result["action"] == "temporal_update"
    assert _active_texts(mem) == ["Ada sold all equity in Acme to Bob."]
    assert [row["text"] for row in mem.recall("Ada", as_of=120.0)] == [
        "Ada owns the majority share in Acme."]
    assert [row["text"] for row in mem.get_history("Ada")] == [
        "Ada owns the majority share in Acme.",
        "Ada sold all equity in Acme to Bob.",
    ]
    assert mem.temporal_history_violations() == []
    assert _corrections(mem) == []


def test_recipient_is_not_a_second_entity():
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "owns", "Acme", 100.0)
    _absorb(mem, "Ada", "sold all equity in", "Acme to Bob", 150.0)

    # Bob is the recipient of the disposal, never the subject of a fact.
    assert mem.recall("Bob") == []
    assert [row["text"] for row in mem.recall("Ada")] == [
        "Ada sold all equity in Acme to Bob."]


def test_recipient_naming_the_owned_entity_is_not_identity_proof():
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "owns", "Acme", 100.0)
    result = _absorb(
        mem, "Ada", "sold all equity in", "Globex to Acme", 150.0)

    assert result["action"] == "insert"
    assert set(_active_texts(mem)) == {
        "Ada owns Acme.",
        "Ada sold all equity in Globex to Acme.",
    }
    assert _corrections(mem) == []


@pytest.mark.parametrize("old_object,new_object", [
    ("Acme", "Acme North"),
    ("Acme Inc", "Acme Ltd"),
])
def test_similar_name_or_legal_form_is_not_identity_proof(
        old_object, new_object):
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "owns", old_object, 100.0)
    result = _absorb(mem, "Ada", "sold all equity in", new_object, 150.0)

    assert result["action"] == "insert"
    assert len(_active_texts(mem)) == 2
    assert _corrections(mem) == []


def test_unspecified_transfer_preserves_both_and_exposes_uncertainty():
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "owns", "Acme", 100.0)
    result = _absorb(mem, "Ada", "transferred", "Acme to Bob", 150.0)

    assert result["action"] == "insert"
    assert set(_active_texts(mem)) == {
        "Ada owns Acme.",
        "Ada transferred Acme to Bob.",
    }
    findings = _corrections(mem)
    assert len(findings) == 1
    assert findings[0]["reason"] == UNRESOLVED_TEMPORAL_CORRECTION
    assert findings[0]["transition_kind"] == "unspecified"
    outcome = mem.eligible_query("What does Ada own?")
    assert outcome["temporal"]["answer_is_settled"] is False
    assert len(outcome["rows"]) == 2


def test_partial_disposal_keeps_ownership():
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "owns", "Acme", 100.0)
    result = _absorb(
        mem, "Ada", "sold some shares of", "Acme", 150.0)

    assert result["action"] == "insert"
    assert set(_active_texts(mem)) == {
        "Ada owns Acme.",
        "Ada sold some shares of Acme.",
    }
    assert _corrections(mem) == []


def test_negated_disposal_does_not_supersede_ownership():
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "owns", "Acme", 100.0)
    result = _absorb(mem, "Ada", "did not sell", "Acme", 150.0)

    # "Did not sell Acme" negates a change, not the ownership state.
    assert result["action"] == "insert"
    assert set(_active_texts(mem)) == {
        "Ada owns Acme.",
        "Ada did not sell Acme.",
    }
    assert _corrections(mem) == []

    # A negated ownership predicate is still a correction of the state.
    other = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(other, "Ada", "owns", "Acme", 100.0)
    denied = _absorb(other, "Ada", "does not own", "Acme", 150.0)
    assert denied["action"] == "temporal_update"
    assert _active_texts(other) == ["Ada does not own Acme."]


def test_reverse_order_reacquisition_is_not_a_correction():
    mem = _engine(verifier=RuleVerifier(default="CONTRADICTION"))
    _absorb(mem, "Ada", "sold all equity in", "Acme", 100.0)
    result = _absorb(mem, "Ada", "owns", "Acme", 150.0)

    assert result["action"] == "insert"
    assert set(_active_texts(mem)) == {
        "Ada sold all equity in Acme.",
        "Ada owns Acme.",
    }
    assert _corrections(mem) == []
    assert [row["text"] for row in mem.query("Acme", as_of=120.0)] == [
        "Ada sold all equity in Acme."]
    assert set(row["text"] for row in mem.query("Acme", as_of=160.0)) == {
        "Ada sold all equity in Acme.",
        "Ada owns Acme.",
    }


def test_incremental_ingestion_and_checkpoint_keep_ownership_history(
        tmp_path):
    verifier = RuleVerifier(default="CONTRADICTION")
    mem = _engine(verifier=verifier)
    _absorb(mem, "Ada", "owns the majority share in", "Acme", 100.0)
    assert _active_texts(mem) == ["Ada owns the majority share in Acme."]

    _absorb(mem, "Ada", "sold all equity in", "Acme to Bob", 150.0)
    assert _active_texts(mem) == ["Ada sold all equity in Acme to Bob."]

    path = tmp_path / "r5-ownership.npz"
    mem.save_checkpoint(str(path))

    restored = _engine(verifier=verifier)
    restored.load_checkpoint(str(path))
    assert _active_texts(restored) == [
        "Ada sold all equity in Acme to Bob."]
    assert [row["text"] for row in restored.recall("Ada", as_of=120.0)] == [
        "Ada owns the majority share in Acme."]
    assert [row["text"] for row in restored.get_history("Ada")] == [
        "Ada owns the majority share in Acme.",
        "Ada sold all equity in Acme to Bob.",
    ]
    assert restored.temporal_history_violations() == []
    assert _corrections(restored) == []


# ── F4: value compatibility and destructive consolidation ─────────────────


@pytest.mark.parametrize("order", [
    ("guitar", "piano"),
    ("piano", "guitar"),
])
def test_semantic_same_does_not_merge_incompatible_values(order):
    mem = _engine(verifier=RuleVerifier(default="SAME"))
    _absorb(mem, "Ada", "plays", order[0], 1.0)
    second = _absorb(mem, "Ada", "performs", order[1], 2.0)

    assert second["action"] == "insert"
    assert len(_active_texts(mem)) == 2
    assert mem.total_incompatible_value_separations >= 1
    # The whole-statement SAME was rejected lexically before any verifier
    # value call, in non-strict mode too.
    assert mem.total_value_verifications == 0


def test_strict_mode_counts_incompatible_value_separation():
    mem = _engine(
        verifier=RuleVerifier(default="SAME"), strict=True)
    _absorb(mem, "Ada", "plays", "guitar", 1.0)
    _absorb(mem, "Ada", "plays", "piano", 2.0)

    assert set(_active_texts(mem)) == {
        "Ada plays guitar.",
        "Ada plays piano.",
    }
    # Same attribute, provably different values: the SAME verdict never
    # merges them, and strict mode records that it refused the merge.
    assert mem.total_incompatible_value_separations >= 1
    assert mem.total_suppressed_value_merges >= 1


@pytest.mark.parametrize("strict", [False, True])
def test_equivalent_paraphrase_still_consolidates(strict):
    mem = _engine(
        verifier=RuleVerifier(default="SAME"), strict=strict)
    _absorb(mem, "Ada", "plays", "guitar", 1.0)
    second = _absorb(mem, "Ada", "performs", "guitar", 2.0)

    assert second["action"] == "consolidate"
    assert _active_texts(mem) == ["Ada performs guitar."]
    assert mem.memory.slots[0].update_count == 2


@pytest.mark.parametrize("predicate_a,predicate_b,value_a,value_b", [
    ("role", "role",
     "leads cybersecurity operations", "directs cybersecurity operations"),
    ("directed the regional", "conducted the local",
     "orchestral ensemble", "symphony orchestra"),
])
def test_value_paraphrase_still_consolidates_under_verifier_same(
        predicate_a, predicate_b, value_a, value_b):
    """The compatibility guard separates provably different values only.

    Two values with no lexical evidence of difference are still a paraphrase
    when the verifier returns SAME in both directions; strict mode keeps its
    separate, conservative refusal of every unequal value.
    """
    mem = _engine(verifier=RuleVerifier(default="SAME"))
    _absorb(mem, "Ada", predicate_a, value_a, 1.0)
    second = _absorb(mem, "Ada", predicate_b, value_b, 2.0)

    assert second["action"] == "consolidate"
    assert _active_texts(mem) == [f"Ada {predicate_b} {value_b}."]
    assert mem.memory.slots[0].update_count == 2
    assert mem.total_incompatible_value_separations == 0


def test_lexical_value_compatibility_gate():
    mem = _engine()
    compatible = [
        ("guitar", "guitar"),
        ("Paris", "Paris, France"),
        ("biography", "biographical"),
        ("Acme", "the Acme Corporation"),
        ("Acme", "Acmee"),
    ]
    incompatible = [
        ("guitar", "piano"),
        ("the quantum team", "the data team"),
        ("Acme Inc", "Acme Ltd"),
        ("the Roman Empire", "the Roman Republic"),
    ]
    for value_a, value_b in compatible:
        assert mem._values_compatible_for_merge(value_a, value_b) is True
    for value_a, value_b in incompatible:
        assert mem._values_compatible_for_merge(value_a, value_b) is False


# ── Real local models (already installed; nothing is downloaded) ──────────


def test_real_local_models_keep_incompatible_values_and_merge_paraphrase():
    from minemos_engine.intelligence.nli import NliSemanticGate
    from minemos_engine.intelligence.dense import (
        DenseSemanticEncoder,
        LEXICAL_ENCODER,
    )
    from minemos_engine.black_hole_memory import BlackHoleMemory

    encoder = DenseSemanticEncoder()
    gate = NliSemanticGate()

    def real_engine(strict: bool):
        return BlackHoleMemory(
            capacity=32,
            dim=4096,
            dense_encoder=encoder,
            nli_gate=gate,
            lex_encoder=LEXICAL_ENCODER,
            strict_entity_identity=strict,
        )

    # Equivalent paraphrase consolidates, including a reworded value.
    paraphrase = real_engine(strict=False)
    _absorb(paraphrase, "Ada", "plays", "guitar", 1.0)
    second = _absorb(paraphrase, "Ada", "performs", "guitar", 2.0)
    assert second["action"] == "consolidate"
    assert _active_texts(paraphrase) == ["Ada performs guitar."]

    value_paraphrase = real_engine(strict=False)
    _absorb(value_paraphrase, "David", "role",
            "leads cybersecurity operations", 1.0)
    second = _absorb(value_paraphrase, "David", "role",
                     "directs cybersecurity operations", 2.0)
    assert second["action"] == "consolidate"
    assert _active_texts(value_paraphrase) == [
        "David role directs cybersecurity operations."]

    # A whole-statement SAME must not erase a distinct instrument.
    incompatible = real_engine(strict=True)
    _absorb(incompatible, "Ada", "plays", "guitar", 1.0)
    second = _absorb(incompatible, "Ada", "performs", "piano", 2.0)
    assert second["action"] == "insert"
    assert set(_active_texts(incompatible)) == {
        "Ada plays guitar.",
        "Ada performs piano.",
    }

    # Conflicting copular state values are preserved and exposed.
    state = real_engine(strict=False)
    _absorb(state, "Saturn project", "phase is", "design", 1.0)
    _absorb(state, "Saturn project", "phase is", "build", 2.0)
    assert len(_active_texts(state)) == 2
    outcome = state.eligible_query("What is Saturn project's phase?")
    assert outcome["temporal"]["answer_is_settled"] is False
