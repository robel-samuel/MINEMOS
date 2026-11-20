"""Adversarial identity acceptance set (V4).

Every case is synthetic.  The set targets the two failures that would make
memory untrustworthy even when extraction is right: merging two different
people into one identity, and attaching a third-person statement to the wrong
subject (including the speaker).  Abstention is a pass: where the evidence
does not pick a person, the resolver must say so rather than guess.

The whole set is also scored as one acceptance report in
``test_acceptance_set_has_no_false_merges_or_wrong_subjects``; the per-case
test fails with the individual scenario name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest

from minemos.resolution import Resolver, SourceMessage


@dataclass(frozen=True)
class Expectation:
    message: int
    predicate: str
    value: str
    entity: str
    kind: str
    status: str = "resolved"
    ambiguity: Optional[str] = None


@dataclass(frozen=True)
class Case:
    name: str
    turns: Tuple[Tuple[str, str], ...]
    expectations: Tuple[Expectation, ...] = ()
    # Messages where no emitted fact may be attributed to the speaker.
    no_speaker_facts: Tuple[int, ...] = ()


CASES: Tuple[Case, ...] = (
    Case(
        "misspelled_relation_noun",
        (("My siter is called Ama.", "a"),),
        (Expectation(0, "name", "Ama", "sister", "derived"),),
    ),
    Case(
        "misspelled_relation_attribute",
        (("My brohter works at Acme.", "a"),),
        (Expectation(0, "works_at", "Acme", "brother", "derived"),),
    ),
    Case(
        "transposed_relation_nouns",
        (("My partmer is called Nia.", "a"),
         ("My siister lives in Kumasi.", "a")),
        (Expectation(0, "name", "Nia", "partner", "derived"),
         Expectation(1, "lives_in", "Kumasi", "sister", "derived")),
    ),
    Case(
        "capitalised_transposition_is_a_different_person",
        (("Mira Ashford lives in Bath.", "a"),
         ("Mira Ahsford lives in York.", "a")),
        (Expectation(0, "lives_in", "Bath", "mira-ashford", "named"),
         Expectation(1, "lives_in", "York", "mira-ahsford", "named")),
    ),
    Case(
        "similar_surnames_are_different_people",
        (("Clara Caldwell lives in Leeds.", "a"),
         ("Clara Cardwell lives in York.", "a")),
        (Expectation(0, "lives_in", "Leeds", "clara-caldwell", "named"),
         Expectation(1, "lives_in", "York", "clara-cardwell", "named")),
    ),
    Case(
        "two_relations_one_name_stay_distinct",
        (("My brother is called Sam.", "a"),
         ("My colleague is called Sam.", "a"),
         ("My brother lives in Oslo.", "a"),
         ("My colleague works at Acme.", "a")),
        (Expectation(0, "name", "Sam", "brother-role", "derived"),
         Expectation(1, "name", "Sam", "colleague-role", "derived"),
         Expectation(2, "lives_in", "Oslo", "brother", "named"),
         Expectation(3, "works_at", "Acme", "colleague", "named")),
    ),
    Case(
        "bare_mention_of_a_shared_name_abstains",
        (("My brother is called Sam.", "a"),
         ("My colleague is called Sam.", "a"),
         ("Sam lives in Oslo.", "a")),
        (Expectation(2, "lives_in", "Oslo", "bare-sam", "named",
                     status="ambiguous",
                     ambiguity="ambiguous_name_reference"),),
    ),
    Case(
        "pronoun_between_namesakes_abstains",
        (("My brother is called Sam.", "a"),
         ("My colleague is called Sam.", "a"),
         ("He lives in Oslo.", "a")),
        (Expectation(2, "lives_in", "Oslo", "pronoun", "unknown",
                     status="ambiguous",
                     ambiguity="pronoun_ambiguous_name"),),
    ),
    Case(
        "pronoun_between_two_female_relations_abstains",
        (("My sister lives in Accra.", "a"),
         ("My aunt lives in Cape Coast.", "a"),
         ("She works at a bank.", "a")),
        (Expectation(2, "works_at", "bank", "pronoun", "unknown",
                     status="ambiguous",
                     ambiguity="pronoun_multiple_antecedents"),),
    ),
    Case(
        "pronoun_after_one_named_relation_resolves",
        (("My brother is called Tom.", "a"),
         ("He lives in Oslo.", "a")),
        (Expectation(1, "lives_in", "Oslo", "brother", "named"),),
    ),
    Case(
        "correction_extends_the_relation_name",
        (("My brother is called Sam.", "a"),
         ("I mean Sam Waters.", "a"),
         ("He lives in Oslo.", "a")),
        (Expectation(1, "name", "Sam Waters", "brother", "named"),
         Expectation(2, "lives_in", "Oslo", "brother", "named")),
    ),
    Case(
        "correction_between_namesakes_abstains",
        (("My brother is called Sam.", "a"),
         ("My colleague is called Sam.", "a"),
         ("I mean Sam Waters.", "a")),
        (Expectation(2, "name", "Sam Waters", "correction", "named",
                     status="ambiguous",
                     ambiguity="correction_target_ambiguous"),),
    ),
    Case(
        "correction_after_an_unrelated_message_abstains",
        (("My brother is called Sam.", "a"),
         ("The weather is grey.", "a"),
         ("I mean Sam Waters.", "a")),
        (Expectation(2, "name", "Sam Waters", "correction", "unknown",
                     status="ambiguous",
                     ambiguity="correction_target_unknown"),),
    ),
    Case(
        "same_relation_from_two_speakers_is_two_people",
        (("My brother is called Sam.", "a"),
         ("My brother is called Sam.", "b"),
         ("My brother lives in Oslo.", "a"),
         ("He lives in Accra.", "a")),
        (Expectation(2, "lives_in", "Oslo", "brother-a", "named"),
         Expectation(3, "lives_in", "Accra", "pronoun", "unknown",
                     status="ambiguous",
                     ambiguity="pronoun_ambiguous_name")),
    ),
    Case(
        "speaker_name_is_not_the_namesake",
        (("My name is Sam.", "a"),
         ("My brother is called Sam.", "a"),
         ("He lives in Oslo.", "a")),
        (Expectation(0, "name", "Sam", "speaker", "speaker"),
         Expectation(2, "lives_in", "Oslo", "brother", "named")),
    ),
    Case(
        "informal_alias_is_not_silently_merged",
        (("My brother is called Sam.", "a"),
         ("Everyone calls him Sammy.", "a"),
         ("Sammy lives in Oslo.", "a")),
        (Expectation(0, "name", "Sam", "brother-role", "derived"),
         Expectation(2, "lives_in", "Oslo", "sammy", "named")),
    ),
    Case(
        "possessive_chain_never_attributed_to_the_speaker",
        (("My sister is called Ama.", "a"),
         ("My sister's husband is called Kofi.", "a"),
         ("My brother's wife lives in Accra.", "a")),
        (Expectation(0, "name", "Ama", "sister-role", "derived"),),
        no_speaker_facts=(1, 2),
    ),
)


def _case(name: str) -> Case:
    return next(case for case in CASES if case.name == name)


def _run(case: Case):
    resolver = Resolver()
    records = []
    for order, (text, speaker) in enumerate(case.turns):
        records.append(resolver.resolve(SourceMessage(
            conversation_id=f"adv-{case.name}",
            message_id=f"m{order}", text=text, speaker=speaker,
            order=order)).record)
    return resolver, records


def _matching(record, predicate: str, value: str):
    return [fact for fact in record.facts
            if fact.predicate == predicate
            and fact.value.casefold() == value.casefold()]


def _collect(case: Case):
    """Returns (records, matched, mismatches, speaker_leaks, merges)."""
    _, records = _run(case)
    matched = 0
    mismatches = []
    speaker_leaks = 0
    keys_by_entity = {}
    for expectation in case.expectations:
        found = _matching(
            records[expectation.message],
            expectation.predicate, expectation.value)
        if len(found) != 1:
            mismatches.append(
                f"{case.name}: m{expectation.message} "
                f"{expectation.predicate}={expectation.value!r} "
                f"produced {len(found)} facts")
            continue
        subject = found[0].subject
        actual = (subject.kind, subject.status, subject.ambiguity)
        wanted = (expectation.kind, expectation.status,
                  expectation.ambiguity)
        if actual != wanted:
            mismatches.append(
                f"{case.name}: m{expectation.message} expected {wanted} "
                f"got {actual}")
        else:
            matched += 1
        if expectation.kind != "speaker" and subject.kind == "speaker":
            speaker_leaks += 1
        if expectation.kind in ("named", "speaker"):
            keys_by_entity.setdefault(expectation.entity, set()).add(
                subject.entity_key or subject.label.casefold())
    merges = 0
    for entity, keys in keys_by_entity.items():
        if len(keys) > 1:
            merges += 1
            mismatches.append(f"{case.name}: {entity} split across {keys}")
    flat = [key for keys in keys_by_entity.values() for key in keys]
    if len(flat) != len(set(flat)):
        merges += 1
        mismatches.append(f"{case.name}: distinct people share a key {flat}")
    for index in case.no_speaker_facts:
        speaker_leaks += sum(
            1 for fact in records[index].facts
            if fact.subject.kind == "speaker")
    return records, matched, mismatches, speaker_leaks, merges


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_adversarial_identity_case(case):
    _, matched, mismatches, speaker_leaks, merges = _collect(case)
    assert not mismatches, "\n".join(mismatches)
    assert matched == len(case.expectations)
    assert speaker_leaks == 0
    assert merges == 0


def test_acceptance_set_has_no_false_merges_or_wrong_subjects():
    expected = sum(len(case.expectations) for case in CASES)
    matched = 0
    mismatches = []
    speaker_leaks = 0
    false_merges = 0
    for case in CASES:
        _, case_matched, case_mismatches, case_leaks, case_merges = _collect(
            case)
        matched += case_matched
        mismatches.extend(case_mismatches)
        speaker_leaks += case_leaks
        false_merges += case_merges
    assert mismatches == []
    assert matched == expected
    assert speaker_leaks == 0
    assert false_merges == 0


def test_shared_name_records_both_claimants():
    resolver, _ = _run(_case("two_relations_one_name_stay_distinct"))
    session = resolver.identities.session(
        "adv-two_relations_one_name_stay_distinct")
    first, second = session.ambiguous_names["sam"]
    assert first == "sam"
    assert second != first
    assert session.people[second].relation == "colleague"


def test_shared_name_is_reported_in_diagnostics():
    _, records = _run(_case("two_relations_one_name_stay_distinct"))
    assert any(note.startswith("ambiguous_name:")
               for note in records[1].ambiguities)


def test_resolved_correction_carries_no_stale_ambiguity():
    _, records = _run(_case("correction_extends_the_relation_name"))
    fact, = records[1].facts
    assert fact.subject.status == "resolved"
    assert fact.subject.ambiguity is None
