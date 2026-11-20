"""Stage 3 tests: speaker, subject, pronoun and conversation identity.

The rules under test are the ones that keep memory honest: a statement about a
relative never becomes a first-person fact, a pronoun is only resolved from
sufficient preceding evidence, and identity state never crosses a
conversation boundary.
"""

from __future__ import annotations

import pytest

from minemos.resolution import (
    IdentitySession,
    Resolver,
    SourceMessage,
    detect_language,
    interpret,
    normalize_text,
    resolve_identity,
)


def _run(turns):
    """Resolves (message_id, text, speaker) turns in one conversation."""
    resolver = Resolver()
    records = []
    for order, (message_id, text, speaker) in enumerate(turns):
        records.append(resolver.resolve(SourceMessage(
            "conv", message_id, text, speaker, float(order), order)).record)
    return resolver, records


def test_relation_statement_is_not_attributed_to_the_speaker():
    _, records = _run([("m0", "My brother lives in Oslo.", "a")])
    fact, = records[0].facts
    assert fact.subject.kind == "derived"
    assert fact.subject.relation == "brother"
    assert fact.subject.owner == "a"
    assert fact.subject.label == "user's brother"
    assert records[0].attributions[0].status == "resolved"


def test_relation_fact_never_becomes_first_person():
    # The relation guard runs even if Stage 2 mislabels the subject as speaker.
    resolver, records = _run([("m0", "My brother lives in Oslo.", "a")])
    fact, = records[0].facts
    assert fact.subject.label != "a"
    assert records[0].attributions[0].subject.label == "user's brother"
    assert "relation_subject_first_person" not in " ".join(
        records[0].abstentions)


def test_confirmed_pronoun_is_resolved_from_the_preceding_turn():
    _, records = _run([
        ("m0", "My brother is called Tom.", "a"),
        ("m1", "He lives in Oslo.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Tom"
    assert fact.subject.status == "resolved"
    assert fact.subject.confidence < 1.0


def test_pronoun_without_supporting_context_is_ambiguous():
    _, records = _run([("m0", "He lives in Oslo.", "a")])
    fact, = records[0].facts
    assert fact.subject.status != "resolved"
    assert fact.subject.kind == "unknown"
    assert any(item.startswith("unresolved_subject:")
               for item in records[0].abstentions)


def test_pronoun_gender_gate_blocks_a_mismatch():
    _, records = _run([
        ("m0", "My brother is called Tom.", "a"),
        ("m1", "She lives in Rome.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.status != "resolved"


def test_speaker_change_is_recorded_as_a_transition():
    _, records = _run([
        ("m0", "My name is Ada.", "a"),
        ("m1", "My name is Bob.", "b"),
    ])
    kinds = [transition.kind for transition in records[1].transitions]
    assert "speaker_change" in kinds
    assert records[0].transitions == ()


def test_relation_naming_registers_the_person():
    resolver, records = _run([("m0", "My brother is called Tom.", "a")])
    transitions = [item for item in records[0].transitions
                   if item.kind == "relation_linked"]
    assert transitions and transitions[0].subject_key == "tom"
    people = resolver.identities.session("conv").people
    assert "tom" in people
    assert people["tom"].kind == "named"


def test_lowercased_name_variant_links_to_the_canonical_person():
    resolver, records = _run([
        ("m0", "Clara Zeller plays the lute.", "a"),
        ("m1", "claraa zeller plays the lute.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.entity_key == "clara zeller"
    assert fact.subject.resolved_from == "name_variant"
    assert fact.subject.confidence <= 0.65
    assert any(item.kind == "name_change" for item in records[1].transitions)
    assert resolver.identities.session("conv").people["clara zeller"].mentions == 2


def test_conversation_scoped_session_is_not_reused():
    source = SourceMessage("convA", "m0", "I live in Berlin.", "a", 1.0, 0)
    normalized = normalize_text(source.text)
    interpretation = interpret(source, normalized, detect_language(source.text))
    with pytest.raises(ValueError):
        resolve_identity(IdentitySession(conversation_id="convB"),
                         source, interpretation)


def test_two_conversations_do_not_share_identity_state():
    resolver = Resolver()
    resolver.resolve(SourceMessage(
        "left", "m0", "My brother is called Tom.", "a", 1.0, 0))
    result = resolver.resolve(SourceMessage(
        "right", "m0", "He lives in Oslo.", "a", 1.0, 0))
    fact, = result.record.facts
    assert fact.subject.status != "resolved"


def test_attributions_mirror_the_facts():
    _, records = _run([("m0", "My brother lives in Oslo.", "a")])
    record = records[0]
    assert len(record.attributions) == len(record.facts)
    attribution = record.attributions[0]
    assert attribution.fact_id == record.facts[0].fact_id
    assert attribution.status == record.facts[0].subject.status
    assert attribution.scope == "conversation"
    assert attribution.speaker.id == "a"
    assert "direct_first_person" not in attribution.evidence
    assert "third_party_relation" in attribution.evidence

def test_pronoun_resolves_a_uniquely_named_person_without_a_gender():
    # "colleague" carries no gender, but the person was explicitly named and is
    # the only candidate, so the reference is decidable.
    _, records = _run([
        ("m0", "My colleague is called Ivo.", "a"),
        ("m1", "He speaks Dutch.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Ivo"
    assert fact.subject.status == "resolved"
    assert fact.subject.resolved_from == "pronoun_antecedent"


def test_two_unnamed_gender_people_leave_the_pronoun_ambiguous():
    _, records = _run([
        ("m0", "My colleague is called Ivo. My manager is called Piet.", "a"),
        ("m1", "He speaks Dutch.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.status != "resolved"
    assert fact.subject.ambiguity in ("pronoun_multiple_antecedents",
                                      "pronoun_needs_antecedent")


def test_pronoun_is_not_bound_to_a_relation_that_was_never_named():
    _, records = _run([
        ("m0", "My colleague owns a bakery.", "a"),
        ("m1", "He speaks Dutch.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.status != "resolved"


def test_an_unverified_name_never_becomes_a_person():
    # A prose fragment that reads like a name must not join the people
    # registry, so a later pronoun cannot be bound to it.
    resolver = Resolver()
    resolver.resolve(SourceMessage(
        "conv", "m0", "Dose this works at the new place.", "a", 1.0, 0))
    session = resolver.identities.session("conv")
    assert "dose this" not in session.people
    result = resolver.resolve(SourceMessage(
        "conv", "m1", "He lives in Oslo.", "a", 2.0, 1))
    fact, = result.record.facts
    assert fact.subject.status != "resolved"


def test_lowercased_exact_name_links_without_a_typo():
    # Losing the capital of a proper noun is the evidence: the spelling is
    # unchanged, so the mention resolves to the person already introduced.
    resolver, records = _run([
        ("m0", "Clara Caldwell plays the lute.", "a"),
        ("m1", "clara caldwell reads books.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.entity_key == "clara caldwell"
    assert fact.subject.status == "resolved"
    # Original spelling is preserved on the subject.
    assert fact.subject.label == "clara caldwell"
    assert not any(item.kind == "name_change" for item in records[1].transitions)
    assert resolver.identities.session("conv").people["clara caldwell"].mentions == 2


def test_capitalised_one_letter_variant_is_not_merged():
    # Similarity alone never merges two people: without the lost capital or an
    # explicit correction the two spellings stay separate identities.
    resolver, records = _run([
        ("m0", "Clara Caldwell plays the lute.", "a"),
        ("m1", "Claraa Caldwell plays the lute.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.label == "Claraa Caldwell"
    assert fact.subject.resolved_from == "named_mention"
    assert not any(item.kind == "name_change" for item in records[1].transitions)
    people = resolver.identities.session("conv").people
    assert set(people) == {"clara caldwell", "claraa caldwell"}


def test_near_miss_surnames_never_merge():
    resolver, records = _run([
        ("m0", "Clara Whitlock plays the lute.", "a"),
        ("m1", "Clara Whitlocke plays the lute.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.label == "Clara Whitlocke"
    people = resolver.identities.session("conv").people
    assert "clara whitlock" in people and "clara whitlocke" in people


def test_variant_conflict_is_preserved_as_ambiguous():
    resolver, records = _run([
        ("m0", "Clara Whitlock plays the lute.", "a"),
        ("m1", "Clara Whitlocke works at Acme.", "a"),
        ("m2", "Clara Whitlocke studies botany.", "a"),
        ("m3", "clara whitlok plays the lute.", "a"),
    ])
    fact, = records[3].facts
    assert fact.subject.status == "ambiguous"
    assert fact.subject.ambiguity == "name_variant_ambiguous"
    assert any(item.startswith("name_variant_ambiguous:")
               for item in records[3].ambiguities)
    assert any(item.startswith("unresolved_subject:")
               for item in records[3].abstentions)
    # No third identity is invented from the conflict.
    assert len(resolver.identities.session("conv").people) == 2


def test_speaker_name_is_never_a_pronoun_antecedent():
    # The only other person is the speaker, whose name is not a third-person
    # antecedent, so the pronoun stays unresolved instead of becoming "Ada".
    _, records = _run([
        ("m0", "My name is Ada.", "a"),
        ("m1", "He lives in Oslo.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.status != "resolved"
    assert fact.subject.label != "Ada"


def test_speaker_name_is_excluded_from_ungendered_antecedents():
    resolver, records = _run([
        ("m0", "My name is Ada.", "a"),
        ("m1", "My colleague is called Ivo.", "a"),
        ("m2", "He speaks Dutch.", "a"),
    ])
    fact, = records[2].facts
    assert fact.subject.label == "Ivo"
    assert fact.subject.resolved_from == "pronoun_antecedent"


def test_a_namesake_does_not_merge_with_the_speaker():
    resolver, records = _run([
        ("m0", "My name is Ada.", "a"),
        ("m1", "Ada lives in Oslo.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.kind == "named"
    assert fact.subject.entity_key == "ada"
    people = resolver.identities.session("conv").people
    assert people["speaker:a"].is_speaker
    assert not people["ada"].is_speaker


def test_explicit_correction_replaces_the_named_relation():
    resolver, records = _run([
        ("m0", "My brother is called Sam, I mean Sam Waters.", "a"),
    ])
    fact, = records[0].facts
    assert fact.value == "Sam Waters"
    assert fact.rule == "relation_called_correction"
    link, = [item for item in records[0].transitions
             if item.kind == "relation_linked"]
    assert link.subject_key == "sam waters"
    people = resolver.identities.session("conv").people
    assert "sam waters" in people and "sam" not in people


def test_rejected_spelling_is_not_kept_as_a_person():
    resolver, records = _run([
        ("m0", "My brother is called Sam Waters, not Sam Wallis.", "a"),
    ])
    fact, = records[0].facts
    assert fact.value == "Sam Waters"
    people = resolver.identities.session("conv").people
    assert "sam waters" in people
    assert "sam wallis" not in people


def test_self_name_correction_updates_the_speaker_record():
    resolver, records = _run([
        ("m0", "My name is Ada, I meant Adele.", "a"),
    ])
    fact, = records[0].facts
    assert fact.value == "Adele"
    assert fact.rule == "speaker_name_correction"
    people = resolver.identities.session("conv").people
    assert people["speaker:a"].label == "Adele"
    # The rejected spelling never becomes its own person.
    assert "ada" not in people


def test_corrected_relation_name_is_used_by_a_later_mention():
    resolver, records = _run([
        ("m0", "My brother is called Sam, I mean Sam Waters.", "a"),
        ("m1", "sam waters lives in Oslo.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.entity_key == "sam waters"
    assert resolver.identities.session("conv").people[
        "sam waters"].mentions == 2


def test_a_three_letter_name_survives_normalization():
    _, records = _run([
        ("m0", "My colleague is called Sam.", "a"),
        ("m1", "sam lives in Oslo.", "a"),
    ])
    assert "sam" in records[1].normalized.text
    fact, = records[1].facts
    assert fact.subject.label.casefold() == "sam"
    assert fact.subject.status == "resolved"


def test_a_noisy_surname_before_a_verb_still_links():
    # "lopes" is one substitution from the vocabulary verb "loves"; Stage 1
    # must leave it alone so the recorded "Ana Lopez" can be recognised.
    resolver, records = _run([
        ("m0", "My sister is called Ana Lopez.", "a"),
        ("m1", "ana lopes lives in Madrid.", "a"),
    ])
    assert records[1].normalized.text == "ana lopes lives in Madrid."
    fact, = records[1].facts
    assert fact.subject.entity_key == "ana lopez"
    assert fact.subject.label == "ana lopes"
    assert fact.subject.status == "resolved"
    assert resolver.identities.session("conv").people[
        "ana lopez"].label == "Ana Lopez"


def test_correction_in_the_next_message_renames_the_relation():
    resolver, records = _run([
        ("m0", "My brother is called Sam.", "a"),
        ("m1", "I mean Sam Waters.", "a"),
    ])
    fact, = records[1].facts
    assert fact.value == "Sam Waters"
    assert fact.rule == "name_correction"
    assert fact.subject.entity_key == "sam"
    change, = [item for item in records[1].transitions
               if item.kind == "name_change"]
    assert change.subject_key == "sam"
    assert "evidence=speaker_correction" in change.evidence
    session = resolver.identities.session("conv")
    assert session.people["sam"].label == "Sam Waters"
    assert session.relation_links["a:relation:brother"] == "sam"


def test_correction_targets_the_person_whose_name_it_extends():
    # "Sam Waters" extends the brother's recorded name, so the correction is
    # not stolen by Tom, who was merely mentioned after the naming statement.
    resolver, records = _run([
        ("m0", "My brother is called Sam. Tom lives in Oslo.", "a"),
        ("m1", "I mean Sam Waters.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.entity_key == "sam"
    people = resolver.identities.session("conv").people
    assert people["sam"].label == "Sam Waters"
    assert people["tom"].label == "Tom"
    assert "sam waters" not in people
    assert len(people) == 3  # sam, Tom and the derived "user's brother"


def test_correction_after_an_unrelated_message_is_abstained():
    resolver, records = _run([
        ("m0", "My brother is called Sam.", "a"),
        ("m1", "The weather is cold.", "a"),
        ("m2", "I mean Sam Waters.", "a"),
    ])
    fact, = records[2].facts
    assert fact.subject.status != "resolved"
    assert any(item.startswith("correction_target_unknown:")
               for item in records[2].ambiguities)
    assert any(item.startswith("unresolved_subject:")
               for item in records[2].abstentions)
    assert resolver.identities.session("conv").people["sam"].label == "Sam"


def test_correction_conflict_with_another_person_is_preserved():
    resolver, records = _run([
        ("m0", "Clara Caldwell plays the lute.", "a"),
        ("m1", "clara caldwel plays the lute.", "a"),
        ("m2", "Tom lives in Oslo.", "a"),
        ("m3", "I mean clara caldwel.", "a"),
    ])
    fact, = records[3].facts
    assert fact.subject.status == "ambiguous"
    assert fact.subject.ambiguity == "correction_conflicts_with_established_name"
    assert any(item.startswith("correction_conflicts_with_established_name:")
               for item in records[3].ambiguities)
    assert any(item.startswith("unresolved_subject:")
               for item in records[3].abstentions)
    people = resolver.identities.session("conv").people
    assert people["tom"].label == "Tom"
    assert people["clara caldwell"].label == "Clara Caldwell"


def test_correction_can_update_the_speakers_own_name():
    resolver, records = _run([
        ("m0", "My name is Sam.", "a"),
        ("m1", "I mean Sam Waters.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.entity_key == "speaker:a"
    change, = [item for item in records[1].transitions
               if item.kind == "name_change"]
    assert change.subject_key == "speaker:a"
    person = resolver.identities.session("conv").people["speaker:a"]
    assert person.label == "Sam Waters"
    assert person.is_speaker


def test_confirmation_of_the_recorded_name_invents_no_identity():
    resolver, records = _run([
        ("m0", "My brother is called Sam. Tom lives in Oslo.", "a"),
        ("m1", "I mean Sam.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.entity_key == "sam"
    assert fact.subject.status == "resolved"
    assert not [item for item in records[1].transitions
                if item.kind == "name_change"]
    people = resolver.identities.session("conv").people
    assert people["sam"].label == "Sam"
    assert set(people) == {"sam", "tom", "a:relation:brother"}


def test_exact_recorded_name_outranks_a_longer_name_it_extends():
    resolver, records = _run([
        ("m0", "My brother is called Sam.", "a"),
        ("m1", "Sam Waters lives in Oslo.", "a"),
        ("m2", "I mean Sam.", "a"),
    ])
    fact, = records[2].facts
    assert fact.subject.entity_key == "sam"
    people = resolver.identities.session("conv").people
    assert people["sam"].label == "Sam"
    assert people["sam waters"].label == "Sam Waters"


def test_repeating_a_linked_variant_keeps_the_canonical_label():
    resolver, records = _run([
        ("m0", "Clara Caldwell plays the lute.", "a"),
        ("m1", "clara caldwel plays the lute.", "a"),
        ("m2", "I mean clara caldwel.", "a"),
    ])
    fact, = records[2].facts
    assert fact.subject.entity_key == "clara caldwell"
    assert not [item for item in records[2].transitions
                if item.kind == "name_change"]
    assert resolver.identities.session("conv").people[
        "clara caldwell"].label == "Clara Caldwell"


def test_a_correction_between_namesakes_is_ambiguous():
    resolver, records = _run([
        ("m0", "My name is Ada.", "a"),
        ("m1", "Ada lives in Oslo.", "a"),
        ("m2", "I mean Ada Lovelace.", "a"),
    ])
    fact, = records[2].facts
    assert fact.subject.status == "ambiguous"
    assert fact.subject.ambiguity == "correction_target_ambiguous"
    assert any(item.startswith("correction_target_ambiguous:")
               for item in records[2].ambiguities)
    people = resolver.identities.session("conv").people
    assert people["speaker:a"].label == "Ada"
    assert people["ada"].label == "Ada"


def test_misspelled_relation_subject_is_not_the_speaker():
    _, records = _run([("m0", "myy brther lives in Oslo.", "a")])
    fact, = records[0].facts
    attribution = records[0].attributions[0]
    assert fact.subject.kind == "derived"
    assert fact.subject.owner == "a"
    assert fact.subject.label == "user's brother"
    assert attribution.speaker.id == "a"
    assert "third_party_relation" in attribution.evidence
    assert "direct_first_person" not in attribution.evidence


def test_misspelled_relation_links_to_the_named_person():
    resolver, records = _run([
        ("m0", "myy brther's name is Sam.", "a"),
        ("m1", "my brther lives in Oslo.", "a"),
    ])
    naming, statement = records[0], records[1]
    link, = [item for item in naming.transitions if item.kind == "relation_linked"]
    assert link.subject_key == "sam"
    fact, = statement.facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Sam"
    assert fact.subject.resolved_from == "relation_link"
    assert resolver.identities.session("conv").relation_links[
        "a:relation:brother"] == "sam"


def test_a_later_named_person_wins_over_the_misspelled_relation():
    _, records = _run([
        ("m0", "my brther lives in Oslo.", "a"),
        ("m1", "My brother is called Tom.", "a"),
        ("m2", "myy brther works at Acme.", "a"),
    ])
    fact, = records[2].facts
    assert fact.subject.kind == "named"
    assert fact.subject.label == "Tom"
    assert fact.subject.resolved_from == "relation_link"


def test_a_correction_applies_to_a_misspelled_relation_name():
    resolver, records = _run([
        ("m0", "myy brther's name is Sam.", "a"),
        ("m1", "I mean Sam Waters.", "a"),
        ("m2", "my brther lives in Oslo.", "a"),
    ])
    change, = [item for item in records[1].transitions
               if item.kind == "name_change"]
    assert change.subject_key == "sam"
    fact, = records[2].facts
    assert fact.subject.entity_key == "sam"
    people = resolver.identities.session("conv").people
    assert people["sam"].label == "Sam Waters"


def test_lowercase_alias_transposition_links_once():
    resolver, records = _run([
        ("m0", "Clara Caldwell lives in Oslo.", "a"),
        ("m1", "clara cladwell works at Acme.", "a"),
    ])
    change, = [item for item in records[1].transitions
               if item.kind == "name_change"]
    assert change.subject_key == "clara caldwell"
    assert "edit_distance=1" in change.evidence
    facts = [fact for record in records for fact in record.facts]
    assert len({fact.subject.entity_key or fact.subject.label
                for fact in facts if fact.subject.kind == "named"}) == 1
    assert resolver.identities.session("conv").people[
        "clara caldwell"].mentions == 2


def test_a_capitalized_typo_is_a_separate_person():
    # Capitalisation is the evidence for a name; similarity alone must not
    # merge two people whose spellings are close but both explicit.
    resolver, records = _run([
        ("m0", "Clara Caldwell lives in Oslo.", "a"),
        ("m1", "Clara Cardwell works at Acme.", "a"),
    ])
    assert not records[1].transitions
    people = resolver.identities.session("conv").people
    assert people["clara caldwell"].label == "Clara Caldwell"
    assert people["clara cardwell"].label == "Clara Cardwell"


def test_pronoun_between_two_unnamed_relations_stays_ambiguous():
    _, records = _run([
        ("m0", "My sister lives in Oslo. My mother lives in Bergen.", "a"),
        ("m1", "She works at Acme.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.status != "resolved"
    assert fact.subject.kind == "unknown"
    assert any(item.startswith("unresolved_subject:")
               for item in records[1].abstentions)
    assert any(item.startswith("pronoun_multiple_antecedents:")
               for item in records[1].ambiguities)


def test_pronoun_never_resolves_to_the_speaker():
    # The speaker's own name is excluded as an antecedent; a pronoun with no
    # other eligible person must abstain rather than become first-person.
    _, records = _run([
        ("m0", "My name is Ada.", "a"),
        ("m1", "He works at Acme.", "a"),
    ])
    fact, = records[1].facts
    assert fact.subject.status != "resolved"
    assert fact.subject.kind == "unknown"


def test_a_noisy_relation_does_not_cross_conversations():
    resolver = Resolver()
    resolver.resolve(SourceMessage(
        "left", "m0", "myy brther's name is Sam.", "a", 1.0, 0))
    result = resolver.resolve(SourceMessage(
        "right", "m0", "my brther lives in Oslo.", "a", 2.0, 0))
    fact, = result.record.facts
    assert fact.subject.kind == "derived"
    assert fact.subject.label == "user's brother"
    left = resolver.identities.session("left")
    right = resolver.identities.session("right")
    assert "sam" in left.people
    assert "sam" not in right.people
