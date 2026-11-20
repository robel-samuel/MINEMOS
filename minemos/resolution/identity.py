"""Stage 3 — identity and provenance resolution.

Stage 2 answers *what was said*.  This stage answers *who it is about* and
*how sure we are*, using only evidence already present in the conversation:

* every message carries a host-supplied speaker; the resolver never guesses a
  speaker from the message body, and an absent speaker stays unknown;
* a possessive subject (``user's brother``) stays distinct from the speaker —
  a statement about a brother is never promoted to a fact about the user;
* pronouns resolve only from a preceding, gender-compatible subject inside the
  **same conversation** and within a bounded window; otherwise the subject
  stays ``ambiguous`` or ``unknown`` with a reason;
* identities are conversation-scoped by construction: sessions are keyed by
  conversation id and never share people, so no cross-conversation merge can
  happen without explicit evidence;
* a relation that is later named (``my brother`` → ``Sam``) is linked by an
  append-only transition.  Facts recorded before the link keep the subject
  they were resolved with at the time.

The stage produces no new memory effects: it returns resolved fact copies,
per-fact attributions and transition notes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Tuple

from .config import ResolutionConfig
from .contracts import (
    Attribution,
    FactCandidate,
    IdentityTransition,
    SourceMessage,
    SpeakerRef,
    SubjectRef,
)
from .interpret import Interpretation
from .lexicon import RELATION_GENDER
from .similarity import edit_distance, is_transposition

_GENDER_PRONOUNS = {
    "he": "male", "him": "male", "his": "male", "himself": "male",
    "she": "female", "her": "female", "hers": "female", "herself": "female",
    "they": "neutral", "them": "neutral", "their": "neutral",
    "it": "neuter", "its": "neuter",
    "él": "male", "el": "male", "ella": "female",
    "il": "male", "elle": "female", "ils": "neutral", "elles": "neutral",
    "er": "male", "sie": "female", "es": "neuter",
    # Amharic third-person pronouns.
    "እሱ": "male", "እሷ": "female", "እነሱ": "neutral",
}

_RELATION_RULES = frozenset({
    "relation_name", "relation_called", "relation_attribute",
    # A possessive whose spelling was recovered from chat noise is still a
    # relation statement, so it can never be recorded as first-person.
    "relation_name_near_match", "relation_attribute_near_match",
    # The speaker supplied the spelling explicitly ("my brother is called
    # Sam, I mean Sam Waters"); the corrected clause is still a relation
    # statement and must never become first-person.
    "relation_name_correction", "relation_called_correction",
})

_UNRESOLVED_STATUSES = ("ambiguous", "unknown")


@dataclass(frozen=True)
class PersonRecord:
    key: str
    label: str
    kind: str
    relation: Optional[str]
    owner: Optional[str]
    gender: Optional[str]
    first_message_id: str
    last_message_id: str
    aliases: Tuple[str, ...] = ()
    mentions: int = 1
    # True for the name a speaker gave for themselves.  Such a record can link
    # later spellings of the same name, but it is never a pronoun antecedent:
    # a third-person pronoun must not resolve back to the speaker.
    is_speaker: bool = False


@dataclass(frozen=True)
class _VariantMatch:
    person: PersonRecord
    distance: int


@dataclass
class _RecentSubject:
    subject: SubjectRef
    message_id: str
    fact_id: str


@dataclass
class IdentitySession:
    """Conversation-scoped identity state.

    One session per conversation.  The registry refuses to share a session
    between conversations, which is what keeps cross-conversation identity
    merging impossible without explicit host evidence.
    """

    conversation_id: str
    people: Dict[str, PersonRecord] = field(default_factory=dict)
    relation_links: Dict[str, str] = field(default_factory=dict)
    # Lowercased typing variants that were linked to a registered person.
    aliases: Dict[str, str] = field(default_factory=dict)
    recent: Deque[_RecentSubject] = field(
        default_factory=lambda: deque(maxlen=64))
    transitions: List[IdentityTransition] = field(default_factory=list)
    last_speaker: Optional[str] = None
    last_order: int = -1
    last_message_id: Optional[str] = None
    # The person most recently named in this conversation, and the message
    # that named them.  A correction in the next message ("I meant Waters")
    # applies only to a name from the message right before it.
    last_name_target: Optional[str] = None
    last_name_message_id: Optional[str] = None
    # Base name (casefolded) claimed by more than one relation, mapped to the
    # distinct person keys that claim it.  Two relations with the same name are
    # two people until the speaker says otherwise, so a bare mention of that
    # name cannot pick one.
    ambiguous_names: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("IdentitySession needs a conversation id")

    def observe_order(self, order: int) -> None:
        self.last_order = max(self.last_order, order)


class IdentityRegistry:
    """Holds one session per conversation for the life of a resolver."""

    def __init__(self) -> None:
        self._sessions: Dict[str, IdentitySession] = {}

    def session(self, conversation_id: str) -> IdentitySession:
        if not conversation_id:
            raise ValueError("a conversation id is required")
        session = self._sessions.get(conversation_id)
        if session is None:
            session = IdentitySession(conversation_id)
            self._sessions[conversation_id] = session
        return session

    def conversations(self) -> Tuple[str, ...]:
        return tuple(self._sessions)

    def transitions(self) -> Tuple[IdentityTransition, ...]:
        found: List[IdentityTransition] = []
        for session in self._sessions.values():
            found.extend(session.transitions)
        return tuple(found)


@dataclass(frozen=True)
class IdentityResult:
    facts: Tuple[FactCandidate, ...]
    attributions: Tuple[Attribution, ...]
    transitions: Tuple[IdentityTransition, ...] = ()
    ambiguities: Tuple[str, ...] = ()
    abstentions: Tuple[str, ...] = ()
    confidence: float = 0.0


def subject_key(subject: SubjectRef) -> Optional[str]:
    """A stable, conversation-local identifier for a subject."""
    if subject.kind == "named":
        return subject.entity_key or subject.label.casefold()
    if subject.kind == "derived" and subject.relation:
        return subject.entity_key or (
            f"{subject.owner or 'user'}:relation:{subject.relation}")
    if subject.kind == "speaker":
        return f"speaker:{subject.owner or subject.label}"
    return None


def _gender_of(subject: SubjectRef, session: IdentitySession
               ) -> Optional[str]:
    if subject.kind == "derived" and subject.relation:
        gender = RELATION_GENDER.get(subject.relation.casefold())
        if gender:
            return gender
    key = subject_key(subject)
    if key and key in session.people:
        return session.people[key].gender
    return None


def _relation_link_key(subject: SubjectRef) -> Optional[str]:
    """Links are keyed by owner and relation, never by relation alone.

    ``my brother`` from one speaker and ``my brother`` from another are two
    people, so the link key carries the owner.
    """
    if not subject.relation:
        return None
    return subject.entity_key or (
        f"{subject.owner or 'user'}:relation:{subject.relation}")


def _linked_person(subject: SubjectRef, session: IdentitySession
                   ) -> Optional[PersonRecord]:
    key = _relation_link_key(subject)
    if key is None:
        return None
    linked = session.relation_links.get(key)
    if linked is None:
        return None
    return session.people.get(linked)


def _pronoun_gender(subject: SubjectRef) -> Optional[str]:
    if subject.kind != "unknown":
        return None
    return _GENDER_PRONOUNS.get(subject.label.casefold())


def _antecedent_ref(item: _RecentSubject, session: IdentitySession
                    ) -> SubjectRef:
    subject = item.subject
    person = _linked_person(subject, session)
    if person is not None:
        subject = SubjectRef(
            kind="named", label=person.label, confidence=subject.confidence,
            owner=subject.owner, relation=subject.relation,
            entity_key=person.key, resolved_from="relation_link",
            status="resolved")
    return subject


def _resolve_pronoun(subject: SubjectRef, session: IdentitySession,
                     config: ResolutionConfig) -> SubjectRef:
    """Resolves a pronoun only when exactly one antecedent fits.

    An antecedent with a recorded gender must agree with the pronoun.  A
    uniquely named person whose gender is not recorded ("my colleague is
    called Ivo") is accepted only when it is the single person in the window,
    because a competing mention would make the reference ambiguous.
    """
    gender = _pronoun_gender(subject)
    if gender is None or gender in ("neutral", "neuter"):
        return replace(
            subject, kind="unknown", status="unknown",
            ambiguity="pronoun_needs_antecedent",
            resolved_from="pronoun_unresolved", confidence=0.0)
    window = max(1, config.pronoun_window)
    matched: List[_RecentSubject] = []
    ungendered: List[_RecentSubject] = []
    claimants = set()
    seen = set()
    for item in list(session.recent)[-window:]:
        if item.subject.status != "resolved":
            continue
        ref = _antecedent_ref(item, session)
        key = subject_key(ref)
        if key is None or key in seen:
            continue
        person = session.people.get(key)
        if person is not None and (person.kind != "named" or person.is_speaker):
            # A relation-only subject is not an identifiable person, and the
            # speaker's own name is never a third-person antecedent.
            seen.add(key)
            continue
        if person is not None \
                and person.label.casefold() in session.ambiguous_names:
            # Two different people were given this name; a pronoun that could
            # point at either is not decidable from the name alone.
            seen.add(key)
            claimants.add(person.key)
        ref_gender = _gender_of(ref, session)
        if ref_gender is None:
            if person is not None:
                seen.add(key)
                ungendered.append(
                    _RecentSubject(ref, item.message_id, item.fact_id))
            continue
        if ref_gender == gender:
            seen.add(key)
            matched.append(_RecentSubject(ref, item.message_id, item.fact_id))
    if len(claimants) > 1:
        return replace(
            subject, kind="unknown", status="ambiguous",
            ambiguity="pronoun_ambiguous_name",
            resolved_from="pronoun_unresolved", confidence=0.0)
    if matched:
        candidates = matched
    elif len(ungendered) == 1:
        candidates = ungendered
    else:
        candidates = []
    if not candidates:
        return replace(
            subject, kind="unknown", status="unknown",
            ambiguity="pronoun_needs_antecedent",
            resolved_from="pronoun_unresolved", confidence=0.0)
    if len(candidates) > 1:
        return replace(
            subject, kind="unknown", status="ambiguous",
            ambiguity="pronoun_multiple_antecedents",
            resolved_from="pronoun_unresolved", confidence=0.0)
    antecedent = candidates[0]
    source = antecedent.subject
    return SubjectRef(
        kind="named" if source.kind in ("named", "derived") else source.kind,
        label=source.label,
        confidence=round(min(0.72, source.confidence * 0.85), 4),
        owner=source.owner,
        relation=source.relation,
        entity_key=subject_key(source),
        resolved_from="pronoun_antecedent",
        status="resolved",
        antecedent_message_id=antecedent.message_id,
        antecedent_fact_id=antecedent.fact_id,
    )


def _register_name(subject: SubjectRef, fact: FactCandidate,
                   session: IdentitySession) -> bool:
    """Registers ``my brother is called Sam`` and links the relation.

    Returns True when this call established a new relation link.

    A name already carried by a *different* relation does not merge the two
    people: the speaker has described two roles, and nothing says both roles
    are one person.  The second relation keeps its own scoped identity and the
    bare name is recorded as ambiguous for later mentions and pronouns.
    """
    relation = subject.relation
    if not relation or not subject.owner:
        return False
    base = fact.value.casefold()
    key = base
    link_key = _relation_link_key(subject) or relation
    existing = session.people.get(base)
    if existing is not None and existing.relation is not None and (
            existing.relation != relation or existing.owner != subject.owner):
        key = f"{base}@{link_key}"
        claimants = session.ambiguous_names.get(base, ())
        for claimant in (existing.key, key):
            if claimant not in claimants:
                claimants += (claimant,)
        session.ambiguous_names[base] = claimants
    if session.relation_links.get(link_key) == key:
        return False
    existing = session.people.get(key)
    session.people[key] = PersonRecord(
        key=key,
        label=fact.value,
        kind="named",
        relation=relation,
        owner=subject.owner,
        gender=RELATION_GENDER.get(relation.casefold()),
        first_message_id=existing.first_message_id if existing
        else fact.evidence[0].message_id,
        last_message_id=fact.evidence[0].message_id,
        aliases=existing.aliases if existing else (subject.label,),
    )
    derived_key = subject_key(subject)
    if derived_key:
        session.people[derived_key] = PersonRecord(
            key=derived_key,
            label=subject.label,
            kind="derived",
            relation=relation,
            owner=subject.owner,
            gender=RELATION_GENDER.get(relation.casefold()),
            first_message_id=fact.evidence[0].message_id,
            last_message_id=fact.evidence[0].message_id,
            aliases=(fact.value,),
        )
    session.relation_links[link_key] = key
    session.last_name_target = key
    session.last_name_message_id = fact.evidence[0].message_id
    return True


def _register_speaker_name(speaker: SpeakerRef, fact: FactCandidate,
                           session: IdentitySession) -> None:
    """Records the name a speaker gave for themselves.

    The record is never a pronoun antecedent and never a variant source: a
    third-person pronoun must not resolve back to the speaker.  It is keyed by
    speaker id, so a namesake in the same conversation stays a separate
    identity instead of being merged into the speaker.
    """
    name = fact.value.strip()
    if not name:
        return
    key = f"speaker:{speaker.id}"
    existing = session.people.get(key)
    message_id = fact.evidence[0].message_id
    session.people[key] = PersonRecord(
        key=key,
        label=existing.label if existing is not None else name,
        kind="named",
        relation=None,
        owner=speaker.id,
        gender=None,
        first_message_id=(existing.first_message_id
                          if existing is not None else message_id),
        last_message_id=message_id,
        aliases=((existing.aliases + (name,)) if existing is not None
                 else (name,)),
        mentions=(existing.mentions + 1) if existing is not None else 1,
        is_speaker=True,
    )
    session.last_name_target = key
    session.last_name_message_id = message_id


def _apply_relation_link(subject: SubjectRef, session: IdentitySession
                         ) -> SubjectRef:
    person = _linked_person(subject, session)
    if person is None:
        return subject
    return SubjectRef(
        kind="named",
        label=person.label,
        confidence=round(min(0.9, subject.confidence), 4),
        owner=subject.owner,
        relation=subject.relation,
        entity_key=person.key,
        resolved_from="relation_link",
        status="resolved",
        antecedent_message_id=person.first_message_id,
    )


_LABEL_PUNCT = " \t,;:.!?\"'`()[]{}"


def _name_variant(subject: SubjectRef, session: IdentitySession,
                  config: ResolutionConfig
                  ) -> Tuple[Optional[_VariantMatch], bool]:
    """Finds the registered person a lowercased mention is a spelling of.

    Only a mention that lost the capitalisation of a proper noun is eligible:
    ``clara cardwell`` may be a typing variant of an explicitly capitalised
    ``Clara Caldwell``, while ``Mira Ahsford`` is a different person.  The
    lost capital is the evidence; similarity alone never merges two people.

    Returns ``(match, ambiguous)``.  ``ambiguous`` means more than one
    registered person is within the bound, which is a conflict to preserve
    rather than a reason to create a third identity.
    """
    if not config.link_repeated_variants:
        return None, False
    label = subject.label.strip()
    new_tokens = [token.strip(_LABEL_PUNCT) for token in label.split()]
    if not new_tokens or not all(new_tokens):
        return None, False
    if not all(token.islower() for token in new_tokens):
        return None, False
    matches: List[_VariantMatch] = []
    for person in session.people.values():
        if person.kind != "named" or person.is_speaker:
            continue
        old_tokens = [token.strip(_LABEL_PUNCT) for token in person.label.split()]
        if len(old_tokens) != len(new_tokens):
            continue
        total_distance = 0
        changed = 0
        eligible = True
        for new_token, old_token in zip(new_tokens, old_tokens):
            new_key = new_token.casefold()
            old_key = old_token.casefold()
            if new_key == old_key:
                continue
            if len(new_token) < 4 or len(old_token) < 4 \
                    or not old_token[:1].isupper():
                eligible = False
                break
            distance = edit_distance(
                new_key, old_key, config.name_variant_max_distance)
            if distance > config.name_variant_max_distance:
                eligible = False
                break
            if len(new_token) == 4 and distance == 1 \
                    and not is_transposition(new_key, old_key):
                eligible = False
                break
            total_distance += distance
            changed += 1
        if not eligible or not changed:
            continue
        required = (1 if total_distance == changed
                    else max(1, config.repeated_variant_support))
        if person.mentions < required:
            continue
        matches.append(_VariantMatch(person=person,
                                     distance=total_distance))
    if not matches:
        return None, False
    if len(matches) > 1:
        return None, True
    return matches[0], False


def _speaker_ref(source: SourceMessage) -> SpeakerRef:
    supplied = source.speaker
    if supplied:
        return SpeakerRef(id=supplied, label=supplied, confidence=1.0,
                          source="host")
    return SpeakerRef(id="unknown", label="unknown", confidence=0.0,
                      source="missing")


def _attribution_evidence(fact: FactCandidate, subject: SubjectRef,
                          speaker: SpeakerRef) -> Tuple[str, ...]:
    evidence: List[str] = [f"rule={fact.rule}"]
    if subject.kind == "speaker":
        evidence.append("direct_first_person")
    elif subject.kind == "derived":
        evidence.append("third_party_relation")
    elif subject.kind == "named":
        evidence.append("third_party_report")
        if subject.resolved_from == "pronoun_antecedent":
            evidence.append("pronoun_antecedent")
        elif subject.resolved_from == "relation_link":
            evidence.append("relation_link")
        elif subject.resolved_from == "name_variant":
            evidence.append("name_variant")
        elif subject.resolved_from == "speaker_correction":
            evidence.append("speaker_correction")
    else:
        evidence.append("subject_unresolved")
    if speaker.id == "unknown":
        evidence.append("speaker_missing")
    return tuple(evidence)


def _correction_targets(corrected: str,
                        session: IdentitySession) -> List[str]:
    """Established people whose name the corrected spelling reproduces.

    ``I mean Sam Waters`` after ``my brother is called Sam`` extends the name
    already recorded for the brother.  A mention of somebody else that came
    later ("Tom lives in Oslo") must not steal the correction: the recorded
    name that the correction builds on is the evidence for its target.
    """
    tokens = [token.strip(_LABEL_PUNCT).casefold()
              for token in corrected.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return []
    exact: List[str] = []
    extended: List[str] = []
    for person in session.people.values():
        if person.kind != "named":
            continue
        label = [token.strip(_LABEL_PUNCT).casefold()
                 for token in person.label.split()]
        label = [token for token in label if token]
        if not label:
            continue
        if label == tokens:
            exact.append(person.key)
            continue
        shorter, longer = ((tokens, label) if len(tokens) <= len(label)
                           else (label, tokens))
        if longer[:len(shorter)] != shorter:
            continue
        extended.append(person.key)
    # An exact repetition of a recorded name outranks a name it extends.
    return exact or extended


def _apply_correction_fact(subject: SubjectRef, fact: FactCandidate,
                           session: IdentitySession, source: SourceMessage,
                           transitions: List[IdentityTransition],
                           ambiguities: List[str]) -> SubjectRef:
    """Links a corrected spelling to the name the speaker used just before.

    The corrected spelling becomes an alias of that person instead of a second
    identity.  When the spelling already belongs to someone else the speaker
    has not said the two are the same person, so the conflict stays open.
    """
    corrected = fact.value.strip()
    if not corrected \
            or session.last_name_message_id != session.last_message_id:
        ambiguities.append(f"correction_target_unknown:{fact.fact_id}")
        return subject
    established = _correction_targets(corrected, session)
    if len(established) > 1:
        # Two registered people fit the spelling; choosing either would merge
        # identities on similarity alone.
        message = "correction_target_ambiguous"
        ambiguities.append(f"{message}:{fact.fact_id}")
        return replace(subject, kind="named", label=corrected,
                       status="ambiguous", ambiguity=message,
                       resolved_from="speaker_correction", confidence=0.0)
    target_key = established[0] if established else session.last_name_target
    person = session.people.get(target_key) if target_key else None
    if person is None:
        ambiguities.append(f"correction_target_unknown:{fact.fact_id}")
        return subject
    key = corrected.casefold()
    canonical = session.aliases.get(key, key)
    if canonical != target_key \
            and (key in session.people or session.aliases.get(key) is not None):
        message = "correction_conflicts_with_established_name"
        ambiguities.append(f"{message}:{fact.fact_id}")
        return replace(subject, kind="named", label=corrected,
                       status="ambiguous", ambiguity=message,
                       resolved_from="speaker_correction", confidence=0.0)
    if key != target_key:
        session.aliases[key] = target_key
        if canonical != target_key:
            # A spelling already linked to this person is a known variant; a
            # repeat must not replace the canonical label with the variant.
            aliases = (person.aliases if corrected in person.aliases
                       else person.aliases + (corrected,))
            session.people[target_key] = replace(
                person, label=corrected, aliases=aliases,
                last_message_id=source.message_id,
                mentions=person.mentions + 1)
            transitions.append(IdentityTransition(
                kind="name_change", message_id=source.message_id,
                subject_key=target_key,
                evidence=("evidence=speaker_correction",
                          f"corrected_key={key}")))
    session.last_name_target = target_key
    session.last_name_message_id = source.message_id
    return replace(subject, kind="named", label=corrected,
                   owner=person.owner, relation=person.relation,
                   entity_key=target_key, status="resolved",
                   resolved_from="speaker_correction", confidence=0.9,
                   ambiguity=None,
                   antecedent_message_id=person.first_message_id)


def resolve_identity(session: IdentitySession, source: SourceMessage,
                     interpretation: Interpretation,
                     config: Optional[ResolutionConfig] = None
                     ) -> IdentityResult:
    """Resolves speakers, subjects and pronouns for one message."""
    cfg = config or ResolutionConfig()
    if session.conversation_id != source.conversation_id:
        raise ValueError(
            "identity sessions are conversation-scoped; refusing to reuse "
            f"session {session.conversation_id!r} for "
            f"{source.conversation_id!r}")

    speaker = _speaker_ref(source)
    ambiguities: List[str] = []
    abstentions: List[str] = []
    transitions: List[IdentityTransition] = []
    facts: List[FactCandidate] = []
    attributions: List[Attribution] = []
    confidence_terms: List[float] = []

    if speaker.id == "unknown":
        ambiguities.append("speaker_not_supplied")
    elif session.last_speaker is not None and session.last_speaker != speaker.id:
        transitions.append(IdentityTransition(
            kind="speaker_change", message_id=source.message_id,
            evidence=(f"previous={session.last_speaker}",
                      f"current={speaker.id}")))
    session.last_speaker = speaker.id

    for fact in interpretation.facts:
        subject = fact.subject

        if fact.rule in _RELATION_RULES and subject.kind == "speaker":
            # Defence in depth: a relation statement must never be recorded as
            # first-person, whatever Stage 2 produced.
            subject = replace(
                subject, status="ambiguous",
                ambiguity="relation_subject_not_the_speaker",
                resolved_from="relation_guard", confidence=0.0)
            ambiguities.append(f"relation_subject_first_person:{fact.fact_id}")

        elif subject.kind == "derived":
            owner = subject.owner or speaker.id
            entity_key = subject.entity_key or (
                f"{owner}:relation:{subject.relation}")
            subject = replace(subject, owner=owner, entity_key=entity_key)
            if fact.predicate == "name":
                if _register_name(subject, fact, session):
                    transitions.append(IdentityTransition(
                        kind="relation_linked", message_id=source.message_id,
                        subject_key=fact.value.casefold(),
                        evidence=(f"relation={subject.relation}",
                                  "evidence=naming_fact")))
                if fact.value.casefold() in session.ambiguous_names:
                    ambiguities.append(f"ambiguous_name:{fact.fact_id}")
            else:
                subject = _apply_relation_link(subject, session)

        elif fact.rule == "name_correction":
            subject = _apply_correction_fact(
                subject, fact, session, source, transitions, ambiguities)

        elif subject.kind == "speaker" and fact.predicate == "name":
            _register_speaker_name(speaker, fact, session)

        elif subject.kind == "unknown" and cfg.resolve_pronouns:
            subject = _resolve_pronoun(subject, session, cfg)
            if subject.status in _UNRESOLVED_STATUSES:
                ambiguities.append(f"{subject.ambiguity}:{fact.fact_id}")

        elif subject.kind == "named" and subject.status == "resolved" \
                and subject.label.casefold() in session.ambiguous_names:
            # The bare name was given to two different relations earlier in
            # this conversation.  Attaching the fact to either one would be a
            # guess, and a third identity would be an invention.
            subject = replace(
                subject, status="ambiguous",
                ambiguity="ambiguous_name_reference",
                resolved_from="ambiguous_name", confidence=0.0)
            ambiguities.append(f"ambiguous_name_reference:{fact.fact_id}")

        elif subject.kind == "named" and subject.status == "resolved":
            # Only an attested name joins the conversation's people registry;
            # an unverified candidate must not become a pronoun antecedent or
            # a variant source for later messages.
            key = subject.entity_key or subject.label.casefold()
            canonical = session.aliases.get(key)
            if canonical is not None and canonical in session.people:
                subject = replace(
                    subject, entity_key=canonical,
                    resolved_from="name_variant",
                    confidence=round(min(subject.confidence, 0.65), 4))
                key = canonical
            previous = session.people.get(key)
            variant_ambiguous = False
            if previous is None:
                linked, variant_ambiguous = _name_variant(subject, session, cfg)
                if linked is not None:
                    person = linked.person
                    session.aliases[key] = person.key
                    transitions.append(IdentityTransition(
                        kind="name_change",
                        message_id=source.message_id,
                        subject_key=person.key,
                        evidence=(f"variant_key={key}",
                                  f"edit_distance={linked.distance}",
                                  "evidence=lowercase_typo")))
                    subject = replace(
                        subject, entity_key=person.key,
                        resolved_from="name_variant",
                        confidence=round(min(subject.confidence, 0.65), 4))
                    key = person.key
                    previous = session.people.get(key)
                elif variant_ambiguous:
                    # Two established people fit the mention.  Picking either
                    # would merge identities on similarity alone, so the
                    # conflict is preserved and no third identity is created.
                    subject = replace(
                        subject, status="ambiguous",
                        ambiguity="name_variant_ambiguous",
                        resolved_from="name_variant_ambiguous",
                        confidence=0.0)
                    ambiguities.append(
                        f"name_variant_ambiguous:{fact.fact_id}")
            if not variant_ambiguous:
                if previous is None:
                    session.people[key] = PersonRecord(
                        key=key, label=subject.label, kind="named",
                        relation=None, owner=subject.owner, gender=None,
                        first_message_id=source.message_id,
                        last_message_id=source.message_id)
                elif previous.last_message_id != source.message_id:
                    session.people[key] = replace(
                        previous, last_message_id=source.message_id,
                        mentions=previous.mentions + 1)
                session.last_name_target = key
                session.last_name_message_id = source.message_id

        if subject.status in _UNRESOLVED_STATUSES:
            abstentions.append(f"unresolved_subject:{fact.fact_id}")

        resolved = fact
        if subject != fact.subject:
            resolved = replace(fact, subject=subject)
        facts.append(resolved)

        attributions.append(Attribution(
            speaker=speaker,
            subject=subject,
            status=subject.status,
            evidence=_attribution_evidence(resolved, subject, speaker),
            fact_id=fact.fact_id,
            scope="conversation",
            referenced=(subject_key(subject),) if subject_key(subject) else (),
        ))
        confidence_terms.append(subject.confidence)

        if subject.status == "resolved":
            session.recent.append(_RecentSubject(
                subject=subject, message_id=source.message_id,
                fact_id=fact.fact_id))

    if not facts:
        known = speaker.id != "unknown"
        default_subject = SubjectRef(
            kind="speaker" if known else "unknown",
            label=speaker.label, confidence=speaker.confidence,
            owner=speaker.id if known else None,
            resolved_from="message_speaker",
            status="resolved" if known else "unknown",
            ambiguity=None if known else "speaker_missing",
        )
        attributions.append(Attribution(
            speaker=speaker, subject=default_subject,
            status=default_subject.status,
            evidence=("message_level",), scope="conversation"))
        confidence_terms.append(default_subject.confidence)

    session.observe_order(source.order)
    session.last_message_id = source.message_id
    mean = (sum(confidence_terms) / len(confidence_terms)
            if confidence_terms else 0.0)
    return IdentityResult(
        facts=tuple(facts),
        attributions=tuple(attributions),
        transitions=tuple(transitions),
        ambiguities=tuple(dict.fromkeys(ambiguities)),
        abstentions=tuple(dict.fromkeys(abstentions)),
        confidence=round(mean, 4),
    )


__all__ = [
    "IdentityRegistry",
    "IdentityResult",
    "IdentitySession",
    "PersonRecord",
    "resolve_identity",
    "subject_key",
]
