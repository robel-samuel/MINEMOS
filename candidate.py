"""
Shared interface between extraction layers and the memory layer.

Any encoder (RuleBasedExtractor, ConversationExtractor, SemanticEncoder,
future encoders) implements extract_sentence(text) -> Optional[Candidate]
and extract_conversation(sentences) -> list[Candidate]. MemoryStore never
needs to know which one produced a candidate.
"""

from __future__ import annotations

from typing import Optional


class Candidate:
    """
    A single proposed (subject, predicate, object) fact extracted from
    text, with optional confidence and subject override.

    confidence=None means "let MemoryStore use its own default" -- this is
    what the rule-based extractors do. Encoders that model uncertainty
    (e.g. SemanticEncoder) set an explicit value.

    subject=None means "the caller already knows the subject" (e.g. the
    conversation benchmark passes in person_i explicitly). Some encoders
    infer the subject from the sentence itself (e.g. "my coworker uses
    Python" implies subject=coworker, not subject=user) and set it here.
    """

    def __init__(self, predicate: str, object_: str,
                 confidence: Optional[float] = None,
                 subject: Optional[str] = None,
                 negated: bool = False):
        self.predicate = predicate
        self.object = object_
        self.confidence = confidence
        self.subject = subject
        self.negated = negated

    def __repr__(self):
        conf = f", conf={self.confidence}" if self.confidence is not None else ""
        subj = f", subj={self.subject}" if self.subject is not None else ""
        return f"Candidate({self.predicate}={self.object!r}{conf}{subj})"


class BaseExtractor:
    """Interface every extractor implements."""

    def extract_sentence(self, sentence: str) -> Optional[Candidate]:
        raise NotImplementedError

    def extract_conversation(self, sentences: list[str]) -> list[Candidate]:
        out = []
        for s in sentences:
            c = self.extract_sentence(s)
            if c is not None:
                out.append(c)
        return out
