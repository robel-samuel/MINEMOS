"""
Extractor for the realistic-conversation benchmark.

Maps the paraphrased sentence templates from conversation_generator.py to
canonical predicates, so that different wordings of the same underlying
fact absorb into the same (subject, predicate) slot and can reinforce/
consolidate rather than each creating a separate memory unit.

Same honesty note as core/encoder.py: this is pattern-matching tuned to
the generator's template vocabulary, not general-purpose NLP. It exists
to test the memory/consolidation layer in isolation from extraction
quality (see conversation_generator.py's module docstring).
"""

from __future__ import annotations

import re
from typing import Optional

from core.candidate import Candidate, BaseExtractor


# Fact is kept as an alias so existing benchmark code that imports Fact
# and does .predicate / .object keeps working unchanged.
Fact = Candidate


class Fact:
    def __init__(self, predicate: str, object_: str):
        self.predicate = predicate
        self.object = object_

    def __repr__(self):
        return f"Fact({self.predicate}={self.object!r})"


# Each pattern maps to a canonical predicate. Multiple patterns map to the
# same predicate on purpose -- that's the paraphrase-consolidation test.
PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^I use (?P<v>.+?) for backend development\.?$", re.I), "backend_language"),
    (re.compile(r"^(?P<v>.+?) is my preferred language for backend work\.?$", re.I), "backend_language"),
    (re.compile(r"^My API projects normally use (?P<v>.+?)\.?$", re.I), "backend_language"),
    (re.compile(r"^I built my latest backend using (?P<v>.+?)\.?$", re.I), "backend_language"),
    (re.compile(r"^Most of my backend code is written in (?P<v>.+?)\.?$", re.I), "backend_language"),

    (re.compile(r"^I'm using (?P<v>.+?) as the framework\.?$", re.I), "framework"),
    (re.compile(r"^The project uses (?P<v>.+?) for the database\.?$", re.I), "database"),
    (re.compile(r"^The application is deployed on (?P<v>.+?)\.?$", re.I), "deploy_platform"),
    (re.compile(r"^The backend depends on (?P<v>.+?) being available\.?$", re.I), "depends_on"),
    (re.compile(r"^Deployment failed once because of (?P<v>.+?)\.?$", re.I), "known_failure"),
    (re.compile(r"^I decided that deployments should always (?P<v>.+?)\.?$", re.I), "lesson"),

    # Continual-learning / temporal test sentences (Task 3).
    (re.compile(r"^I prefer (?P<v>.+?)\.?$", re.I), "prefers_language"),
    (re.compile(r"^I still prefer (?P<v>.+?)\.?$", re.I), "prefers_language"),
    (re.compile(r"^I switched to (?P<v>.+?)\.?$", re.I), "prefers_language"),
]


class ConversationExtractor(BaseExtractor):
    def extract_sentence(self, sentence: str) -> Optional[Fact]:
        sentence = sentence.strip()
        for pattern, predicate in PATTERNS:
            m = pattern.match(sentence)
            if m:
                return Fact(predicate, m.group("v").strip().rstrip("."))
        return None

    def extract_conversation(self, sentences: list[str]) -> list[Fact]:
        facts = []
        for s in sentences:
            f = self.extract_sentence(s)
            if f is not None:
                facts.append(f)
        return facts
