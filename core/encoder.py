"""
Encoder / extractor stub (Section 10).

Phase 1 does not need a trained encoder — it needs *a* pluggable interface
so the memory/update logic can be tested independently of extraction
quality. This module ships two extractors:

  - RuleBasedExtractor: tiny pattern matcher, good enough for synthetic
    benchmark sentences like "Person 3 uses Python." (Section 45).
  - LLMExtractor: stub showing where a real structured-extraction call
    (e.g. an Anthropic API call asking for strict JSON) would plug in.

Swap extractors without touching MemoryStore.
"""

from __future__ import annotations

import re
from typing import Optional


class Triple:
    def __init__(self, subject: str, predicate: str, object_: str):
        self.subject = subject
        self.predicate = predicate
        self.object = object_

    def __repr__(self):
        return f"({self.subject}, {self.predicate}, {self.object})"


class RuleBasedExtractor:
    """
    Minimal pattern-based extractor for Phase 1 synthetic tests.
    Not meant to generalize — only to unblock testing MemoryStore.
    """

    PATTERNS = [
        (re.compile(r"^(?P<subj>[\w\s]+?)\s+uses\s+(?P<obj>.+?)\.?$", re.I),
         "uses"),
        (re.compile(r"^(?P<subj>[\w\s]+?)\s+likes\s+(?P<obj>.+?)\.?$", re.I),
         "likes"),
        (re.compile(r"^(?P<subj>[\w\s]+?)\s+prefers\s+(?P<obj>.+?)\.?$", re.I),
         "prefers"),
        (re.compile(r"^(?P<subj>[\w\s]+?)\s+lives in\s+(?P<obj>.+?)\.?$", re.I),
         "lives_in"),
        (re.compile(r"^(?P<subj>[\w\s]+?)\s+owns\s+(?P<obj>.+?)\.?$", re.I),
         "owns"),
        (re.compile(r"^(?P<subj>[\w\s]+?)\s+now prefers\s+(?P<obj>.+?)\.?$", re.I),
         "prefers"),
        (re.compile(r"^(?P<subj>[\w\s]+?)\s+deployed .* on\s+(?P<obj>.+?)\.?$", re.I),
         "deployed_on"),
    ]

    def extract(self, text: str) -> Optional[Triple]:
        text = text.strip()
        for pattern, predicate in self.PATTERNS:
            m = pattern.match(text)
            if m:
                subj = m.group("subj").strip().lower().replace(" ", "_")
                obj = m.group("obj").strip()
                return Triple(subj, predicate, obj)
        return None


class LLMExtractor:
    """
    Placeholder for a real structured-extraction call. Not wired to a live
    API in this prototype — swap in a call to /v1/messages requesting
    strict JSON output of {subject, predicate, object} when ready
    (see Section 30, "SIMPLE SDK").
    """

    def extract(self, text: str) -> Optional[Triple]:
        raise NotImplementedError(
            "Wire this up to a real structured-extraction call when moving "
            "past synthetic-data testing."
        )
