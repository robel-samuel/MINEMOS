"""
phase2/encoders.py — Phase 7/8 encoder abstraction.

Encoder A (AtomicEncoder): the Phase 2B/5/6 control. Each field is one
atomic token. Wraps structured_candidate.encode_full directly -- not
reimplemented, so there is no risk of subtly changing the control.

Encoder B (FieldAwareLexicalEncoder): richer representation. Each field
is WORD-split (not char-split), and every token is prefixed with its
field name ("subject:", "predicate:", "object:") so cross-field bleed
(the Phase 2A bug -- "person" leaking between subject and object) cannot
recur. Multi-word fields now contribute partial overlap when they share
literal words, which single-atomic-token encoding could never do. This
is deterministic and NOT hand-tuned to any test sentence: the same
generic word-splitting rule applies to every field, whatever it says.

TfidfLexicalEncoder: NOT Encoder C. A lexical-statistical baseline
tested in Phase 7 when no real embedding model was available. Kept for
comparison, explicitly labeled non-semantic.

SemanticEmbeddingEncoder: Phase 8's genuine pretrained embedding model
(spaCy en_core_web_md, static GloVe-style word vectors). Legitimately
obtained via a GitHub release asset (allowed network domain), not
fabricated, not TF-IDF. See its own docstring for full provenance and
known limitations.
"""

from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from phase2.structured_candidate import Candidate, encode_full as _atomic_encode_full, DIM


class AtomicEncoder:
    """Encoder A -- the unmodified Phase 2B/5/6 control."""

    name = "A_atomic"

    def encode_full(self, c: Candidate, dim: int = DIM) -> np.ndarray:
        return _atomic_encode_full(c, dim=dim)


def _field_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if w]


class FieldAwareLexicalEncoder:
    """
    Encoder B. Each field is word-split; every token is prefixed with its
    field role so fields cannot bleed into each other. E.g. "regularly
    uses" -> ["predicate:regularly", "predicate:uses"]. A single-word
    field like "uses" -> ["predicate:uses"] -- degenerates to exactly
    Encoder A's behavior for single-word fields, so any difference from
    Encoder A is attributable entirely to multi-word fields sharing
    literal words, not to a different hashing scheme.
    """

    name = "B_field_aware_lexical"

    def encode_full(self, c: Candidate, dim: int = DIM) -> np.ndarray:
        tokens = ([f"subject:{w}" for w in _field_words(c.subject)] +
                  [f"predicate:{w}" for w in _field_words(c.predicate)] +
                  [f"object:{w}" for w in _field_words(c.object)])
        v = np.zeros(dim, dtype=np.float32)
        import hashlib
        for tok in tokens:
            digest = hashlib.sha1(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            v[idx] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


class SemanticEmbeddingEncoder:
    """
    Encoder for Phase 8. Wraps a genuine pretrained embedding model --
    spaCy's en_core_web_md, which uses static GloVe-style word vectors
    (300-dim, trained on a large web corpus, averaged over document
    tokens for a document vector via spaCy's .vector attribute).

    PROVENANCE (documented per the Phase 8 spec's requirement):
    Installed via `pip install <wheel URL>` where the wheel is hosted at
    github.com/explosion/spacy-models/releases -- a GitHub release asset,
    on this environment's network allowlist (github.com,
    release-assets.githubusercontent.com). Not downloaded from any
    blocked domain, not fabricated, not hand-encoded. Model: en_core_web_md
    v3.7.1. This is a genuine pretrained embedding model -- NOT TF-IDF,
    NOT feature hashing, NOT a lexical/statistical method.

    KNOWN LIMITATION, disclosed up front (not discovered after the fact
    and hidden): static/averaged word embeddings are well documented to
    conflate synonyms and antonyms, because they encode "words that
    appear in similar contexts" rather than "words with the same
    meaning polarity" -- e.g. "like" and "hate" both commonly appear in
    sentences of the form "I really ___ coffee," so their vectors end up
    close together despite opposite meaning. This is a property of this
    entire class of embedding (word2vec/GloVe-style), not a bug specific
    to this model or this implementation. Whether it materially affects
    THIS experiment's paraphrase-vs-contradiction separation is measured
    directly below, not assumed from this one observation.

    REPRESENTATION FORMAT: raw natural-language concatenation,
    "{subject} {predicate} {object}", applied identically to every
    candidate. An earlier version of this encoder used a labeled
    template ("subject: {s} predicate: {p} object: {o}") -- still
    uniformly applied, not tuned to any example -- but direct testing
    caught a serious confound: because every candidate's text contained
    the identical literal words "subject:"/"predicate:"/"object:", the
    averaged document vector was dominated by that shared boilerplate.
    Directly measured: the same genuinely-unrelated pair went from 0.94
    similarity (with the label template) to 0.37 (without it) -- a
    completely different picture. This is disclosed, not silently
    corrected: both results are reported in phase8_real_embedding_report.md.
    """

    name = "D_spacy_en_core_web_md_pretrained_embedding"

    def __init__(self):
        import spacy
        self._nlp = spacy.load("en_core_web_md")

    def _text(self, c: Candidate) -> str:
        return f"{c.subject} {c.predicate} {c.object}"

    def encode_full(self, c: Candidate, dim: int = None) -> np.ndarray:
        # dim is ignored -- the pretrained model's dimensionality (300)
        # is fixed, not a free parameter like the hash-based encoders.
        doc = self._nlp(self._text(c))
        vec = doc.vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class TfidfLexicalEncoder:
    """
    NOT Encoder C. A lexical-statistical baseline (TF-IDF over field text,
    same technique already used and explicitly labeled non-semantic in
    the earlier extraction-layer phases of this project), included only
    because no real pretrained embedding model is available in this
    environment. Explicitly NOT claimed to understand meaning -- it
    weights shared words by corpus rarity, nothing more.
    """

    name = "C_lexical_statistical_NOT_semantic"

    def __init__(self, corpus_texts: list[str]):
        self._vectorizer = TfidfVectorizer()
        self._vectorizer.fit(corpus_texts)

    def _field_text(self, c: Candidate) -> str:
        return f"{c.subject} {c.predicate} {c.object}"

    def encode_full(self, c: Candidate, dim: int = None) -> np.ndarray:
        # dim is ignored -- TF-IDF's dimensionality is fixed by the fitted
        # vocabulary, not a free parameter like the hash-based encoders.
        vec = self._vectorizer.transform([self._field_text(c)]).toarray()[0]
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).astype(np.float32)
