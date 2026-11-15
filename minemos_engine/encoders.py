"""
Encoder abstraction for structured candidates.

AtomicEncoder: the reference encoding. Each field is one atomic token.
Wraps structured_candidate.encode_full directly -- not reimplemented, so
the reference encoding cannot drift.

FieldAwareLexicalEncoder: richer representation. Each field is WORD-split
(not char-split), and every token is prefixed with its field name
("subject:", "predicate:", "object:") so cross-field bleed ("person"
leaking between subject and object) cannot occur. Multi-word fields
contribute partial overlap when they share literal words, which
single-atomic-token encoding cannot do. This is deterministic and not
tuned to any test sentence: the same generic word-splitting rule applies
to every field, whatever it says.

TfidfLexicalEncoder: a lexical-statistical baseline. Kept for comparison,
explicitly labeled non-semantic.

SemanticEmbeddingEncoder: a pretrained static embedding model (spaCy
en_core_web_md, GloVe-style word vectors). See its own docstring for
provenance and known limitations.
"""

from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from minemos_engine.structured_candidate import Candidate, encode_full as _atomic_encode_full, DIM


class AtomicEncoder:
    """The reference encoding: one atomic token per field."""

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
    Wraps a pretrained embedding model -- spaCy's en_core_web_md, which
    uses static GloVe-style word vectors (300-dim, averaged over document
    tokens for a document vector via spaCy's .vector attribute).

    PROVENANCE: the model is en_core_web_md v3.7.1, installed from its
    published GitHub release wheel. It is a genuine pretrained embedding
    model -- not TF-IDF, not feature hashing, not a lexical method.

    KNOWN LIMITATION: static/averaged word embeddings are well documented
    to conflate synonyms and antonyms, because they encode "words that
    appear in similar contexts" rather than "words with the same meaning
    polarity" -- e.g. "like" and "hate" both commonly appear in sentences
    of the form "I really ___ coffee," so their vectors end up close
    together despite opposite meaning. This is a property of the entire
    word2vec/GloVe embedding class, not a bug specific to this model or
    this implementation.

    REPRESENTATION FORMAT: raw natural-language concatenation,
    "{subject} {predicate} {object}", applied identically to every
    candidate. A labeled template ("subject: {s} predicate: {p}
    object: {o}") was measured and rejected: because every candidate's
    text contained the identical literal words "subject:"/"predicate:"/
    "object:", the averaged document vector was dominated by that shared
    boilerplate -- the same unrelated pair scored 0.94 with the label
    template versus 0.37 without it.
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
