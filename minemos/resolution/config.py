"""Configuration for the semantic resolution layer.

The layer is local-first.  No provider is contacted unless the host passes
one explicitly, and the built-in providers are deterministic: a hash-based
lexical similarity provider (always available) and an optional wrapper around
the engine's existing local sentence encoder.  There is no bundled network
client and no default endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

MODES = ("disabled", "shadow", "active")

ACTIVE_WARNING = (
    "active mode feeds resolution candidates to the engine; the engine remains "
    "the memory authority and keeps its own consolidation, contradiction, "
    "temporal and retention decisions"
)


@dataclass(frozen=True)
class ResolutionConfig:
    """Immutable configuration for one resolver instance.

    ``mode`` is ``"shadow"`` by default: the layer processes conversations and
    produces records, but nothing is written to canonical memory.  ``"active"``
    opt-in is only meaningful for callers that pass records to the engine
    bridge; the resolver itself never mutates memory in either mode.
    """

    mode: str = "shadow"

    # Stage 1 — noise normalization
    normalize_typos: bool = True
    normalize_contractions: bool = True
    normalize_informal: bool = True
    normalize_whitespace: bool = True
    max_typo_edits: int = 1
    # Structural words (verbs and attribute names) may be recovered one edit
    # away; values and names are never recovered this way.
    max_structural_edits: int = 1
    typo_min_length: int = 3
    typo_confidence: float = 0.82
    preserve_capitalized: bool = True
    detect_language: bool = True

    # Stage 2 — semantic interpretation
    extract_facts: bool = True
    max_facts_per_message: int = 12
    extraction_confidence: float = 0.62
    expand_place_aliases: bool = True

    # Stage 3 — identity and provenance
    resolve_pronouns: bool = True
    pronoun_window: int = 6
    conversation_scope_only: bool = True
    link_repeated_variants: bool = True
    repeated_variant_support: int = 2
    name_variant_max_distance: int = 2

    # Stage 4 — fact relationships
    duplicate_similarity: float = 0.86
    ambiguous_similarity: float = 0.62
    semantic_provider: str = "hash"  # "hash" | "local" | "none"

    # Diagnostics
    include_spans: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"mode must be one of {MODES}, got {self.mode!r}")
        if self.max_typo_edits < 0:
            raise ValueError("max_typo_edits cannot be negative")
        if self.max_structural_edits < 0:
            raise ValueError("max_structural_edits cannot be negative")
        if self.typo_min_length < 2:
            raise ValueError("typo_min_length must be at least 2")
        for name in ("typo_confidence", "extraction_confidence",
                     "duplicate_similarity", "ambiguous_similarity"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.ambiguous_similarity >= self.duplicate_similarity:
            raise ValueError(
                "ambiguous_similarity must be below duplicate_similarity")
        if self.pronoun_window < 0:
            raise ValueError("pronoun_window cannot be negative")
        if self.semantic_provider not in ("hash", "local", "none"):
            raise ValueError(
                "semantic_provider must be 'hash', 'local' or 'none'")

    @property
    def shadow(self) -> bool:
        return self.mode == "shadow"

    @property
    def active(self) -> bool:
        return self.mode == "active"

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def with_mode(self, mode: str) -> "ResolutionConfig":
        return replace(self, mode=mode)


DEFAULT_CONFIG = ResolutionConfig()


__all__ = ["ACTIVE_WARNING", "DEFAULT_CONFIG", "MODES", "ResolutionConfig"]
