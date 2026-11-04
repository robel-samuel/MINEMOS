"""MINEMOS controlled-deployment service layer.

This package is a thin application boundary around the existing memory engine.
It does not change consolidation, retention, contradiction handling, temporal
semantics, confidence, supersession, redaction or canonical persistence.  It
adds authentication, authorization, tenant isolation, request validation,
rate limiting, provenance, lifecycle management and a small HTTP surface.

The canonical ``.npz`` checkpoint remains the authoritative state.  Qdrant is
only a derived retrieval index and is never treated as the source of truth.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
