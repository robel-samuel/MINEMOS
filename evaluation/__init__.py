"""Dedicated real-world evaluation environment for MINEMOS.

The package separates three concerns:

* :mod:`evaluation.formats` parses supported source formats and preserves
  document/conversation boundaries and original timestamps;
* :mod:`evaluation.store` owns the dedicated, persistent evaluation directory:
  manifest, canonical checkpoints, provenance, source segments, query files,
  independent ground truth, and versioned results;
* :mod:`evaluation.ingest` and :mod:`evaluation.run` are the operator CLIs.

Expected answers are never written into the engine, its checkpoint, or the
provenance ledger.  Nothing here claims a 30-day run has happened; the package
only makes the environment ready to start one.
"""
