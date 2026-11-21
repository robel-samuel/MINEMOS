# MINEMOS Documentation

MINEMOS is a persistent memory engine for AI applications. It stores
observations in a fixed-capacity canonical memory, consolidates repeated facts,
records supersession and temporal validity, evicts under pressure, and retrieves
relevant state. Qdrant can serve as a derived retrieval index; MINEMOS remains
the memory authority.

Start with the [README](../README.md) for the product overview and quick start,
or [Getting Started](getting-started.md) to run the engine.

## Learn and use

- [Getting Started](getting-started.md) — install, absorb, query, look back in
  time, persist, and restore.
- [Python API](python-api.md) — every public engine method, its result shape,
  and the answer presentation contract.
- [Integration](integration.md) — adding memory to a chat assistant, agent, or
  multi-user application.
- [Data Ingestion](guides/data-ingestion.md) — turning conversations, documents,
  and records into observations.

## Architecture

- [Overview](architecture/overview.md) — components, boundaries, data flow, and
  where state lives.
- [Memory Lifecycle](architecture/memory-lifecycle.md) — representation, absorb
  decisions, consolidation, contradiction and supersession, retention, and
  checkpoints.
- [Retrieval](architecture/retrieval.md) — canonical retrieval, hybrid
  candidates, filters, snapshot backends, and fallback.
- [Qdrant Integration](architecture/qdrant-integration.md) — deployment modes,
  index lifecycle, tenancy, and failure isolation.
- [Semantic Resolution Layer](architecture/semantic-resolution.md) —
  candidate-only normalization, fact interpretation, identity resolution and
  relationship proposals, run in shadow mode in front of the engine.

## Guides

- [Installation](guides/installation.md) — Python 3.12 setup, CPU PyTorch,
  models, smoke test, and optional Qdrant.
- [Configuration](guides/configuration.md) — engine parameters, retrieval modes,
  and environment variables.
- [Releasing](guides/releasing.md) — Trusted Publishing setup and the PyPI
  release workflow.

## Operations

- [Recovery](operations/recovery.md) — checkpoint save/load, backups, privacy
  deletion, Qdrant rebuild, and canonical rollback.
- [Troubleshooting](operations/troubleshooting.md) — common errors and fixes.

## Reading paths

| Goal | Read |
|---|---|
| Try the engine | [Getting Started](getting-started.md) |
| Add memory to an application | [Integration](integration.md) |
| Understand memory decisions | [Memory Lifecycle](architecture/memory-lifecycle.md) |
| Learn the API | [Python API](python-api.md) |
| Use the resolution layer | [Semantic Resolution Layer](architecture/semantic-resolution.md) |
| Enable Qdrant | [Configuration](guides/configuration.md) then [Qdrant Integration](architecture/qdrant-integration.md) |
| Recover or erase state | [Recovery](operations/recovery.md) |
| Contribute | [CONTRIBUTING.md](../CONTRIBUTING.md) |

## Documentation conventions

- Behavior is described from source code.
- Examples use the supported `minemos` import path and placeholder credentials.
- Optional components are labeled as optional.
- Configuration examples never contain real credentials.