# MINEMOS Documentation

MINEMOS is an experimental memory system for AI applications. It stores
observations in a fixed-capacity canonical memory, consolidates repeated facts,
records supersession and temporal validity, evicts under pressure, and retrieves
relevant state. Qdrant can serve as a derived retrieval index, but MINEMOS remains
the memory authority.

Start with the [README](../README.md) for the project overview and quick start.

## Architecture

- [Overview](architecture/overview.md) — components, boundaries, data flow, and
  where state lives.
- [Memory Lifecycle](architecture/memory-lifecycle.md) — representation,
  absorb decisions, consolidation, contradiction/supersession, retention, and
  checkpoints.
- [Retrieval](architecture/retrieval.md) — canonical retrieval, hybrid
  candidates, filters, snapshot backends, and fallback.
- [Qdrant Integration](architecture/qdrant-integration.md) — deployment modes,
  production rules, index lifecycle, tenancy, and failure isolation.

## Guides

- [Installation](guides/installation.md) — Python 3.12 setup, CPU PyTorch,
  models, smoke test, and optional Qdrant.
- [Configuration](guides/configuration.md) — engine parameters, retrieval
  modes, environment variables, and production Qdrant rules.
- [Deployment](guides/deployment.md) — profiles, topology, checklist, scale
  expectations, and rollback.

## Research

- [Research overview](research/README.md) — program, artifact map, and scope.
- [Phase history](research/history.md) — what each phase changed and measured.
- [Benchmark methodology and results](research/benchmark-methodology.md) —
  failure schema, synthetic results, Qdrant tables, RRF experiment, and caveats.
- [`reports/`](../reports/) and [`benchmarks/results/`](../benchmarks/results/)
  — preserved historical reports and raw results.

## Operations

- [Monitoring](operations/monitoring.md) — status fields, retrieval state, logs,
  and capacity checks.
- [Recovery](operations/recovery.md) — checkpoint save/load, backups, Qdrant
  rebuild, and canonical rollback.
- [Troubleshooting](operations/troubleshooting.md) — common errors and fixes.

## Reading paths

| Goal | Read |
|---|---|
| Try the engine | [Installation](guides/installation.md) then [README quick start](../README.md#quick-start) |
| Understand memory decisions | [Memory Lifecycle](architecture/memory-lifecycle.md) |
| Enable Qdrant | [Configuration](guides/configuration.md) then [Qdrant Integration](architecture/qdrant-integration.md) |
| Deploy cautiously | [Deployment](guides/deployment.md) and [Recovery](operations/recovery.md) |
| Review evidence | [Phase history](research/history.md) and [Benchmark Methodology](research/benchmark-methodology.md) |
| Contribute | [CONTRIBUTING.md](../CONTRIBUTING.md) |

## Documentation conventions

- Implemented behavior is described from source code.
- Experimental or opt-in behavior is labeled as such.
- Historical results are reported as measured, with their original caveats.
- Planned work is not described as implemented.
- Configuration examples use placeholders, never real credentials.
