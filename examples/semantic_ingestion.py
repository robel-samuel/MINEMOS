"""Semantic ingestion end to end: conversation -> facts -> restart -> retrieval.

Run from the repository root::

    python examples/semantic_ingestion.py

The example is self-contained: it uses a temporary checkpoint directory, a
synthetic conversation, and the built-in heuristic extractor.  It performs no
network calls and writes nothing outside the temporary directory.

Semantic ingestion is opt-in.  ``ResolutionConfig(mode="active")`` is the
explicit switch; without it the ingestor refuses to run rather than silently
writing nothing.  Extraction quality is not guaranteed and has not been
validated against independent human labels.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from minemos import (
    BlackHoleMemory,
    EngineSink,
    ResolutionConfig,
    SemanticIngestor,
    SourceMessage,
)


def conversation() -> list:
    return [
        SourceMessage(
            conversation_id="demo-conversation",
            message_id="m-1",
            text="My name is Ada and I work at Acme Corp.",
            speaker="user",
            timestamp=1_730_000_000.0,
            order=0,
        ),
        SourceMessage(
            conversation_id="demo-conversation",
            message_id="m-2",
            text="I live in Berlin.",
            speaker="user",
            timestamp=1_730_000_060.0,
            order=1,
        ),
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="minemos-example-") as temporary:
        checkpoint = Path(temporary) / "memory.npz"

        memory = BlackHoleMemory(capacity=1000)
        ingestor = SemanticIngestor(
            EngineSink(memory),
            config=ResolutionConfig(mode="active"),
            ingestion_timestamp=1_730_000_060.0,
        )
        report = ingestor.ingest_conversation(conversation())
        print("ingestion:", report.summary())

        rows = memory.recall("user")
        print("recalled:", [row["text"] for row in rows])

        rows = memory.query("Where does the user work?", top_k=2)
        print("retrieved:", [row["text"] for row in rows])

        memory.save_checkpoint(str(checkpoint))
        memory.close()

        restored = BlackHoleMemory(capacity=1000)
        restored.load_checkpoint(str(checkpoint))
        print("after restart:",
              [row["text"] for row in restored.recall("user")])
        restored.close()


if __name__ == "__main__":
    main()
