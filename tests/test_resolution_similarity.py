"""Contract tests for the Stage 4 similarity providers.

``hash`` and ``none`` are self-contained.  ``local`` must reuse the engine's
dense encoder when it is importable, reuse one encoder per process, fall back
to ``hash`` without touching the network when it is not, and report which
provider actually ran.
"""

from __future__ import annotations

import sys
import types

import pytest

from minemos.resolution import ResolutionConfig, Resolver, SourceMessage
from minemos.resolution import similarity as sim
from minemos.resolution.similarity import (
    ExactSimilarity,
    HashSimilarity,
    LocalEncoderSimilarity,
    build_similarity,
)

STAMP = 1_700_000_000.0


class FakeDenseEncoder:
    """Stand-in for the engine encoder; records how many times it loaded."""

    instances = 0

    def __init__(self) -> None:
        type(self).instances += 1

    def encode(self, text: str):
        return {
            "acme corp": [1.0, 0.0],
            "acme corporation": [1.0, 0.0],
            "berlin": [0.0, 1.0],
        }.get(text, [0.0, 0.0])


@pytest.fixture(autouse=True)
def reset_local_cache(monkeypatch):
    monkeypatch.setattr(sim, "_LOCAL_ENCODER", None)
    monkeypatch.setattr(sim, "_LOCAL_ENCODER_FAILED", False)
    FakeDenseEncoder.instances = 0


def install_fake_engine(monkeypatch):
    module = types.ModuleType("minemos_engine.intelligence.dense")
    module.DenseSemanticEncoder = FakeDenseEncoder
    monkeypatch.setitem(sys.modules, "minemos_engine.intelligence.dense",
                        module)


def _message(text: str, message_id: str = "m1") -> SourceMessage:
    return SourceMessage(
        conversation_id="conv-sim", message_id=message_id, text=text,
        speaker="user", timestamp=STAMP, order=0)


def test_build_similarity_selects_each_provider():
    assert isinstance(build_similarity(ResolutionConfig(semantic_provider="hash")),
                      HashSimilarity)
    assert isinstance(build_similarity(ResolutionConfig(semantic_provider="none")),
                      ExactSimilarity)
    assert isinstance(build_similarity(ResolutionConfig(semantic_provider="local")),
                      LocalEncoderSimilarity)


def test_hash_provider_is_deterministic_and_bounded():
    provider = HashSimilarity()
    first = provider("Acme Corp", "Acme Corporation")
    assert first == provider("Acme Corp", "Acme Corporation")
    assert 0.0 <= first <= 1.0
    assert provider("Acme Corp", "Berlin") < first


def test_none_provider_matches_only_normalized_equality():
    provider = ExactSimilarity()
    assert provider("Acme Corp.", "acme   corp") == 1.0
    assert provider("Acme Corp", "Acme Corporation") == 0.0


def test_local_provider_uses_the_engine_encoder(monkeypatch):
    install_fake_engine(monkeypatch)
    provider = LocalEncoderSimilarity()
    assert provider("Acme Corp.", "Acme Corporation") == 1.0
    assert provider("Acme Corp.", "Berlin") == 0.0
    assert provider.name == "local"


def test_local_encoder_loads_once_per_process(monkeypatch):
    install_fake_engine(monkeypatch)
    first = LocalEncoderSimilarity()
    second = LocalEncoderSimilarity()
    assert first("Acme Corp", "Acme Corporation") == 1.0
    assert second("Acme Corp", "Berlin") == 0.0
    assert FakeDenseEncoder.instances == 1


def test_local_provider_falls_back_to_hash_without_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "minemos_engine.intelligence.dense", None)
    provider = LocalEncoderSimilarity()
    left, right = "Acme Corp", "Acme Corporation"
    assert provider.name == "hash"
    assert provider(left, right) == HashSimilarity()(left, right)


def test_record_provider_matches_the_provider_that_ran(monkeypatch):
    install_fake_engine(monkeypatch)
    local = Resolver(ResolutionConfig(mode="shadow",
                                      semantic_provider="local"))
    assert local.resolve(_message("I work at Acme Corp.")).record.provider == (
        "deterministic:local")

    monkeypatch.setitem(sys.modules, "minemos_engine.intelligence.dense", None)
    monkeypatch.setattr(sim, "_LOCAL_ENCODER", None)
    monkeypatch.setattr(sim, "_LOCAL_ENCODER_FAILED", False)
    fallback = Resolver(ResolutionConfig(mode="shadow",
                                         semantic_provider="local"))
    assert fallback.resolve(_message(
        "I work at Acme Corp.")).record.provider == "deterministic:hash"
