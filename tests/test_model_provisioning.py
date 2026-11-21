"""First-use model provisioning and lazy engine initialization.

These tests use fake factories: they prove the cache-first, download-once,
offline-error and lazy-load behaviour without touching the network.  The real
download and cache reuse are exercised by the clean-environment installation
acceptance run.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from minemos_engine import model_provisioning
from minemos_engine.intelligence import dense, nli


@pytest.fixture(autouse=True)
def isolated_provisioning(monkeypatch):
    model_provisioning.clear_loaded_components()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 model_provisioning.CACHE_DIR_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    model_provisioning.clear_loaded_components()


class FakeComponent:
    def __init__(self):
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_cached_model_is_loaded_without_network(monkeypatch):
    calls = []

    def factory(model_name, **kwargs):
        calls.append(kwargs)
        return FakeComponent()

    model_provisioning._load("tokenizer", "org/model", factory)
    assert calls == [{"cache_dir": None, "local_files_only": True}]


def test_missing_model_downloads_once_then_reuses_component(monkeypatch):
    calls = []

    def factory(model_name, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return FakeComponent()

    first = model_provisioning._load(
        "sentence-embedding", "org/model", factory, device="cpu")
    second = model_provisioning._load(
        "sentence-embedding", "org/model", factory, device="cpu")

    assert first is second
    assert [call.get("local_files_only") for call in calls] == [True, None]
    assert first.eval_called is True


def test_offline_mode_never_tries_the_network(monkeypatch):
    calls = []

    def factory(model_name, **kwargs):
        calls.append(kwargs)
        raise OSError("not in cache")

    with pytest.raises(model_provisioning.ModelProvisioningError) as error:
        model_provisioning._load(
            "NLI", "org/model", factory, local_files_only=True)

    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    message = str(error.value)
    assert "org/model" in message
    assert "HF_HUB_OFFLINE" in message
    assert "TRANSFORMERS_OFFLINE" in message


def test_offline_environment_variable_is_honoured(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert model_provisioning.offline_mode() is True
    calls = []

    def factory(model_name, **kwargs):
        calls.append(kwargs)
        raise OSError("not in cache")

    with pytest.raises(model_provisioning.ModelProvisioningError):
        model_provisioning._load("NLI", "org/model", factory)
    assert calls == [{"cache_dir": None, "local_files_only": True}]


def test_download_failure_message_is_actionable(monkeypatch):
    def factory(model_name, **kwargs):
        raise OSError("network down")

    with pytest.raises(model_provisioning.ModelProvisioningError) as error:
        model_provisioning._load("NLI", "org/model", factory)

    message = str(error.value)
    assert "Could not download" in message
    assert "org/model" in message
    assert model_provisioning.CACHE_DIR_ENV in message


def test_explicit_cache_dir_and_environment(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    calls = []
    model_provisioning._load(
        "tokenizer", "org/model",
        lambda name, **kw: calls.append(kw) or FakeComponent(),
        cache_dir=str(explicit))
    assert calls == [{"cache_dir": str(explicit.resolve()),
                      "local_files_only": True}]
    assert str(explicit.resolve()) == model_provisioning.resolve_cache_dir(
        str(explicit))

    configured = tmp_path / "configured"
    monkeypatch.setenv(model_provisioning.CACHE_DIR_ENV, str(configured))
    assert model_provisioning.resolve_cache_dir() == str(configured.resolve())


def test_explicit_cache_dir_wins_over_environment(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv(model_provisioning.CACHE_DIR_ENV, str(configured))
    assert model_provisioning.resolve_cache_dir(str(explicit)) == str(
        explicit.resolve())


def test_offline_mode_loads_a_cached_model(monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "yes")
    calls = []

    def factory(model_name, **kwargs):
        calls.append(kwargs)
        return FakeComponent()

    component = model_provisioning._load("tokenizer", "org/model", factory)
    assert component is not None
    assert calls == [{"cache_dir": None, "local_files_only": True}]


def test_download_failure_is_retried_and_not_cached():
    attempts = []

    def factory(model_name, **kwargs):
        attempts.append(kwargs.get("local_files_only"))
        if len(attempts) < 3:
            raise OSError("temporary failure")
        return FakeComponent()

    with pytest.raises(model_provisioning.ModelProvisioningError):
        model_provisioning._load("tokenizer", "org/model", factory)
    component = model_provisioning._load("tokenizer", "org/model", factory)
    assert component is not None
    assert attempts == [True, None, True]


def test_components_are_shared_across_encoder_instances(monkeypatch):
    built = []

    def tokenizer_factory(model_name, **kwargs):
        built.append("tokenizer")
        return FakeComponent()

    def model_factory(model_name, **kwargs):
        built.append("model")
        return FakeComponent()

    def fake_load_tokenizer(model_name, **kwargs):
        return model_provisioning._load(
            "tokenizer", model_name, tokenizer_factory,
            cache_dir=kwargs.get("cache_dir"),
            local_files_only=kwargs.get("local_files_only"))

    def fake_load_base_model(model_name, **kwargs):
        return model_provisioning._load(
            "sentence-embedding", model_name, model_factory,
            cache_dir=kwargs.get("cache_dir"),
            local_files_only=kwargs.get("local_files_only"),
            device=kwargs.get("device", "cpu"))

    monkeypatch.setattr(dense, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(dense, "load_base_model", fake_load_base_model)

    first = dense.DenseSemanticEncoder()
    second = dense.DenseSemanticEncoder()
    assert built == ["tokenizer", "model"]
    assert first.tokenizer is second.tokenizer
    assert first.model is second.model


def test_lazy_resolution_is_thread_safe(monkeypatch):
    constructed = []

    class SlowEncoder:
        dim = 384

        def __init__(self, *args, **kwargs):
            constructed.append("constructed")
            time.sleep(0.05)

        def encode(self, text):
            return np.zeros(384, dtype=np.float32)

    monkeypatch.setattr(dense, "DenseSemanticEncoder", SlowEncoder)
    encoder = dense.LazyDenseSemanticEncoder()
    outputs = []

    def worker():
        outputs.append(encoder.encode("hello"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert constructed == ["constructed"]
    assert len(outputs) == 8


def test_lazy_proxy_ignores_private_attributes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("private attribute access must not load a model")

    monkeypatch.setattr(dense, "DenseSemanticEncoder", forbidden)
    encoder = dense.LazyDenseSemanticEncoder()
    with pytest.raises(AttributeError):
        encoder._not_a_real_attribute


def test_cache_location_uses_hf_home_when_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert model_provisioning.cache_location() == str(
        tmp_path / "hf" / "hub")


def test_lazy_dense_encoder_does_not_load_at_construction(monkeypatch):
    calls = []

    class FakeEncoder:
        dim = 384

        def __init__(self, *args, **kwargs):
            calls.append("constructed")

        def encode(self, text):
            return np.zeros(384, dtype=np.float32)

        def encode_batch(self, texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(dense, "DenseSemanticEncoder", FakeEncoder)
    encoder = dense.LazyDenseSemanticEncoder()
    assert encoder.dim == 384
    assert calls == []
    encoder.encode("hello")
    assert calls == ["constructed"]
    encoder.encode_batch(["hello", "world"])
    assert calls == ["constructed"]


def test_lazy_nli_gate_does_not_load_at_construction(monkeypatch):
    calls = []

    class FakeGate:
        def __init__(self, *args, **kwargs):
            calls.append("constructed")

        def predict_pair(self, premise, hypothesis, **kwargs):
            return {"decision": "NEUTRAL"}

        def predict_batch(self, pairs, **kwargs):
            return [{"decision": "NEUTRAL"} for _ in pairs]

    monkeypatch.setattr(nli, "NliSemanticGate", FakeGate)
    gate = nli.LazyNliSemanticGate()
    assert calls == []
    gate.predict_pair("a", "b")
    assert calls == ["constructed"]
    gate.predict_batch([("a", "b")])
    assert calls == ["constructed"]


def test_constructing_a_memory_provisions_nothing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model provisioning must not run at construction")

    monkeypatch.setattr(model_provisioning, "_load", forbidden)

    from minemos_engine.black_hole_memory import BlackHoleMemory

    memory = BlackHoleMemory(capacity=4)
    assert memory.dense_encoder._encoder is None
    assert memory.nli_gate._gate is None


def test_memory_cache_dir_is_forwarded():
    from minemos_engine.black_hole_memory import BlackHoleMemory

    memory = BlackHoleMemory(capacity=4, model_cache_dir="/tmp/minemos-models")
    assert memory.dense_encoder.cache_dir == "/tmp/minemos-models"
    assert memory.nli_gate.cache_dir == "/tmp/minemos-models"


def test_structured_types_and_default_resolution_need_no_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("this path must not provision a model")

    monkeypatch.setattr(model_provisioning, "_load", forbidden)

    from minemos import Candidate, MemoryState
    from minemos.resolution.config import ResolutionConfig
    from minemos.resolution.similarity import HashSimilarity, build_similarity

    state = MemoryState(capacity=2, dim=8)
    assert state.slots == []
    assert Candidate("alice", "city", "Paris").subject == "alice"

    similarity = build_similarity(ResolutionConfig())
    assert isinstance(similarity, HashSimilarity)
    assert similarity("New York", "new york") == 1.0


def test_provisioning_error_is_public():
    from minemos import ModelProvisioningError as PublicError
    from minemos_engine.model_provisioning import ModelProvisioningError

    assert PublicError is ModelProvisioningError
