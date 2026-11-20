"""Tests for the host-level shadow observer (V6).

The observer is the only place a host calls the resolution layer beside its
canonical write path.  These tests pin its contract: disabled by default,
shadow-only, non-raising, bounded and content-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minemos.resolution import (
    HostShadowObserver,
    HostShadowReport,
    ResolutionConfig,
    ShadowSettings,
    SourceMessage,
    write_json_report,
)
from minemos.resolution import host as host_module


def _messages(*texts: str, conversation: str = "conv-1"):
    return [
        SourceMessage(
            conversation_id=conversation,
            message_id=f"m{index:02d}",
            text=text,
            speaker="user",
            timestamp=float(index),
            order=index,
        )
        for index, text in enumerate(texts, start=1)
    ]


def _observer(**settings):
    settings.setdefault("log_every", 0)
    return HostShadowObserver(ShadowSettings(enabled=True, **settings))


# ── configuration ─────────────────────────────────────────────────────────


def test_settings_default_is_disabled():
    settings = ShadowSettings()
    assert settings.enabled is False
    assert settings.effective_mode == "disabled"
    described = settings.describe()
    assert described["mode"] == "disabled"
    assert described["activation"] == "explicit-opt-in"
    assert described["canonical_writes"] == 0


def test_settings_from_env_reads_only_explicit_opt_in():
    disabled = ShadowSettings.from_env({})
    assert disabled.enabled is False
    enabled = ShadowSettings.from_env({
        "MINEMOS_RESOLUTION_SHADOW": "true",
        "MINEMOS_RESOLUTION_SHADOW_REPORT": "var/shadow.json",
        "MINEMOS_RESOLUTION_SHADOW_LOG_EVERY": "25",
    })
    assert enabled.enabled is True
    assert enabled.report_path == Path("var/shadow.json")
    assert enabled.log_every == 25


@pytest.mark.parametrize("value", ["maybe", "2", ""])
def test_settings_from_env_rejects_invalid_bool(value):
    with pytest.raises(ValueError):
        ShadowSettings.from_env({"MINEMOS_RESOLUTION_SHADOW": value})


def test_settings_rejects_negative_log_every():
    with pytest.raises(ValueError):
        ShadowSettings(enabled=True, log_every=-1)


def test_observer_requires_explicit_enablement():
    with pytest.raises(ValueError, match="disabled"):
        HostShadowObserver(ShadowSettings())


def test_observer_rejects_non_shadow_modes():
    for mode in ("active", "disabled"):
        with pytest.raises(ValueError, match="shadow"):
            HostShadowObserver(
                ShadowSettings(enabled=True), ResolutionConfig(mode=mode))


def test_host_module_exposes_no_memory_writer():
    assert not hasattr(host_module, "absorb_record")
    assert not hasattr(host_module, "absorb_ingested")
    assert "absorb" not in host_module.__dict__


# ── observation counters ──────────────────────────────────────────────────


def test_observation_counts_without_writing():
    observer = _observer()
    delta = observer.observe(_messages(
        "I live in Berlin and work at Acme.",
        "My father lives in Oslo.",
        "My father doesn't live in Oslo, he lives in Bergen.",
        "I live in Berlin.",
    ))
    assert delta["messages"] == 4
    assert delta["segments"] == 4
    assert delta["facts_proposed"] >= 4
    assert delta["facts_eligible_if_active"] >= 1
    assert delta["negated_facts"] == 1
    assert delta["duplicate_candidates"] == 1
    assert delta["resolver_failures"] == 0
    assert delta["active_absorption"] is False
    assert delta["canonical_writes"] == 0
    assert delta["declines"]["eligible"] == delta["facts_eligible_if_active"]
    assert "en" in delta["languages"]


def test_repeated_identical_input_is_skipped():
    observer = _observer()
    messages = _messages("I live in Berlin.")
    first = observer.observe(messages)
    second = observer.observe(messages)
    assert first["facts_proposed"] >= 1
    assert second["facts_proposed"] == 0
    assert second["inputs_skipped"] == 1
    assert second["cumulative"]["messages"] == 1


def test_long_message_is_segmented_for_observation():
    observer = _observer()
    text = ("The office in the old quarter recorded the figures. " * 250).strip()
    assert len(text) > 8192
    delta = observer.observe(_messages(text))
    assert delta["messages"] == 1
    assert delta["segments"] == 2


def test_empty_input_is_a_no_op():
    observer = _observer()
    delta = observer.observe([])
    assert delta["messages"] == 0
    assert delta["resolver_failures"] == 0


def test_malformed_input_is_counted_not_raised():
    observer = _observer()
    delta = observer.observe(["not a source message", object()])
    assert delta["resolver_failures"] == 2
    assert delta["messages"] == 0


# ── failure isolation ─────────────────────────────────────────────────────


def test_ingestion_failure_is_confined(monkeypatch):
    observer = _observer()

    def explode(*_args, **_kwargs):
        raise RuntimeError("synthetic resolver failure")

    monkeypatch.setattr(host_module, "ingest_message", explode)
    delta = observer.observe(_messages("I live in Berlin."))
    assert delta["resolver_failures"] == 1
    assert delta["facts_proposed"] == 0


def test_each_failing_message_is_counted(monkeypatch):
    observer = _observer()
    seen = []

    def fake_ingest(resolver, message, *, config=None, policy=None):
        # Force a pipeline-level failure record through the real path.
        seen.append(message.message_id)
        raise ValueError("pipeline error")

    monkeypatch.setattr(host_module, "ingest_message", fake_ingest)
    delta = observer.observe(_messages("I live in Berlin.", "I work at Acme."))
    assert delta["resolver_failures"] == 2
    assert seen == ["m01", "m02"]


# ── privacy ───────────────────────────────────────────────────────────────


def test_reports_and_logs_contain_no_message_content():
    observer = _observer()
    secret = "Zelda Quux lives in Atlantis."
    observer.observe(_messages(secret))
    snapshot = json.dumps(observer.snapshot())
    log_line = observer.report.log_line()
    for fragment in ("zelda", "quux", "atlantis", "lives in"):
        assert fragment not in snapshot.lower()
        assert fragment not in log_line.lower()


def test_snapshot_shape_matches_report_contract():
    observer = _observer()
    observer.observe(_messages("I live in Berlin."))
    snapshot = observer.snapshot()
    assert set(HostShadowReport.__dataclass_fields__)  # fields exist
    for key in (
        "messages", "segments", "facts_proposed",
        "facts_eligible_if_active", "facts_declined", "unresolved_subjects",
        "ambiguous_identities", "negated_facts", "duplicate_candidates",
        "resolver_failures", "abstentions", "transitions",
        "latency_seconds_total", "latency_ms_mean", "active_absorption",
        "canonical_writes",
    ):
        assert key in snapshot


# ── report writing ────────────────────────────────────────────────────────


def test_report_is_written_atomically(tmp_path):
    target = tmp_path / "nested" / "report.json"
    assert write_json_report(target, {"facts_proposed": 3, "writes": 0})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "facts_proposed": 3, "writes": 0}
    assert not list(target.parent.glob("*.tmp"))


def test_report_failure_returns_false_not_raises(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert write_json_report(blocker / "report.json", {"a": 1}) is False
