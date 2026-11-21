"""Clean-environment acceptance checks for an installed MINEMOS wheel.

Run from outside the repository with the target virtual environment's
interpreter.  ``install_acceptance.py`` creates the venv, installs the wheel
and points ``HF_HOME`` at an empty directory so that the first-use download is
genuinely cold.  This script records evidence as JSON and exits non-zero if
any check fails.

Modes
-----
``base``    install, import, metadata, no-download paths, real first-use
            provisioning, cache reuse, offline behaviour, quick start,
            checkpoint restore, shadow no-write and active ingestion.
``qdrant``  optional Qdrant extra after it has been installed separately.
``--probe`` internal subprocess probes (run by the ``base`` checks).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

DENSE_ID = "sentence-transformers/all-MiniLM-L6-v2"
NLI_ID = "cross-encoder/nli-distilroberta-base"
DENSE_DIR = "models--sentence-transformers--all-MiniLM-L6-v2"
NLI_DIR = "models--cross-encoder--nli-distilroberta-base"

WORK_ITEMS = (
    "import_minemos",
    "metadata",
    "no_download_paths",
    "construct_without_download",
    "cold_provisioning",
    "both_models_load",
    "cache_reuse_offline",
    "offline_miss_message",
    "quick_start",
    "checkpoint_restore",
    "shadow_no_write",
    "active_ingestion",
    "independent_install",
    "no_bundled_weights",
)


class Failure(AssertionError):
    pass


def expect(condition: object, message: str) -> None:
    if not condition:
        raise Failure(message)


def _env_clean() -> dict:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _files(root: Path) -> dict:
    """Snapshot of model cache files, ignoring transient lock metadata."""
    root = Path(root)
    if not root.exists():
        return {}
    snapshot = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if ".locks" in relative or ".no_exist" in relative:
            continue
        snapshot[relative] = path.stat().st_size
    return snapshot


def _model_dirs() -> list[str]:
    from minemos_engine.model_provisioning import cache_location

    root = Path(cache_location())
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _cache_bytes(root: Path) -> int:
    return sum(_files(root).values())


def _run_probe(name: str, workdir: Path, extra_env: dict) -> dict:
    env = _env_clean()
    env.update(extra_env)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe", name],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.perf_counter() - started
    payload = None
    for line in reversed(completed.stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    return {
        "returncode": completed.returncode,
        "seconds": round(elapsed, 3),
        "payload": payload,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
    }


# ---------------------------------------------------------------------------
# subprocess probes
# ---------------------------------------------------------------------------

def probe_no_download() -> int:
    """Import, construct, restore-free structured use and default resolution."""
    import minemos
    from minemos import (
        BlackHoleMemory,
        Candidate,
        MemoryState,
        ResolutionConfig,
        Resolver,
        ShadowRunner,
        SourceMessage,
    )

    state = MemoryState(capacity=4, dim=8)
    candidate = Candidate("Alice", "city", "Paris", timestamp=1.0)
    memory = BlackHoleMemory(capacity=4)

    message = SourceMessage(
        conversation_id="conv-1",
        message_id="m-1",
        text="My name is Samuel and I work at Acme Corp.",
        timestamp=1.0,
        order=0,
    )
    record = Resolver(ResolutionConfig()).resolve(message)
    runner = ShadowRunner(ResolutionConfig())
    runner.run(message)

    print(json.dumps({
        "version": minemos.__version__,
        "slots": len(state.slots),
        "candidate": candidate.subject,
        "memory_slots": len(memory.memory.slots),
        "dense_loaded": memory.dense_encoder._encoder is not None,
        "nli_loaded": memory.nli_gate._gate is not None,
        "facts": len(record.record.facts),
        "shadow": runner.summary(),
    }, sort_keys=True))
    return 0


def probe_use() -> int:
    from minemos import BlackHoleMemory, Candidate

    memory = BlackHoleMemory(capacity=8)
    actions = [
        memory.absorb(Candidate("Alice", "city", "Paris", timestamp=1.0))["action"],
        memory.absorb(Candidate("Alice", "city", "Boston"), timestamp=2.0)["action"],
    ]
    rows = memory.query("what city is Alice in", top_k=3)
    print(json.dumps({
        "actions": actions,
        "top_text": rows[0]["text"] if rows else None,
        "top_valid_until": rows[0]["valid_until"] if rows else None,
    }, sort_keys=True))
    return 0


def probe_offline_miss() -> int:
    from minemos import BlackHoleMemory, Candidate

    try:
        memory = BlackHoleMemory(capacity=4)
        memory.absorb(Candidate("Alice", "city", "Paris"), timestamp=1.0)
    except Exception as exc:  # noqa: BLE001 - the error is the evidence
        print(json.dumps({
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True))
        return 0
    print(json.dumps({"error_type": None, "error": "no error raised"}))
    return 3


PROBES = {
    "no-download": probe_no_download,
    "use": probe_use,
    "offline-miss": probe_offline_miss,
}


# ---------------------------------------------------------------------------
# base checks
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.results: list[dict] = []
        self.started = time.time()

    def run(self, name: str, fn) -> dict:
        if name not in WORK_ITEMS:
            raise KeyError(name)
        started = time.perf_counter()
        print(f"[acceptance] {name} ...", flush=True)
        try:
            detail = fn() or {}
            record = {"name": name, "ok": True, "seconds":
                      round(time.perf_counter() - started, 3),
                      "detail": detail}
        except Exception as exc:  # noqa: BLE001 - recorded as evidence
            record = {
                "name": name,
                "ok": False,
                "seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
        self.results.append(record)
        print(f"[acceptance] {name}: {'PASS' if record['ok'] else 'FAIL'} "
              f"({record['seconds']} s)", flush=True)
        self.write()
        if not record["ok"]:
            raise SystemExit(1)
        return record

    def write(self) -> None:
        payload = {
            "python": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "hf_home": os.environ.get("HF_HOME"),
            "model_cache": os.environ.get("MINEMOS_MODEL_CACHE"),
            "offline": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            },
            "started": self.started,
            "elapsed_seconds": round(time.time() - self.started, 3),
            "results": self.results,
        }
        self.output.write_text(json.dumps(payload, indent=2, sort_keys=True),
                               encoding="utf-8")


def run_base(workspace: Path, output: Path,
             repo_root: Path | None) -> int:
    recorder = Recorder(output)
    probe_dir = workspace / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    def check_import():
        import minemos
        from importlib.metadata import version as distribution_version

        expect(Path(minemos.__file__).is_file(), "minemos has no __file__")
        installed = distribution_version("minemos")
        expect(minemos.__version__ == installed,
               f"minemos.__version__ {minemos.__version__!r} does not match "
               f"installed metadata {installed!r}")
        return {"version": minemos.__version__,
                "file": minemos.__file__}

    def check_metadata():
        from importlib.metadata import files, metadata, requires

        md = metadata("minemos")
        license_expression = md["License-Expression"]
        expect(license_expression == "Apache-2.0",
               f"license metadata is {license_expression!r}")
        expect(md["Requires-Python"] == ">=3.12,<3.13"
               or md["Requires-Python"] == "<3.13,>=3.12",
               f"Requires-Python is {md['Requires-Python']!r}")
        reqs = requires("minemos") or []
        joined = " ".join(reqs)
        for dependency in ("torch", "transformers", "numpy",
                           "scikit-learn", "certifi"):
            expect(dependency in joined,
                   f"missing runtime dependency {dependency}")
        qdrant_reqs = [req for req in reqs
                       if req.split()[0].split("==")[0].lower()
                       == "qdrant-client"]
        expect(len(qdrant_reqs) == 1,
               f"expected exactly one qdrant-client requirement, got "
               f"{qdrant_reqs!r}")
        expect('extra == "qdrant"' in qdrant_reqs[0],
               "qdrant-client must stay an optional extra, found "
               f"{qdrant_reqs[0]!r}")
        names = [str(item) for item in (files("minemos") or [])]
        expect(any("LICENSE" in name for name in names),
               "LICENSE is not installed")
        expect(any("NOTICE" in name for name in names),
               "NOTICE is not installed")
        expect(importlib.util.find_spec("qdrant_client") is None,
               "qdrant_client is present before the optional extra is installed")
        return {"license": license_expression,
                "requires_python": md["Requires-Python"],
                "requires": reqs,
                "qdrant_requirement": qdrant_reqs[0],
                "license_files": [n for n in names
                                  if "LICENSE" in n or "NOTICE" in n]}

    def check_no_download_paths():
        empty_hf = workspace / "hf-nodownload"
        empty_hf.mkdir(exist_ok=True)
        probe = _run_probe(
            "no-download", probe_dir,
            {"HF_HOME": str(empty_hf), "MINEMOS_MODEL_CACHE": ""},
        )
        expect(probe["returncode"] == 0,
               f"probe failed: {probe['stderr_tail']}")
        payload = probe["payload"] or {}
        expect(payload.get("memory_slots") == 0, f"unexpected slots {payload}")
        expect(payload.get("dense_loaded") is False,
               "dense encoder loaded during import/construct")
        expect(payload.get("nli_loaded") is False,
               "NLI gate loaded during import/construct")
        expect(payload.get("facts"), "resolver produced no facts")
        entries = [p.name for p in empty_hf.rglob("*")
                   if p.is_dir() and p.name.startswith("models--")]
        expect(entries == [], f"models downloaded on a no-semantic path: {entries}")
        return {"probe": probe, "model_dirs": entries}

    def check_construct_without_download():
        from minemos_engine.model_provisioning import cache_location

        root = Path(cache_location())
        before = _files(root)
        from minemos import BlackHoleMemory

        memory = BlackHoleMemory(capacity=8)
        expect(memory.dense_encoder._encoder is None,
               "dense encoder was eagerly loaded")
        expect(memory.nli_gate._gate is None,
               "NLI gate was eagerly loaded")
        after = _files(root)
        expect(after == before,
               "constructing BlackHoleMemory touched the model cache")
        return {"cache_files": len(before)}

    def check_cold_provisioning():
        from minemos_engine.model_provisioning import cache_location

        root = Path(cache_location())
        expect(_files(root) == {},
               f"the target cache is not empty: {sorted(_files(root))[:5]}")
        expect(_model_dirs() == [], f"models already present: {_model_dirs()}")

        from minemos import BlackHoleMemory, Candidate

        memory = BlackHoleMemory(capacity=8)
        first_started = time.perf_counter()
        first = memory.absorb(Candidate("Alice", "city", "Paris"),
                              timestamp=1.0)
        first_seconds = time.perf_counter() - first_started
        expect((root / DENSE_DIR).exists(),
               "the sentence encoder was not downloaded on first absorb")
        expect(not (root / NLI_DIR).exists(),
               "the NLI model was loaded by a first, comparison-free absorb")
        encoder_only = _files(root)

        second_started = time.perf_counter()
        second = memory.absorb(Candidate("Alice", "city", "Boston"),
                               timestamp=2.0)
        second_seconds = time.perf_counter() - second_started
        expect(second["action"] == "temporal_update",
               f"expected temporal_update, got {second['action']!r}")
        expect((root / NLI_DIR).exists(),
               "the NLI model was not downloaded on the first comparison")

        log_lines = [r.getMessage() for r in _LOG_RECORDS]
        expect(any("Downloading" in line and DENSE_ID in line
                   for line in log_lines),
               f"no first-download log line for the encoder: {log_lines}")
        expect(any("Downloading" in line and NLI_ID in line
                   for line in log_lines),
               f"no first-download log line for the NLI model: {log_lines}")

        state["memory"] = memory
        return {
            "action_first": first["action"],
            "encoder_download_seconds": round(first_seconds, 3),
            "nli_download_seconds": round(second_seconds, 3),
            "encoder_cache_files": len(encoder_only),
            "total_cache_bytes": _cache_bytes(root),
            "model_dirs": _model_dirs(),
            "log_lines": log_lines,
        }

    state: dict = {}

    def check_both_models_load():
        memory = state["memory"]

        dense = memory.dense_encoder
        gate = memory.nli_gate
        expect(dense.model_name == DENSE_ID, f"encoder is {dense.model_name!r}")
        expect(gate.model_name == NLI_ID, f"NLI model is {gate.model_name!r}")
        embedding = dense.encode("where does Alice live")
        expect(len(embedding) == 384, f"embedding width {len(embedding)}")
        prediction = gate.predict_pair(
            "Alice lives in Paris.", "Alice lives in Boston.")
        expect(set(prediction) >= {"decision", "p_entail", "p_contra"},
               f"unexpected NLI result {prediction}")
        expect(len(gate.id2label) == 3, "NLI head is not three-way")
        return {
            "encoder": dense.model_name,
            "encoder_dim": len(embedding),
            "nli": gate.model_name,
            "nli_labels": len(gate.id2label),
            "nli_decision": prediction["decision"],
        }

    def check_cache_reuse_offline():
        from minemos_engine.model_provisioning import cache_location

        root = Path(cache_location())
        before = _files(root)
        offline_env = {
            "HF_HOME": os.environ["HF_HOME"],
            "MINEMOS_MODEL_CACHE": os.environ.get("MINEMOS_MODEL_CACHE", ""),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        offline = _run_probe("use", probe_dir, offline_env)
        expect(offline["returncode"] == 0,
               f"offline reuse failed: {offline['stderr_tail']}")
        payload = offline["payload"] or {}
        expect(payload.get("actions") == ["insert", "temporal_update"],
               f"offline actions {payload.get('actions')}")
        expect("Boston" in (payload.get("top_text") or ""),
               f"offline retrieval returned {payload.get('top_text')!r}")
        after = _files(root)
        expect(after == before,
               "the offline run changed the model cache")

        online = _run_probe(
            "use", probe_dir,
            {"HF_HOME": os.environ["HF_HOME"],
             "MINEMOS_MODEL_CACHE": os.environ.get("MINEMOS_MODEL_CACHE", "")},
        )
        expect(online["returncode"] == 0,
               f"online cache reuse failed: {online['stderr_tail']}")
        expect(_files(root) == before,
               "an online run re-downloaded files into a populated cache")
        return {
            "offline": offline,
            "online_without_download": online,
            "cache_files": len(before),
            "cache_bytes": _cache_bytes(root),
        }

    def check_offline_miss():
        empty = workspace / "hf-offline-empty"
        empty.mkdir(exist_ok=True)
        probe = _run_probe(
            "offline-miss", probe_dir,
            {"HF_HOME": str(empty), "MINEMOS_MODEL_CACHE": "",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        payload = probe["payload"] or {}
        expect(probe["returncode"] == 0,
               f"probe crashed: {probe['stderr_tail']}")
        expect(payload.get("error_type") == "ModelProvisioningError",
               f"unexpected error: {payload}")
        message = payload.get("error") or ""
        for fragment in (DENSE_ID, "HF_HOME", "MINEMOS_MODEL_CACHE",
                         "HF_HUB_OFFLINE"):
            expect(fragment in message,
                   f"error message lacks {fragment!r}: {message}")
        entries = [p.name for p in empty.rglob("*")
                   if p.is_dir() and p.name.startswith("models--")]
        expect(entries == [], f"offline miss created {entries}")
        return {"payload": payload, "empty_cache_dirs": entries}

    def check_quick_start():
        from minemos import BlackHoleMemory, Candidate

        memory = BlackHoleMemory(capacity=50)
        memory.absorb(Candidate("Alice", "city", "New York"), timestamp=10.0)
        result = memory.absorb(Candidate("Alice", "city", "Boston"),
                               timestamp=20.0)
        expect(result["action"] == "temporal_update",
               f"quick start action {result['action']!r}")
        rows = memory.query("what city is Alice in", top_k=3)
        expect(rows and "Boston" in rows[0]["text"],
               f"quick start retrieved {rows}")
        state["quick_start_memory"] = memory
        return {"action": result["action"], "top": rows[0]["text"]}

    def check_checkpoint_restore():
        memory = state["quick_start_memory"]
        from minemos import BlackHoleMemory

        checkpoint = workspace / "memory.npz"
        memory.save_checkpoint(str(checkpoint), allowed_base_dir=str(workspace))
        expect(checkpoint.is_file(), "checkpoint was not written")
        restored = BlackHoleMemory(capacity=50)
        restored.load_checkpoint(str(checkpoint),
                                 allowed_base_dir=str(workspace))
        rows = restored.query("what city is Alice in", top_k=5,
                              active_only=False)
        texts = [row["text"] for row in rows]
        expect(any("Boston" in text for text in texts),
               f"restored memory lost the current value: {texts}")
        expect(any("New York" in text for text in texts),
               f"restored memory lost the superseded value: {texts}")
        active = [row for row in rows if row["valid_until"] is None]
        expect(active and "Boston" in active[0]["text"],
               f"restored active value is {active}")
        return {"checkpoint_bytes": checkpoint.stat().st_size,
                "texts": texts}

    def check_shadow_no_write():
        from minemos import (
            BlackHoleMemory,
            Candidate,
            EngineSink,
            ResolutionConfig,
            ShadowRunner,
            SourceMessage,
            SemanticIngestionError,
            SemanticIngestor,
        )

        memory = BlackHoleMemory(capacity=10)
        memory.absorb(Candidate("Alice", "city", "Paris"), timestamp=1.0)
        before = [
            (slot.memory_id, slot.text, slot.valid_from, slot.valid_until,
             slot.update_count)
            for slot in memory.memory.slots
        ]

        messages = [
            SourceMessage("conv-shadow", "m-1",
                          "My name is Samuel and I work at Acme Corp.",
                          timestamp=100.0, order=0),
            SourceMessage("conv-shadow", "m-2",
                          "I moved to Berlin last year.",
                          timestamp=101.0, order=1),
        ]
        runner = ShadowRunner(ResolutionConfig())
        results = runner.run_conversation(messages)
        expect(len(results) == 2, f"shadow produced {len(results)} results")
        expect(runner.summary()["messages"] == 2, "shadow counters are wrong")

        refused = False
        try:
            SemanticIngestor(EngineSink(memory),
                             config=ResolutionConfig(mode="shadow"))
        except SemanticIngestionError:
            refused = True
        expect(refused, "shadow ingestion was not refused")

        after = [
            (slot.memory_id, slot.text, slot.valid_from, slot.valid_until,
             slot.update_count)
            for slot in memory.memory.slots
        ]
        expect(before == after, "shadow mode mutated canonical memory")
        return {"shadow": runner.summary(), "slots": len(after)}

    def check_active_ingestion():
        from minemos import (
            BlackHoleMemory,
            EngineSink,
            ResolutionConfig,
            SemanticIngestor,
            SourceMessage,
        )

        memory = BlackHoleMemory(capacity=20)
        ingestor = SemanticIngestor(
            EngineSink(memory), config=ResolutionConfig(mode="active"))
        report = ingestor.ingest_conversation([
            SourceMessage("conv-active", "m-1",
                          "My name is Samuel and I work at Acme Corp.",
                          timestamp=100.0, order=0),
        ])
        expect(report.observations >= 2,
               f"only {report.observations} observations")
        expect(report.absorbed >= 2,
               f"only {report.absorbed} observations absorbed")
        slots = len(memory.memory.slots)
        expect(slots >= 2, f"canonical memory has {slots} slots")
        return {"report": report.summary(), "slots": slots}

    def check_independent_install():
        import minemos
        import minemos_engine

        package = Path(minemos.__file__).resolve()
        engine = Path(minemos_engine.__file__).resolve()
        prefix = Path(sys.prefix).resolve()
        expect(prefix in package.parents,
               f"{package} is not inside the virtual environment")
        expect(prefix in engine.parents,
               f"{engine} is not inside the virtual environment")
        detail = {"package": str(package), "engine": str(engine),
                  "prefix": str(prefix), "cwd": str(Path.cwd().resolve())}
        if repo_root is not None:
            cwd = Path.cwd().resolve()
            expect(repo_root not in cwd.parents and cwd != repo_root,
                   f"cwd {cwd} is inside the repository")
            expect(str(repo_root) not in sys.path,
                   "the repository is on sys.path")
            detail["repo_root"] = str(repo_root)
        return detail

    def check_no_bundled_weights():
        import minemos
        import minemos_engine

        roots = [Path(minemos.__file__).resolve().parent,
                 Path(minemos_engine.__file__).resolve().parent]
        weights = []
        large = []
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix in {".bin", ".safetensors", ".pt", ".pth",
                                   ".onnx", ".h5"}:
                    weights.append(str(path))
                if path.stat().st_size > 10 * 1024 * 1024:
                    large.append((str(path), path.stat().st_size))
        expect(weights == [], f"model weights are packaged: {weights}")
        expect(large == [], f"unexpectedly large packaged files: {large}")
        return {"roots": [str(root) for root in roots],
                "packaged_files": sum(
                    1 for root in roots
                    for path in root.rglob("*") if path.is_file())}

    recorder.run("import_minemos", check_import)
    recorder.run("metadata", check_metadata)
    recorder.run("no_download_paths", check_no_download_paths)
    recorder.run("construct_without_download", check_construct_without_download)

    log_handler = _LogCapture(_LOG_RECORDS)
    logger = logging.getLogger("minemos_engine.model_provisioning")
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    try:
        recorder.run("cold_provisioning", check_cold_provisioning)
    finally:
        logger.removeHandler(log_handler)

    recorder.run("both_models_load", check_both_models_load)
    recorder.run("cache_reuse_offline", check_cache_reuse_offline)
    recorder.run("offline_miss_message", check_offline_miss)
    recorder.run("quick_start", check_quick_start)
    recorder.run("checkpoint_restore", check_checkpoint_restore)
    recorder.run("shadow_no_write", check_shadow_no_write)
    recorder.run("active_ingestion", check_active_ingestion)
    recorder.run("independent_install", check_independent_install)
    recorder.run("no_bundled_weights", check_no_bundled_weights)
    recorder.write()
    return 0


_LOG_RECORDS: list[logging.LogRecord] = []


class _LogCapture(logging.Handler):
    def __init__(self, sink: list) -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record)


# ---------------------------------------------------------------------------
# optional Qdrant extra
# ---------------------------------------------------------------------------

def run_qdrant(workspace: Path, output: Path) -> int:
    started = time.perf_counter()
    detail: dict = {}
    import qdrant_client  # noqa: F401 - import is the availability check
    from minemos import BlackHoleMemory, Candidate, QdrantConfig

    try:
        from importlib.metadata import version as _dist_version

        detail["qdrant_client_version"] = _dist_version("qdrant-client")
    except Exception:  # noqa: BLE001 - informational only
        detail["qdrant_client_version"] = "unknown"
    memory = BlackHoleMemory(
        capacity=8,
        retrieval_backend="qdrant",
        qdrant_config=QdrantConfig(path=":memory:"),
    )
    memory.absorb(Candidate("Alice", "city", "Paris"), timestamp=1.0)
    rows = memory.query("where does Alice live", top_k=2)
    if not rows or "Paris" not in rows[0]["text"]:
        raise Failure(f"embedded Qdrant retrieval returned {rows}")
    detail["top"] = rows[0]["text"]
    detail["seconds"] = round(time.perf_counter() - started, 3)
    output.write_text(json.dumps({"ok": True, "detail": detail},
                                 indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(detail, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("base", "qdrant"))
    parser.add_argument("--probe", choices=sorted(PROBES))
    parser.add_argument("--workspace")
    parser.add_argument("--output")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    if args.probe:
        return PROBES[args.probe]()

    if not args.workspace or not args.output:
        parser.error("--workspace and --output are required")
    workspace = Path(args.workspace)
    output = Path(args.output)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None

    if args.mode == "base":
        return run_base(workspace, output, repo_root)
    if args.mode == "qdrant":
        try:
            return run_qdrant(workspace, output)
        except Exception as exc:  # noqa: BLE001 - recorded as evidence
            output.write_text(json.dumps({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }, indent=2), encoding="utf-8")
            print(f"qdrant check failed: {type(exc).__name__}: {exc}")
            return 1
    parser.error("choose a mode or a probe")
    return 2


if __name__ == "__main__":
    sys.exit(main())