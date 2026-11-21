"""Driver for clean-environment acceptance runs of the MINEMOS distribution.

The driver creates a virtual environment outside the repository, installs the
built wheel (or the published PyPI release), and runs ``acceptance_checks.py``
with an empty ``HF_HOME`` so that first-use provisioning is genuinely cold.
It is the same path CI runs on every supported platform.

Examples
--------
    python scripts/install_acceptance.py --wheel dist/minemos-0.1.1-py3-none-any.whl
    python scripts/install_acceptance.py --published 0.1.1
    python scripts/install_acceptance.py --wheel ... --cpu-torch
    python scripts/install_acceptance.py --wheel ... --evidence-dir /tmp/evidence
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = HERE / "acceptance_checks.py"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CPU_TORCH = "torch==2.14.0+cpu"


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _stream(command: list[str], *, cwd: Path, env: dict, log: Path) -> int:
    print(f"\n$ {' '.join(str(part) for part in command)}", flush=True)
    started = time.perf_counter()
    with log.open("ab") as handle:
        handle.write(f"\n$ {' '.join(str(part) for part in command)}\n".encode())
        handle.flush()
        completed = subprocess.run(command, cwd=str(cwd), env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
    print(f"  exit {completed.returncode} "
          f"({time.perf_counter() - started:.1f} s)", flush=True)
    return completed.returncode


def _capture(command: list[str], *, cwd: Path, env: dict,
             log: Path) -> subprocess.CompletedProcess:
    with log.open("ab") as handle:
        handle.write(f"\n$ {' '.join(str(part) for part in command)}\n".encode())
        handle.flush()
        return subprocess.run(command, cwd=str(cwd), env=env,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)


def _python_env(workspace: Path) -> dict:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for name in ("MINEMOS_MODEL_CACHE", "HF_HUB_OFFLINE",
                 "TRANSFORMERS_OFFLINE", "HF_HUB_CACHE"):
        env.pop(name, None)
    # Every subprocess uses the same, initially empty cache root so the first
    # semantic operation exercises the download path.
    env["HF_HOME"] = str(workspace / "hf-cold")
    return env


def _prepare_workspace(prefix: str) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix=prefix))
    (workspace / "run").mkdir()
    return workspace


def _create_venv(workspace: Path, env: dict, log: Path) -> Path:
    venv = workspace / "venv"
    code = _stream([sys.executable, "-m", "venv", str(venv)],
                   cwd=workspace, env=env, log=log)
    if code != 0:
        raise SystemExit("virtual environment creation failed")
    python = _venv_python(venv)
    if not python.is_file():
        raise SystemExit(f"interpreter missing at {python}")
    _stream([str(python), "-m", "pip", "--version"],
            cwd=workspace, env=env, log=log)
    return python


def _torch_report(python: Path, workspace: Path, env: dict, log: Path) -> dict:
    code = (
        "import json, platform, torch; "
        "print(json.dumps({"
        "'python': platform.python_version(), "
        "'platform': platform.platform(), "
        "'machine': platform.machine(), "
        "'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'torch_threads': torch.get_num_threads()}))"
    )
    completed = _capture([str(python), "-c", code], cwd=workspace,
                         env=env, log=log)
    (workspace / "torch.txt").write_text(
        completed.stdout or completed.stderr or "", encoding="utf-8")
    text = completed.stdout.strip().splitlines()[-1] if completed.stdout else ""
    return json.loads(text) if text.startswith("{") else {"raw": text}


def _copy_evidence(workspace: Path, evidence_dir: Path | None,
                   outputs: dict[str, str]) -> dict[str, str]:
    if evidence_dir is None:
        return {}
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for target, source in outputs.items():
        source_path = workspace / source
        if source_path.exists():
            destination = evidence_dir / target
            shutil.copy2(source_path, destination)
            copied[target] = str(destination)
    return copied


def _install_target(args, python: Path, workspace: Path, env: dict,
                    log: Path) -> str:
    if args.published:
        spec = f"minemos=={args.published}"
        if _stream([str(python), "-m", "pip", "install",
                    "--disable-pip-version-check", spec],
                   cwd=workspace, env=env, log=log) != 0:
            raise SystemExit("published release installation failed")
        return spec

    wheel = Path(args.wheel).resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")
    if args.cpu_torch:
        if _stream([str(python), "-m", "pip", "install",
                    "--disable-pip-version-check", CPU_TORCH,
                    "--index-url", CPU_INDEX],
                   cwd=workspace, env=env, log=log) != 0:
            raise SystemExit("CPU-only PyTorch preinstall failed")
    if _stream([str(python), "-m", "pip", "install",
                "--disable-pip-version-check", str(wheel)],
               cwd=workspace, env=env, log=log) != 0:
        raise SystemExit("wheel installation failed")
    return str(wheel)


def run(args) -> int:
    workspace = _prepare_workspace("minemos-install-accept-")
    log = workspace / "driver.log"
    env = _python_env(workspace)
    print(f"workspace: {workspace}", flush=True)
    print(f"HF_HOME:   {env['HF_HOME']} (must be empty)", flush=True)

    python = _create_venv(workspace, env, log)
    target = _install_target(args, python, workspace, env, log)

    pip_check = _capture([str(python), "-m", "pip", "check"],
                         cwd=workspace, env=env, log=log)
    (workspace / "pip-check.txt").write_text(
        pip_check.stdout or "", encoding="utf-8")
    if pip_check.returncode != 0:
        raise SystemExit("pip check reported a broken environment")
    pip_list = _capture([str(python), "-m", "pip", "list",
                         "--format=json"], cwd=workspace, env=env, log=log)
    (workspace / "pip-list-base.json").write_text(
        pip_list.stdout or "", encoding="utf-8")
    torch_info = _torch_report(python, workspace, env, log)
    print(f"torch: {torch_info}", flush=True)

    checks = workspace / "acceptance_checks.py"
    shutil.copy2(CHECKS, checks)
    base_code = _stream([
        str(python), str(checks), "base",
        "--workspace", str(workspace),
        "--output", str(workspace / "base.json"),
        "--repo-root", args.repo_root,
    ], cwd=workspace / "run", env=env, log=log)

    qdrant_code = None
    if base_code == 0 and not args.skip_qdrant:
        extra = ("minemos[qdrant]" if args.published
                 else f"{target}[qdrant]")
        if _stream([str(python), "-m", "pip", "install",
                    "--disable-pip-version-check", extra],
                   cwd=workspace, env=env, log=log) == 0:
            qdrant_code = _stream([
                str(python), str(checks), "qdrant",
                "--workspace", str(workspace),
                "--output", str(workspace / "qdrant.json"),
            ], cwd=workspace / "run", env=env, log=log)
        else:
            qdrant_code = 1

    evidence_dir = Path(args.evidence_dir).resolve() if args.evidence_dir else None
    summary = {
        "mode": "published" if args.published else "wheel",
        "target": target,
        "workspace": str(workspace),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hf_home": env["HF_HOME"],
        "torch": torch_info,
        "pip_check": (pip_check.stdout or "").strip(),
        "base_exit": base_code,
        "qdrant_exit": qdrant_code,
        "evidence": _copy_evidence(workspace, evidence_dir, {
            "base.json": "base.json",
            "qdrant.json": "qdrant.json",
            "driver.log": "driver.log",
            "pip-check.txt": "pip-check.txt",
            "pip-list-base.json": "pip-list-base.json",
            "torch.txt": "torch.txt",
        }),
    }
    if evidence_dir is not None:
        (evidence_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.keep:
        print(f"workspace kept at {workspace}", flush=True)
    else:
        shutil.rmtree(workspace, ignore_errors=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if base_code == 0 and qdrant_code in (None, 0) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--wheel")
    target.add_argument("--published")
    parser.add_argument("--cpu-torch", action="store_true",
                        help="preinstall the CPU-only PyTorch wheel (Linux)")
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--repo-root", default="",
                        help="repository path, used to assert it is not on sys.path")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())