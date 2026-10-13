"""Run every pending experiment overnight, then regenerate, verify and export the paper.

    cd python && python -m eval.overnight --dry-run     # preflight and the plan; runs nothing
    cd python && python -m eval.overnight               # the whole run
    cd python && python -m eval.overnight --resume      # continue after an interruption

Three lanes run concurrently, because they compete for different resources:

* **gpu**  -- everything that calls a local model through Ollama (E29, E30, E31). One model is
  resident at a time on the 8 GB GPU, so the lane is ordered to switch models once.
* **wsl**  -- the gocryptfs store experiments in WSL (E16 extension, E32, E6 extension). CPU and
  fsync bound; jobs inside each stage run in parallel (``--jobs``).
* **win**  -- E33, two Syncthing instances and an agent-paced workload; wall-clock bound, idle CPU.

Before the lanes: a preflight, the native module rebuilt and copied into the framework venv, and
the Ollama server started with every model present. After them: every aggregation, the tables,
macros and figures, the verifier, the checked paper build, and the export (PDF, DOCX, LaTeX
sources). A stage that fails does not stop the others; stages that depend on it are skipped. The
machine is kept awake for the duration, and a report is written at the end.

Registration discipline: the predictions these experiments test are in ``eval/exp2/verify_r4.py``
and must be committed before this runs. The preflight refuses to start if that file has
uncommitted changes.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "python"
VENV = ROOT / ".venv" / "Scripts" / "python.exe"
VENV_FW = ROOT / ".venv-fw" / "Scripts" / "python.exe"
MATURIN = ROOT / ".venv" / "Scripts" / "maturin.exe"
OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
OLLAMA_URL = "http://127.0.0.1:11434"
MODELS = ["qwen2.5:7b-instruct", "llama3.1:8b", "nomic-embed-text"]
WSL_DISTRO = "Ubuntu"
WSL_REPO = "/mnt/c/Users/Admin/Desktop/Projects/Code/03_Apps-Products/cybsersec"
LOGS = PY / "eval" / "logs" / "overnight"
STATE = LOGS / "state.json"


@dataclass
class Stage:
    name: str
    lane: str  # prep | gpu | wsl | win | final
    argv: list[str] | None  # None means the stage is a Python callable in ``fn``
    est_min: float
    timeout_min: float
    needs: set[str] = field(default_factory=set)  # ollama | wsl | native
    fn: object = None
    cwd: Path = PY
    optional: bool = False  # a failure is reported but is not an error of the run


def fw(module: str, *args: str) -> list[str]:
    return [str(VENV_FW), "-u", "-m", module, *args]


def py(module: str, *args: str) -> list[str]:
    return [str(VENV), "-u", "-m", module, *args]


def wsl(mode: str, jobs: int) -> list[str]:
    script = f"{WSL_REPO}/python/eval/e8/run_par_wsl.sh"
    return ["wsl", "-d", WSL_DISTRO, "--", "env", f"JOBS={jobs}", "bash", script, mode]


# --------------------------------------------------------------------------- prep steps


def build_native(log) -> None:
    """Rebuild the extension if any Rust source is newer, then mirror it into the framework venv."""
    pyd = ROOT / ".venv" / "Lib" / "site-packages" / "mnemos_native" / "mnemos_native.pyd"
    newest = max(p.stat().st_mtime for p in (ROOT / "crates").rglob("*.rs"))
    if not pyd.exists() or pyd.stat().st_mtime < newest:
        log.write(f"rebuilding mnemos_native (sources newer than {pyd})\n")
        log.flush()
        env = {**os.environ, "VIRTUAL_ENV": str(ROOT / ".venv"), "CARGO_BUILD_JOBS": "8"}
        env["PATH"] = str(ROOT / ".venv" / "Scripts") + os.pathsep + env["PATH"]
        r = subprocess.run(
            [
                str(MATURIN),
                "develop",
                "--profile",
                "release-ffi",
                "-m",
                "crates/mnemos-ffi/Cargo.toml",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if r.returncode != 0:
            raise RuntimeError("maturin develop failed")
    else:
        log.write("mnemos_native is newer than every Rust source; not rebuilt\n")
    src = pyd.parent
    dst = ROOT / ".venv-fw" / "Lib" / "site-packages" / "mnemos_native"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    for v in (VENV, VENV_FW):
        r = subprocess.run(
            [str(v), "-c", "import mnemos_native as m; print(m.__file__)"],
            capture_output=True,
            text=True,
        )
        log.write(f"{v.parent.parent.name}: {r.stdout.strip() or r.stderr.strip()}\n")
        if r.returncode != 0:
            raise RuntimeError(f"mnemos_native does not import in {v}")


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            r.read()
        return True
    except Exception:
        return False


def ollama_models() -> set[str]:
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
        return {m["name"] for m in json.loads(r.read())["models"]}


def start_ollama(log) -> None:
    if not ollama_up():
        if not OLLAMA_EXE.exists():
            raise RuntimeError(f"ollama.exe not found at {OLLAMA_EXE}")
        env = {
            **os.environ,
            "OLLAMA_KEEP_ALIVE": "6h",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "2",
            "OLLAMA_FLASH_ATTENTION": "1",
        }
        subprocess.Popen(
            [str(OLLAMA_EXE), "serve"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        t0 = time.time()
        while not ollama_up():
            if time.time() - t0 > 120:
                raise RuntimeError("ollama serve did not come up")
            time.sleep(2)
        log.write("started ollama serve\n")
    have = ollama_models()
    for m in MODELS:
        if m not in have and f"{m}:latest" not in have:
            log.write(f"pulling {m}\n")
            log.flush()
            r = subprocess.run([str(OLLAMA_EXE), "pull", m], stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                raise RuntimeError(f"ollama pull {m} failed")
    log.write(f"models present: {sorted(ollama_models())}\n")


# --------------------------------------------------------------------------- the plan


E29 = "eval.exp2.e29_mem0_mnemos"
E31 = "eval.exp2.e31_amplify"
N = ("--probes", "60", "--seed-size", "60")


def plan(jobs: int, syncthing_min: float) -> list[Stage]:
    s: list[Stage] = [
        Stage("native", "prep", None, 6, 30, fn=build_native),
        Stage("ollama", "prep", None, 1, 30, needs=set(), fn=start_ollama),
        # gpu lane: E29 first (the application result), then the oracle's breadth.
        # Sixty member and sixty non-member probes per arm (E18 used thirty each). Measured cost:
        # about 2.5 s per mem0 call on the GPU, plus a fixed observation window per E29 probe.
        Stage("e29-qdrant", "gpu", fw(E29, "--arm", "qdrant", *N), 22, 120, {"ollama"}),
        Stage("e29-m_store", "gpu", fw(E29, "--arm", "m_store", *N), 22, 120, {"ollama", "native"}),
        Stage("e29-m_all", "gpu", fw(E29, "--arm", "m_all", *N), 22, 120, {"ollama", "native"}),
        Stage("e30-langmem", "gpu", fw("eval.exp2.e30_langmem", *N), 15, 120, {"ollama"}),
        Stage(
            "e31-qwen", "gpu", fw(E31, "--model", "qwen2.5:7b-instruct", *N), 18, 150, {"ollama"}
        ),
        Stage("e29-p_store", "gpu", fw(E29, "--arm", "p_store", *N), 22, 120, {"ollama", "native"}),
        Stage("e31-llama", "gpu", fw(E31, "--model", "llama3.1:8b", *N), 18, 150, {"ollama"}),
        # wsl lane: the abstract's pgvector claim first, the breadth after.
        # E16 runs at three jobs so I/O contention stays near the conditions of its first seeds.
        Stage("e16-seeds", "wsl", wsl("e16", min(jobs, 3)), 8, 120, {"wsl"}),
        Stage("e32-granularity", "wsl", wsl("e32", jobs), 25, 180, {"wsl"}),
        Stage("e6-seeds", "wsl", wsl("e6", jobs), 45, 240, {"wsl"}, optional=True),
        # win lane
        Stage(
            "e33-agent-paced",
            "win",
            py("eval.exp2.e33_syncthing", "--minutes", str(syncthing_min), "--every", "30"),
            syncthing_min + 3,
            syncthing_min + 45,
            {"native"},
            optional=True,
        ),
        Stage(
            "e33-fast",
            "win",
            py("eval.exp2.e33_syncthing", "--minutes", str(syncthing_min), "--every", "5"),
            syncthing_min + 3,
            syncthing_min + 45,
            {"native"},
            optional=True,
        ),
        # final: aggregate, regenerate, verify, build, export.
        Stage("agg-e16", "final", py("eval.e8.aggregate_e16"), 0.2, 5),
        Stage("agg-e32", "final", py("eval.e8.aggregate_e32"), 0.2, 5),
        Stage("agg-e6", "final", py("eval.exp2.e6_stores", "--collect"), 0.2, 5),
        Stage("agg-e29", "final", fw("eval.exp2.e29_mem0_mnemos", "--aggregate"), 0.1, 5),
        Stage("agg-e31", "final", fw("eval.exp2.e31_amplify", "--aggregate"), 0.1, 5),
        Stage("agg-e33", "final", py("eval.exp2.e33_syncthing", "--aggregate"), 0.1, 5),
        Stage("collect", "final", py("eval.exp2.collect"), 0.5, 10, optional=True),
        Stage("ledger", "final", py("eval.exp2.ledger"), 0.2, 10),
        Stage("tables", "final", py("eval.exp2.make_tables"), 0.2, 10),
        Stage("numbers", "final", py("eval.exp2.numbers"), 0.1, 5),
        Stage("figures", "final", py("eval.exp2.make_figures"), 0.5, 10),
        Stage("verify", "final", py("eval.exp2.verify"), 0.1, 5, optional=True),
        Stage("lint", "final", py("eval.lint_paper"), 0.1, 5, optional=True),
        Stage("paper", "final", py("eval.build_paper"), 1.5, 15),
        Stage("response", "final", py("eval.check_response"), 0.1, 5, optional=True),
        Stage("dist", "final", py("eval.make_dist"), 0.5, 10),
        Stage("docx", "final", py("eval.make_docx"), 1.5, 20),
    ]
    return s


# --------------------------------------------------------------------------- preflight


def preflight() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, ok, detail))

    add(".venv python", VENV.exists(), str(VENV))
    add(".venv-fw python", VENV_FW.exists(), str(VENV_FW))
    add("maturin", MATURIN.exists())
    for v, mods in (
        (VENV, "numpy,sklearn,scipy"),
        (VENV_FW, "mem0,langmem,langchain_openai,qdrant_client"),
    ):
        r = subprocess.run([str(v), "-c", f"import {mods}"], capture_output=True, text=True)
        add(f"imports in {v.parent.parent.name}", r.returncode == 0, r.stderr.strip()[-160:])
    add("ollama.exe", OLLAMA_EXE.exists(), str(OLLAMA_EXE))
    add("ollama server up now", ollama_up(), "started by the run if not")
    if ollama_up():
        have = ollama_models()
        miss = [m for m in MODELS if m not in have and f"{m}:latest" not in have]
        add("ollama models", not miss, f"missing {miss}" if miss else ", ".join(MODELS))
    add("syncthing.exe", (ROOT / "tools/bin/syncthing.exe").exists())
    probe = (
        "test -x ~/bin/gocryptfs && test -x ~/e8venv/bin/python && "
        "~/e8venv/bin/python -c 'import pgserver, psycopg2' && "
        "python3 -c 'import chromadb, lancedb, faiss, sqlite_vec' && echo OK"
    )
    try:
        r = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
        add(
            "WSL: gocryptfs, pgserver, psycopg2, store libraries",
            "OK" in r.stdout,
            r.stderr.strip()[-160:],
        )
    except Exception as e:
        add("WSL reachable", False, str(e))
    add("E16/E6 embeddings", (PY / "eval/e8/data/emb.npy").exists())
    free = shutil.disk_usage(ROOT).free / 2**30
    add("free disk >= 20 GB", free >= 20, f"{free:.0f} GB")
    r = subprocess.run(
        ["git", "status", "--porcelain", "--", "python/eval/exp2/verify_r4.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    add("verify_r4.py committed (registration)", r.stdout.strip() == "", r.stdout.strip())
    return out


# --------------------------------------------------------------------------- running


class Runner:
    def __init__(self, stages: list[Stage], logdir: Path, resume: bool) -> None:
        self.stages = stages
        self.logdir = logdir
        self.state = json.loads(STATE.read_text()) if (resume and STATE.exists()) else {}
        self.lock = threading.Lock()
        self.failed_needs: set[str] = set()

    def _save(self) -> None:
        with self.lock:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(self.state, indent=1))

    def run(self, st: Stage) -> bool:
        prev = self.state.get(st.name, {})
        if prev.get("status") == "ok":
            print(f"[skip] {st.name} (done in an earlier run)", flush=True)
            return True
        blocked = st.needs & self.failed_needs
        if blocked:
            self.state[st.name] = {"status": "skipped", "why": f"needs {sorted(blocked)}"}
            self._save()
            print(f"[skip] {st.name}: needs {sorted(blocked)}", flush=True)
            return False
        log_path = self.logdir / f"{st.name}.log"
        t0 = time.time()
        print(
            f"[{datetime.now():%H:%M:%S}] start {st.name} ({st.lane}, ~{st.est_min:.0f} min)",
            flush=True,
        )
        ok, why = False, ""
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"# {st.name}  {datetime.now().isoformat()}\n# {st.argv}\n")
            log.flush()
            try:
                if st.fn is not None:
                    st.fn(log)
                    ok = True
                else:
                    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
                    env.pop("VIRTUAL_ENV", None)
                    p = subprocess.Popen(
                        st.argv, cwd=st.cwd, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        rc = p.wait(timeout=st.timeout_min * 60)
                        ok, why = rc == 0, f"exit {rc}"
                    except subprocess.TimeoutExpired:
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True
                        )
                        if st.lane == "wsl":
                            subprocess.run(
                                ["wsl", "-d", WSL_DISTRO, "--", "pkill", "-f", "run_e6.py"],
                                capture_output=True,
                            )
                        why = f"timed out after {st.timeout_min:.0f} min"
            except Exception as e:
                why = f"{type(e).__name__}: {e}"
                log.write(f"\n{why}\n")
        dt = (time.time() - t0) / 60
        status = "ok" if ok else ("failed-optional" if st.optional else "failed")
        with self.lock:
            self.state[st.name] = {
                "status": status,
                "minutes": round(dt, 1),
                "why": why,
                "log": str(log_path),
            }
        self._save()
        print(
            f"[{datetime.now():%H:%M:%S}] {status:>15} {st.name} in {dt:.1f} min {why}", flush=True
        )
        if not ok and st.name == "native":
            self.failed_needs.add("native")
        if not ok and st.name == "ollama":
            self.failed_needs.add("ollama")
        return ok

    def lane(self, name: str) -> None:
        for st in self.stages:
            if st.lane == name:
                self.run(st)


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping while the run is in progress (released on exit)."""
    with contextlib.suppress(Exception):
        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def report(stages: list[Stage], logdir: Path, t0: float, state: dict) -> Path:
    lines = [
        f"# Overnight run, {datetime.fromtimestamp(t0):%Y-%m-%d %H:%M} -> {datetime.now():%H:%M}",
        "",
        "| stage | lane | status | minutes | note |",
        "|---|---|---|---|---|",
    ]
    for st in stages:
        s = state.get(st.name, {})
        lines.append(
            f"| {st.name} | {st.lane} | {s.get('status', 'not run')} | {s.get('minutes', '')} | "
            f"{s.get('why', '')} |"
        )
    v = logdir / "verify.log"
    if v.exists():
        lines += [
            "",
            "## Verifier",
            "",
            "```",
            *v.read_text(encoding="utf-8", errors="replace").splitlines()[-45:],
            "```",
        ]
    dist = ROOT / "paper" / "dist"
    if dist.exists():
        lines += ["", "## Export", ""] + [
            f"- {p.relative_to(ROOT)}" for p in sorted(dist.iterdir())
        ]
    for st in stages:
        s = state.get(st.name, {})
        if s.get("status", "").startswith("failed"):
            lg = Path(s.get("log", ""))
            if lg.exists():
                tail = lg.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                lines += ["", f"## {st.name} (last lines)", "", "```", *tail, "```"]
    out = logdir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(out, LOGS / "latest-report.md")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="preflight and print the plan only")
    ap.add_argument("--resume", action="store_true", help="skip stages that finished earlier")
    ap.add_argument("--jobs", type=int, default=4, help="parallel WSL jobs per stage")
    ap.add_argument("--syncthing-minutes", type=float, default=30.0)
    ap.add_argument("--only", nargs="*", help="run only these stages")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these stages")
    ap.add_argument("--force", action="store_true", help="run even if the preflight fails")
    args = ap.parse_args()

    stages = plan(args.jobs, args.syncthing_minutes)
    if args.only:
        stages = [s for s in stages if s.name in args.only or s.lane == "prep"]
    stages = [s for s in stages if s.name not in args.skip]

    print("preflight:")
    checks = preflight()
    for name, ok, detail in checks:
        print(f"  [{'ok' if ok else '!!'}] {name}  {detail}")
    lanes = {}
    for s in stages:
        lanes[s.lane] = lanes.get(s.lane, 0) + s.est_min
    par = max(lanes.get("gpu", 0), lanes.get("wsl", 0), lanes.get("win", 0))
    total = lanes.get("prep", 0) + par + lanes.get("final", 0)
    print("\nplan (estimated minutes):")
    for ln in ("prep", "gpu", "wsl", "win", "final"):
        names = [s.name for s in stages if s.lane == ln]
        if names:
            print(f"  {ln:5} {lanes[ln]:5.0f}  {', '.join(names)}")
    print(f"  wall clock ~{total / 60:.1f} h (lanes gpu/wsl/win run in parallel)")
    if args.dry_run:
        return
    hard = [c for c in checks if not c[1] and not c[0].startswith("ollama server up")]
    if hard and not args.force:
        print("\npreflight failed; fix the items above or pass --force")
        sys.exit(2)

    t0 = time.time()
    logdir = LOGS / datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    runner = Runner(stages, logdir, args.resume)
    keep_awake(True)
    try:
        runner.lane("prep")
        threads = [threading.Thread(target=runner.lane, args=(ln,)) for ln in ("gpu", "wsl", "win")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        runner.lane("final")
    finally:
        keep_awake(False)
        path = report(stages, logdir, t0, runner.state)
        print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
