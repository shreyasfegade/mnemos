"""Shared pieces for the deduplication-oracle experiments (E29, E30, E31).

All three play the same adversary as E18: it supplies a fact to the agent's memory pipeline and
sees only the storage the pipeline writes, never a result. What it measures is how much of the
observed directory changed. This module keeps that observation identical across experiments.

* :func:`facts` is E18's fact pool: one-line facts from the D2 notes, in a fixed order.
* :class:`DirView` snapshots a directory as 4 KiB block hashes (the unit gocryptfs encrypts), and
  :func:`changed_blocks` counts the blocks that differ between two snapshots. Every file is hashed
  in full each time: the stores here are megabytes, Qdrant writes through ``mmap`` (whose mtime is
  not a reliable change signal on Windows), and the LLM call per probe dwarfs the hashing.
* :func:`auc_ci` scores a statistic against membership with a fixed, pre-stated orientation
  (more writing means non-member) and a percentile bootstrap over probes. Nothing is selected on
  the data, so a statistic with no signal sits at 0.5 rather than above it.
* :class:`Clock` drives a clocked Mnemos-M store from a background thread every Delta, the way an
  application timer would, holding the engine lock so a snapshot never sees a half-written flush.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from eval.lab import corpus as C

OLLAMA = "http://127.0.0.1:11434"
BLOCK = 4096


def facts(seed: int = 20261001) -> list[str]:
    """E18's fact pool, in E18's order: the first line of each D2 note, at most 200 characters."""
    rows = [
        json.loads(line)
        for line in (C.repo_root() / "data" / "d2-beir" / "notes.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    return [rows[i]["text"].split("\n")[0][:200] for i in idx]


class DirView:
    """Block-hash snapshots of a directory, optionally under a lock shared with a writer."""

    def __init__(self, roots: list[Path], lock: threading.Lock | None = None) -> None:
        self.roots = roots
        self.lock = lock

    def _snap(self) -> dict[tuple[str, int], str]:
        out: dict[tuple[str, int], str] = {}
        for root in self.roots:
            if not root.exists():
                continue
            for f in sorted(root.rglob("*")):
                if not f.is_file():
                    continue
                try:
                    data = f.read_bytes()
                except OSError:
                    continue
                name = str(f.relative_to(root.parent))
                for i in range(0, max(len(data), 1), BLOCK):
                    out[(name, i // BLOCK)] = hashlib.sha1(data[i : i + BLOCK]).hexdigest()
        return out

    def snapshot(self) -> dict[tuple[str, int], str]:
        if self.lock is None:
            return self._snap()
        with self.lock:
            return self._snap()


def changed_blocks(a: dict, b: dict) -> int:
    return sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k))


def auc_ci(
    y_nonmember: list[int], stat: list[float], rng: np.random.Generator, n_boot: int = 2000
) -> tuple[float, float, float]:
    """AUC of ``stat`` for "non-member", larger meaning more writing, with a bootstrap interval."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_nonmember)
    s = np.asarray(stat, dtype=float)

    def auc(yy, ss) -> float | None:
        if len(np.unique(yy)) < 2:
            return None
        return float(roc_auc_score(yy, ss)) if len(np.unique(ss)) > 1 else 0.5

    point = auc(y, s)
    if point is None:
        return 0.5, 0.5, 0.5
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = auc(y[i], s[i])
        if v is not None:
            boots.append(v)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ollama_ready(models: list[str], timeout_s: float = 600.0) -> None:
    """Fail fast unless the Ollama server answers and has every model; load each one once."""
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
                have = {m["name"] for m in json.loads(r.read())["models"]}
            break
        except Exception:
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Ollama not reachable at {OLLAMA}") from None
            time.sleep(2)
    for m in models:
        if m not in have and f"{m}:latest" not in have:
            raise RuntimeError(f"Ollama model {m!r} missing; run: ollama pull {m}")
    for m in models:
        kind = "embed" if "embed" in m else "generate"
        body = {"model": m, "keep_alive": "6h"}
        body.update(
            {"input": "warm"}
            if kind == "embed"
            else {"prompt": "ok", "options": {"num_predict": 1}}
        )
        req = urllib.request.Request(
            f"{OLLAMA}/api/{kind}", data=json.dumps(body).encode(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()


def ollama_embed(
    texts: list[str], model: str = "nomic-embed-text", batch: int = 64
) -> list[list[float]]:
    """Embeddings in batches: Ollama rejects a single request carrying thousands of inputs."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(
            f"{OLLAMA}/api/embed",
            data=json.dumps(
                {"model": model, "input": texts[i : i + batch], "keep_alive": "6h"}
            ).encode(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            out.extend(json.loads(r.read())["embeddings"])
    return out


def ollama_chat(prompt: str, model: str, temperature: float = 0.7, seed: int = 0) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "6h",
        "options": {"temperature": temperature, "seed": seed},
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=json.dumps(body).encode(), method="POST"
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


class Clock:
    """Calls ``fn`` every ``delta_s`` seconds from a daemon thread, under ``lock``."""

    def __init__(self, fn, delta_s: float, lock: threading.Lock) -> None:
        self.fn = fn
        self.delta = delta_s
        self.lock = lock
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        nxt = time.monotonic() + self.delta
        while not self._stop.is_set():
            time.sleep(max(0.0, nxt - time.monotonic()))
            nxt += self.delta
            try:
                with self.lock:
                    self.fn()
            except Exception as e:  # recorded, never silently dropped
                self.errors.append(f"{type(e).__name__}: {e}")

    def start(self) -> Clock:
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=10)


@contextmanager
def timed():
    t = [time.time(), 0.0]
    yield t
    t[1] = time.time() - t[0]


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)
