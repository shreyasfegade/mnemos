"""Construct v1 and v2 stores for experiments from ``params.toml`` plus explicit overrides.

Every experiment states its overrides in one dictionary, the resulting parameter file is written
next to the store, and the file's SHA-256 is returned so results record the exact configuration.
"""

from __future__ import annotations

import copy
import hashlib
import os
import tomllib
from pathlib import Path

from eval.lab.corpus import repo_root


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def to_toml(d: dict) -> str:
    lines: list[str] = [f"{k} = {_fmt(v)}" for k, v in d.items() if not isinstance(v, dict)]
    for section, body in d.items():
        if not isinstance(body, dict):
            continue
        flat = {k: v for k, v in body.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in body.items() if isinstance(v, dict)}
        if flat:
            lines.append(f"\n[{section}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in flat.items())
        for sub, subbody in nested.items():
            lines.append(f"\n[{section}.{sub}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in subbody.items())
    return "\n".join(lines) + "\n"


def base_params() -> dict:
    with (repo_root() / "params.toml").open("rb") as fh:
        return tomllib.load(fh)


def write_params(path: Path, overrides: dict[str, dict]) -> str:
    """Merge ``{section: {key: value}}`` into params.toml, write to ``path``, return its SHA-256.

    ``epoch.wal_records`` follows ``epoch.epoch_writes`` automatically, and v1's
    ``slot_write_budget`` is raised to at least ``epoch_writes`` so a v1 store with a larger batch
    validates.
    """
    p = copy.deepcopy(base_params())
    for sec, kv in overrides.items():
        p.setdefault(sec, {}).update(kv)
    p["epoch"]["wal_records"] = p["epoch"]["epoch_writes"]
    p["epoch"]["slot_write_budget"] = max(
        p["epoch"]["slot_write_budget"], p["epoch"]["epoch_writes"] * 4
    )
    text = to_toml(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def create_v2(dir_: Path, centroids, overrides: dict[str, dict], ablation: dict | None = None):
    import mnemos_native as m

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    pp = dir_.parent / f"{dir_.name}.params.toml"
    h = write_params(pp, overrides)
    eng = m.EngineV2.create(
        str(dir_), [list(map(float, c)) for c in centroids], str(pp), True, ablation or {}
    )
    return eng, h, pp


def create_v1(dir_: Path, centroids, overrides: dict[str, dict]):
    import mnemos_native as m

    os.environ.setdefault("MNEMOS_I_ACCEPT_INSECURE", "1")
    pp = dir_.parent / f"{dir_.name}.params.toml"
    h = write_params(pp, overrides)
    eng = m.Engine.create(str(dir_), [list(map(float, c)) for c in centroids], str(pp), True)
    return eng, h, pp
