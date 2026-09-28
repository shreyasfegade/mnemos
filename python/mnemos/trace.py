"""The adversary's transcript -- and the boundary that keeps the attacks honest.

Every module in ``python/eval/attacks/`` imports **this** and never :mod:`mnemos.engine`.
That is not a convention, it is the mechanism that makes the attack results mean anything:
an attack that can read engine internals will still produce a number, the number will still
look plausible, and it will measure nothing.

What crosses this boundary
--------------------------
``LeakEvent`` records, exactly as ``mnemos-core::event`` emits them -- the disk-visible
effects a host-compromise adversary observes.

What does not
-------------
Plaintext, key material, ``MemoryId``s, query text, or the ``decoy`` flag on ``BucketRead``.

That last one deserves a note. The engine records whether a bucket read was a decoy, because
the *evaluation* needs it to measure how well decoys work. The adversary obviously cannot see
it. :func:`load_transcript` strips the field; :func:`load_evaluation_trace` does not, and is
importable only from ``python/eval/experiments/``, never from ``python/eval/attacks/``. If
those two functions are ever collapsed into one, every access-pattern result in the paper
becomes worthless -- and nothing will visibly break.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LeakEvent:
    """One disk-visible effect, as observed."""

    kind: str
    """``setup`` | ``epoch_flush`` | ``slot_write`` | ``bucket_read`` | ``slot_read``
    | ``token_advance`` | ``compact`` | ``manifest_write`` | ``grow`` | ``key_op``"""

    fields: dict[str, Any]
    """Variant payload. See ``mnemos-core::event::LeakEvent``."""


#: Fields the adversary cannot observe. Stripped by :func:`load_transcript`.
_ADVERSARY_INVISIBLE = frozenset({"decoy"})


def load_transcript(path: Path, since_epoch: int | None = None) -> Iterator[LeakEvent]:
    """Load a transcript **as the adversary sees it**. Use this in attacks.

    Parameters
    ----------
    path
        JSONL file written by the engine's recording sink.
    since_epoch
        Compromise time ``tau``, expressed as an epoch index. Events before it are dropped,
        because a tau-arrival adversary has no transcript of them -- only the final state,
        which is loaded separately via :func:`load_snapshot`.

        **Passing ``None`` models a from-genesis persistent adversary**, which is the standard
        DSSE threat model and *not* ours. It is available so the difference can be measured,
        not because it is a sensible default.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            kind = raw.pop("kind")
            if since_epoch is not None and raw.get("epoch", 0) < since_epoch:
                continue
            fields = {k: v for k, v in raw.items() if k not in _ADVERSARY_INVISIBLE}
            yield LeakEvent(kind=kind, fields=fields)


def load_evaluation_trace(path: Path) -> Iterator[LeakEvent]:
    """Load a transcript **with adversary-invisible fields intact**.

    For measurement only -- e.g. E4 needs the ``decoy`` flag to report how many reads were
    decoys. **Never import this from ``python/eval/attacks/``.**
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            yield LeakEvent(kind=raw.pop("kind"), fields=raw)


# `load_snapshot` lived here: a planned shell-out to `mnemos attack --dump-snapshot` that would
# define snapshot-observability once, in Rust. It was never implemented and never called --- A3
# takes its snapshot from `eval.lab.schemes`, where each baseline exposes its own on-disk view
# through `Scheme.snapshot`. A stub that raises is worse than no stub: it reads as a planned
# capability rather than a decision already taken elsewhere.
