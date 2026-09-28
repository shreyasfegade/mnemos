"""MCP server -- the surface a real agent stores its real memories through.

Stage 4, and per decision D9 it is **required for the paper**, not optional: the empirical
asymmetry measurement is only credible on a corpus a real agent actually wrote.

Tools exposed
-------------
+--------------------+------------------------------------------+
| ``memory.put``     | ``(text, meta) -> memory_id``            |
| ``memory.search``  | ``(query, top_k=10) -> list[Memory]``    |
| ``memory.update``  | ``(memory_id, text) -> NEW memory_id``   |
| ``memory.delete``  | ``(memory_id) -> None``                  |
| ``memory.stat``    | ``() -> {count, epoch, capacity}``       |
| ``memory.lock``    | ``() -> None``                           |
| ``memory.unlock``  | ``() -> None``                           |
+--------------------+------------------------------------------+

The shape deliberately mirrors what agents already expect, so Mnemos is a drop-in for a
plaintext memory backend and the E7 comparison is like-for-like.

Two things worth knowing before wiring this up
----------------------------------------------
**``update`` returns a new id.** Updates are delete-then-put, never in-place, because an
in-place rewrite is a same-slot write -- exactly the correlation signal a live adversary wants
(ADR-0004). Agents that cache memory ids must handle the change. This is a visible API cost
paid for a security property, and it is the kind of honest consequence that makes a systems
paper credible rather than the kind that gets quietly hidden behind a compatibility shim.

**Writes are not immediately durable.** ``put`` returns as soon as the record is staged; it
lands at the next epoch flush (500 ms or 16 writes, whichever first). That batching is what
bounds Delta, so an agent that calls ``flush`` after every write has disabled the central
security mechanism. ``flush`` is therefore not exposed as a tool.

The consequence, stated because it bit us
-----------------------------------------
``mnemos-store::wal`` stages in memory and reaches disk only at flush, so **a process that exits
without flushing loses up to 16 entries**. A long-lived server is exactly the deployment where
that is fine -- the flush fires on a timer -- but :meth:`MnemosMemoryServer.close` must run, and
the context-manager protocol is provided so it does. ``mnemos-cli`` is the opposite case: one
process per write, so it flushes every time.

Plaintext residency
-------------------
This process holds plaintext before ``put`` and after ``search``, unprotected. Declared and
argued in ``ARCHITECTURE.md`` s6.3: pre-tau the machine is clean by assumption TA-2, and
post-tau the adversary reads process memory by definition, so nothing is lost that was not
already gone. Do not paper over this -- it is in the paper's limitations section on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import embed as embedding
from . import params


class MnemosMemoryServer:
    """MCP server backed by a Mnemos store.

    Use as a context manager, or call :meth:`close` explicitly -- see the module docstring on
    the flush-on-exit requirement.
    """

    def __init__(
        self,
        store_path: Path,
        embedder: Any = None,
        params_path: Path | None = None,
        allow_insecure_keystore: bool = False,
    ) -> None:
        self._store_path = Path(store_path)
        # Injectable so a test can supply a deterministic embedder, and so the E7 comparison can
        # run the *same* agent against a plaintext backend with the same vectors.
        self._embedder = embedder or embedding
        self._params_path = Path(params_path) if params_path else None
        # Explicit, and false by default. Reaching the software keystore additionally needs the
        # compile feature and MNEMOS_I_ACCEPT_INSECURE=1; keeping the third gate a constructor
        # argument means every call site that wants it says so in code review.
        self._allow_insecure_keystore = allow_insecure_keystore
        self._engine: Any = None  # mnemos_native.Engine

    def _params(self) -> dict[str, Any]:
        """Parameters for this server -- the override if given, else the workspace file."""
        if self._params_path is None:
            return params.load()
        import tomllib

        with self._params_path.open("rb") as fh:
            return tomllib.load(fh)

    def _params_arg(self) -> str | None:
        return str(self._params_path) if self._params_path else None

    # -- lifecycle --------------------------------------------------------------

    def open(
        self,
        create_if_missing: bool = False,
        centroids: list[list[float]] | None = None,
    ) -> None:
        """Open the store and unlock it via the hardware key store.

        ``create_if_missing`` trains a partition table from a synthetic sample. That is a
        deliberate one-time act: the table is fixed at creation and persisted, because
        retraining later moves every entry's partition and a store stops finding its own
        contents.

        ``centroids`` supplies that table explicitly instead. Two callers need this. A real
        deployment may want partitions seeded from a representative sample rather than from
        synthetic text. And the E7 utility comparison **requires** it: if Mnemos and the
        plaintext baseline each trained their own table they would partition the corpus
        differently, and the measured gap would be a difference between two k-means runs rather
        than the cost of the security machinery.
        """
        native = _load_native()
        p = self._params()

        if (self._store_path / "manifest.enc").is_file():
            self._engine = native.Engine.open(
                str(self._store_path),
                self._params_arg(),
                self._allow_insecure_keystore,
            )
            self._engine.unlock()
            return

        if not create_if_missing:
            raise FileNotFoundError(
                f"no store at {self._store_path} -- pass create_if_missing=True to make one"
            )

        dim = p["embedding"]["dim"]
        buckets = p["index"]["buckets"]
        if centroids is None:
            sample = self._embedder.embed(_training_sample(buckets * 8), dim=dim)
            centroids = native.train_centroids([list(map(float, v)) for v in sample], buckets, dim)
        if len(centroids) != buckets:
            raise ValueError(
                f"{len(centroids)} centroids supplied but params declares {buckets} partitions"
            )
        self._engine = native.Engine.create(
            str(self._store_path),
            centroids,
            self._params_arg(),
            self._allow_insecure_keystore,
        )

    def close(self) -> None:
        """Flush and lock. **Not optional** -- see the module docstring."""
        if self._engine is None:
            return
        try:
            self._engine.flush()
        finally:
            self._engine.lock()
            self._engine = None

    def __enter__(self) -> MnemosMemoryServer:
        self.open(create_if_missing=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if self._engine is None:
            raise RuntimeError("store is not open -- call open() first")
        return self._engine

    def _vector(self, text: str) -> list[float]:
        dim = self._params()["embedding"]["dim"]
        return [float(x) for x in self._embedder.embed([text], dim=dim)[0]]

    # -- tools ------------------------------------------------------------------

    def put(self, text: str, meta: dict[str, Any] | None = None) -> str:
        """Store a memory. Returns its id.

        Returns as soon as the record is staged; durability follows at the next epoch flush.
        """
        return str(self._require().put(text, self._vector(text), json.dumps(meta or {})))

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve the nearest memories. Constant-volume regardless of what matches.

        ``top_k`` truncates the *returned* list only. The number of slot reads is fixed by
        ``params.toml``, so asking for fewer results does not make the query cheaper and does
        not make it leak less -- which is the whole point, and worth not hiding behind an
        argument that looks like a performance knob.
        """
        hits = self._require().search(self._vector(query))
        out = [
            {"text": text, "meta": json.loads(meta or "{}"), "score": score}
            for text, meta, score in hits
        ]
        return out[:top_k]

    def update(self, memory_id: str, text: str, meta: dict[str, Any] | None = None) -> str:
        """Replace a memory. **Returns a NEW id** -- see the module docstring."""
        return str(
            self._require().update(memory_id, text, self._vector(text), json.dumps(meta or {}))
        )

    def delete(self, memory_id: str) -> None:
        """Delete a memory (Type-II backward privacy)."""
        self._require().delete(memory_id)

    def stat(self) -> dict[str, Any]:
        """Non-secret shape data: entry count, epoch, capacity."""
        s = dict(self._require().stat())
        s["embedding_backend"] = embedding.ACTIVE_BACKEND
        return s

    def lock(self) -> None:
        """Zeroise the master key and the locked arena."""
        self._require().lock()

    def unlock(self) -> None:
        """Unwrap the master key from hardware into the locked arena."""
        self._require().unlock()

    # -- evaluation hook --------------------------------------------------------

    def record_trace(self, path: Path) -> None:
        """Write ``LeakEvent``s to ``path`` as JSONL -- the adversary's transcript.

        For ``python/eval`` only. The attack modules read the file through :mod:`mnemos.trace`,
        never through this object, so an attack cannot accidentally see more than the threat
        model allows.
        """
        self._require().set_trace_sink(str(path))


#: MCP tool schema. Kept as data so the transport layer is swappable -- the ``mcp`` package is
#: an optional dependency and the server is useful (and testable) without it.
TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "memory.put",
        "description": "Store a memory. Returns its id. Durable at the next epoch flush.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "meta": {"type": "object"}},
            "required": ["text"],
        },
    },
    {
        "name": "memory.search",
        "description": "Retrieve the nearest memories. Constant-volume.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "memory.update",
        "description": "Replace a memory. Returns a NEW id -- ids are not stable across updates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "text": {"type": "string"},
                "meta": {"type": "object"},
            },
            "required": ["memory_id", "text"],
        },
    },
    {
        "name": "memory.delete",
        "description": "Delete a memory.",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory.stat",
        "description": "Entry count, epoch, capacity.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def dispatch(server: MnemosMemoryServer, tool: str, args: dict[str, Any]) -> Any:
    """Route one MCP tool call.

    Separated from the transport so the tool surface can be tested without an MCP client, and
    so swapping stdio for HTTP touches nothing here.
    """
    match tool:
        case "memory.put":
            return server.put(args["text"], args.get("meta"))
        case "memory.search":
            return server.search(args["query"], int(args.get("top_k", 10)))
        case "memory.update":
            return server.update(args["memory_id"], args["text"], args.get("meta"))
        case "memory.delete":
            return server.delete(args["memory_id"])
        case "memory.stat":
            return server.stat()
        case _:
            raise ValueError(f"unknown tool {tool!r}")


def _load_native() -> Any:
    try:
        import mnemos_native
    except ImportError as exc:  # pragma: no cover -- depends on the build
        raise ImportError(
            "mnemos_native is not importable. Build it with:\n"
            "    cargo build --profile release-ffi -p mnemos-ffi "
            "--features mnemos-keystore/insecure-software-keystore\n"
            "then copy target/release-ffi/mnemos_native.{dll,so} to python/mnemos_native.{pyd,so}"
        ) from exc
    return mnemos_native


def _training_sample(n: int) -> list[str]:
    """Synthetic documents for the initial partition table.

    Deterministic and vocabulary-sharing, so the sample has cluster structure. It exists only to
    place centroids before any real data arrives; the table is fixed from then on.
    """
    state = 0x9E37_79B9
    docs = []
    for _ in range(n):
        tokens = []
        for _ in range(8):
            state = (state * 1_103_515_245 + 12_345) & 0xFFFF_FFFF
            tokens.append(f"t{state % 4096}")
        docs.append(" ".join(tokens))
    return docs
