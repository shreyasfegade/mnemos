"""Mnemos -- Python layer.

This package is **outside the trusted computing base** for plaintext-residency purposes. The
security guarantees in ``THREAT-MODEL.md`` are made by ``crates/``; everything here handles
plaintext by design, and does so knowingly (``ARCHITECTURE.md`` s6.3, ADR-0001).

+-------------------+------------------------------------------------------------------+
| :mod:`params`     | Reads ``params.toml`` -- the same file Rust reads                 |
| :mod:`embed`      | Embedding + int8 quantisation                                     |
| :mod:`trace`      | The adversary's transcript. **Attacks import this and nothing else.** |
| :mod:`mcp_server` | The MCP surface a real agent stores memories through              |
+-------------------+------------------------------------------------------------------+

The engine itself is ``mnemos_native`` (the PyO3 extension from ``crates/mnemos-ffi``), landing
at Stage 4. Until then the CLI is the interface.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["embed", "mcp_server", "params", "trace"]
