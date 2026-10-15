"""Mnemos -- Python layer.

This package is **outside the trusted computing base** for plaintext-residency purposes. The
security guarantees in the paper's threat model are made by ``crates/``; everything here handles
plaintext by design, and does so knowingly (the paper (Section 6)).

+-------------------+------------------------------------------------------------------+
| :mod:`params`     | Reads ``params.toml`` -- the same file Rust reads                 |
| :mod:`embed`      | Embedding + int8 quantisation                                     |
| :mod:`trace`      | The adversary's transcript. **Attacks import this and nothing else.** |
| :mod:`mcp_server` | The MCP surface a real agent stores memories through              |
+-------------------+------------------------------------------------------------------+

The engine itself is ``mnemos_native`` (the PyO3 extension from ``crates/mnemos-ffi``), landing
through the MCP server; the CLI is the other interface.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["embed", "mcp_server", "params", "trace"]
