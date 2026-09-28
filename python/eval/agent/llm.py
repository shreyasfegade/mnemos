"""LLM backends for the agent-memory experiments.

Two backends behind one interface:

``ollama``
    A local model served by Ollama (``qwen2.5:7b-instruct`` and ``llama3.1:8b`` are pulled on
    this machine). No API key, runs on the RTX 5070.

``openai``
    Any OpenAI-compatible chat endpoint. DeepSeek is reached by setting ``DEEPSEEK_API_KEY`` and
    ``base_url=https://api.deepseek.com``; the harness picks this up automatically when the key is
    present, so a second, non-local model needs only an environment variable.

Both return assistant text, and :func:`chat_json` additionally parses a JSON object out of the
reply, tolerating the markdown fences small models like to add.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


@dataclass
class LLM:
    """A chat model. ``name`` is recorded in every result so a run names its own model."""

    backend: str
    model: str
    base_url: str = ""
    _client: object = None

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.model}"

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        if self.backend == "ollama":
            import ollama

            r = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return r["message"]["content"]
        if self.backend == "openai":
            from openai import OpenAI

            if self._client is None:
                self._client = OpenAI(base_url=self.base_url or None)
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content or ""
        raise ValueError(f"unknown backend {self.backend!r}")

    def chat_json(self, system: str, user: str, **kw) -> object:
        raw = self.chat(system, user + "\n\nRespond with JSON only.", **kw)
        return parse_json(raw)


def parse_json(raw: str) -> object:
    """Extract the first JSON value from a possibly fenced reply; ``None`` if there is none."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        i = raw.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(raw)):
            if raw[j] == opener:
                depth += 1
            elif raw[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[i : j + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def available() -> list[LLM]:
    """Every usable model on this machine, local first."""
    out: list[LLM] = []
    try:
        import ollama

        have = {m.get("model", m.get("name", "")) for m in ollama.list().get("models", [])}
        for want in ("qwen2.5:7b-instruct", "llama3.1:8b"):
            if (
                any(h.startswith(want.split(":")[0]) and want.split(":")[1] in h for h in have)
                or want in have
            ):
                out.append(LLM("ollama", want))
    except Exception:
        pass
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)
        out.append(LLM("openai", "deepseek-chat", base_url="https://api.deepseek.com"))
    if os.environ.get("OPENAI_API_KEY") and not key:
        out.append(LLM("openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")))
    return out
