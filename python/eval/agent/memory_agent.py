"""A mem0-style memory pipeline: the real write path the attack assumes.

The structure-transfer attack rests on one premise the earlier paper asserted but never showed:
that an adversary who supplies a document causes the agent to *write a memory about that topic*.
Here the agent actually does the writing, with a real LLM, using mem0's own two-stage design:

1. **Extraction.** Given a turn (an email, an issue, a conversation snippet), the model extracts a
   list of atomic facts worth remembering, or an empty list. This is mem0's ``FACT_RETRIEVAL``
   stage; the prompt below is adapted from mem0's published one (``mem0/configs/prompts.py`` in
   the ``mem0ai`` package), kept faithful to its instructions.
2. **Update.** For each candidate fact, the model is shown the most similar existing memories and
   decides ADD, UPDATE, or NOOP. This is mem0's ``DEFAULT_UPDATE_MEMORY`` stage.

The point of the experiment is the numbers this produces: how often a turn yields a write, how
many memories per turn, and whether the written memory lands in the same partition as the source
topic. None of it is assumed; all of it is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eval.agent.llm import LLM

EXTRACTION_SYSTEM = """You are a memory-extraction module for a personal AI assistant. \
From the user's message you extract atomic facts, preferences, plans, and relationships that are \
worth remembering for future conversations. Extract only information that is durable and useful \
later; ignore small talk, acknowledgements, and one-off pleasantries.

Return a JSON object {"facts": [...]} where each fact is a short self-contained sentence. If \
there is nothing worth remembering, return {"facts": []}. Do not invent information that is not \
present in the message."""

UPDATE_SYSTEM = """You maintain an AI assistant's long-term memory. You are given a candidate new \
fact and the existing memories most similar to it. Decide the single best action:

- "ADD": the fact is new information not covered by any existing memory.
- "UPDATE": the fact refines or corrects one existing memory; give its index.
- "NOOP": the fact is already fully captured by an existing memory.

Return JSON {"action": "ADD"|"UPDATE"|"NOOP", "index": <int or null>, "text": "<the memory text \
to store, for ADD or UPDATE>"}."""


@dataclass
class MemoryRecord:
    mid: str
    text: str
    source_topic: str
    turn_id: str
    embedding: np.ndarray


@dataclass
class Turn:
    turn_id: str
    topic: str
    text: str
    role: str = "user"


@dataclass
class AgentStats:
    turns: int = 0
    turns_with_write: int = 0
    facts_extracted: int = 0
    adds: int = 0
    updates: int = 0
    noops: int = 0
    partition_hits: int = 0
    partition_total: int = 0
    per_turn_writes: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["write_probability"] = self.turns_with_write / self.turns if self.turns else 0.0
        d["mean_writes_per_turn"] = (
            float(np.mean(self.per_turn_writes)) if self.per_turn_writes else 0.0
        )
        d["partition_hit_rate"] = (
            self.partition_hits / self.partition_total if self.partition_total else 0.0
        )
        return d


class MemoryAgent:
    """Runs the two-stage pipeline against a store, over a stream of turns.

    ``store`` must expose ``put(text, embedding, meta) -> id``, ``search(embedding) -> hits``,
    and ``update(id, text, embedding, meta) -> id``. Both the v1 and v2 FFI engines qualify;
    for the induction study a lightweight in-process store is enough and is provided by the
    caller. ``embed`` maps a list of texts to L2-normalised vectors. ``assign`` maps a vector to
    its ground-truth partition, used only to score partition hits, never shown to the model.
    """

    def __init__(self, llm: LLM, store, embed, assign, top_k: int = 5) -> None:
        self.llm = llm
        self.store = store
        self.embed = embed
        self.assign = assign
        self.top_k = top_k
        self.records: dict[str, MemoryRecord] = {}
        self.stats = AgentStats()

    def _extract(self, turn: Turn) -> list[str]:
        obj = self.llm.chat_json(EXTRACTION_SYSTEM, turn.text)
        items = obj.get("facts") if isinstance(obj, dict) else obj
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for f in items:
            if isinstance(f, str):
                s = f.strip()
            elif isinstance(f, dict):
                # A small model sometimes emits {"key": "value"} facts; flatten to a sentence.
                s = "; ".join(f"{k}: {v}" for k, v in f.items()).strip()
            else:
                s = str(f).strip()
            if s:
                out.append(s)
        return out[:8]

    def _similar(self, vec: np.ndarray) -> list[MemoryRecord]:
        if not self.records:
            return []
        ids = list(self.records)
        mat = np.vstack([self.records[i].embedding for i in ids])
        sims = mat @ vec
        order = np.argsort(-sims)[: self.top_k]
        return [self.records[ids[i]] for i in order]

    def process(self, turn: Turn) -> None:
        self.stats.turns += 1
        facts = self._extract(turn)
        self.stats.facts_extracted += len(facts)
        writes = 0
        if not facts:
            self.stats.per_turn_writes.append(0)
            return
        vecs = self.embed(facts)
        for fact, vec in zip(facts, vecs, strict=True):
            neighbours = self._similar(vec)
            action, text = self._decide(fact, neighbours)
            if action == "NOOP":
                self.stats.noops += 1
                continue
            evec = self.embed([text])[0] if text != fact else vec
            part = int(self.assign(evec))
            self.stats.partition_total += 1
            if part == self._topic_partition(turn.topic):
                self.stats.partition_hits += 1
            if action == "UPDATE" and neighbours:
                mid = self.store.update(
                    neighbours[0].mid, text, evec.tolist(), f'{{"topic": "{turn.topic}"}}'
                )
                self.records.pop(neighbours[0].mid, None)
                self.stats.updates += 1
            else:
                mid = self.store.put(text, evec.tolist(), f'{{"topic": "{turn.topic}"}}')
                self.stats.adds += 1
            self.records[mid] = MemoryRecord(mid, text, turn.topic, turn.turn_id, evec)
            writes += 1
        if writes:
            self.stats.turns_with_write += 1
        self.stats.per_turn_writes.append(writes)

    def _decide(self, fact: str, neighbours: list[MemoryRecord]) -> tuple[str, str]:
        if not neighbours:
            return "ADD", fact
        listing = "\n".join(f"{i}: {n.text}" for i, n in enumerate(neighbours))
        obj = self.llm.chat_json(
            UPDATE_SYSTEM, f"Candidate fact:\n{fact}\n\nExisting memories:\n{listing}"
        )
        if not isinstance(obj, dict):
            return "ADD", fact
        action = str(obj.get("action", "ADD")).upper()
        if action not in {"ADD", "UPDATE", "NOOP"}:
            action = "ADD"
        text = str(obj.get("text") or fact).strip() or fact
        return action, text

    # The topic->partition map is built lazily from the majority partition of a topic's memories,
    # so "partition hit" means "this write landed where this topic's memories congregate", which
    # is exactly what the attack needs to be true.

    def _topic_partition(self, topic: str) -> int:
        vecs = [r.embedding for r in self.records.values() if r.source_topic == topic]
        if not vecs:
            return -1
        parts = [int(self.assign(v)) for v in vecs]
        return max(set(parts), key=parts.count)
