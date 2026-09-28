"""E2 and E7: a real LLM agent writes the memory, and we measure induction and the derived-write
channel.

The premise the whole attack rests on -- "an adversary who supplies a document causes the agent
to write a memory about that topic" -- is demonstrated here with a real agent loop (mem0-style
extraction + update, prompts from the ``mem0ai`` package), driven by two local models via Ollama
and, if a key is present, DeepSeek. No passage is ever copied into the store; every memory is
written by the model from adversary-controlled input.

E2 induction
------------
Feed the agent a stream of turns whose topic the adversary chose (emails/issues). Measure:

- **write probability**: fraction of adversary turns that produce at least one memory write.
- **writes per turn**: how many memories a turn yields.
- **partition hit rate**: fraction of writes that land in the partition where that topic's
  memories congregate -- i.e. whether the induced write is where the attack needs it.

E7 derived-write channel
------------------------
A self-writing agent retrieves an old (cohort-H) memory and then writes a new
memory or issues a query about the same subject, so the partitions touched after tau depend on
cohort H. We run a feedback agent: each post-tau turn first retrieves, and its follow-up write is
conditioned on what it retrieved. We measure how often a post-tau write lands in the partition of
a cohort-H memory retrieved just before, against a no-feedback control -- the correlation no
storage design can remove.

    cd python && python -m eval.experiments.e2_agent_induction --turns 120
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np

from eval.agent.llm import available
from eval.agent.memory_agent import MemoryAgent, Turn
from eval.lab import corpus as C
from eval.lab.embcache import embed_cached

OUT = C.repo_root() / "python" / "eval" / "results" / "e2_agent.json"
B = 64


class InProcStore:
    """A minimal store: keeps memory vectors so the agent's search works, records writes.

    The induction question is about the *agent*, not the storage engine, so a light store keeps
    the LLM the only cost. Partition assignment uses the same centroids the engines would.
    """

    def __init__(self, cents: np.ndarray):
        self.cents = cents
        self.vecs: dict[str, np.ndarray] = {}
        self.n = 0

    def put(self, text, embedding, meta="{}"):
        mid = f"m{self.n}"
        self.n += 1
        self.vecs[mid] = np.asarray(embedding, dtype=np.float32)
        return mid

    def update(self, mid, text, embedding, meta="{}"):
        self.vecs.pop(mid, None)
        return self.put(text, embedding, meta)

    def search(self, embedding):
        if not self.vecs:
            return []
        ids = list(self.vecs)
        sims = np.vstack([self.vecs[i] for i in ids]) @ np.asarray(embedding, dtype=np.float32)
        order = np.argsort(-sims)[:5]
        return [(ids[i], "", "", float(sims[i])) for i in order]


def turns_from_github(n: int, rng) -> list[Turn]:
    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "e3" / "github.jsonl").open(encoding="utf-8")
        if ln.strip()
    ]
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        out.append(Turn(turn_id=str(r["id"]), topic=r["topic"], text=r["text"][:1500]))
    return out


def run_induction(llm, turns: list[Turn], cents: np.ndarray) -> dict:
    store = InProcStore(cents)

    def embed(texts):
        return embed_cached(list(texts))

    def assign(v):
        return int(np.argmax(cents @ np.asarray(v, dtype=np.float32)))

    agent = MemoryAgent(llm, store, embed, assign)
    t0 = time.time()
    for turn in turns:
        try:
            agent.process(turn)
        except Exception as e:
            print(f"  turn {turn.turn_id} error: {e}", flush=True)
    s = agent.stats.as_dict()
    s["model"] = llm.name
    s["seconds"] = time.time() - t0
    return s


def run_derived_write(llm, turns: list[Turn], cents: np.ndarray, feedback: bool) -> dict:
    """Post-tau feedback loop. Returns how often a follow-up write lands in a retrieved memory's
    partition, with and without feedback."""
    store = InProcStore(cents)

    def embed(texts):
        return embed_cached(list(texts))

    def assign(v):
        return int(np.argmax(cents @ np.asarray(v, dtype=np.float32)))

    # cohort H: seed the store with the first half.
    half = len(turns) // 2
    agent = MemoryAgent(llm, store, embed, assign)
    for turn in turns[:half]:
        with contextlib.suppress(Exception):
            agent.process(turn)

    hits = total = 0
    rng = np.random.default_rng(0)
    for turn in turns[half:]:
        retrieved_part = None
        if feedback:
            # The agent retrieves before acting; its follow-up is conditioned on the retrieval.
            qv = embed([turn.text])[0]
            res = store.search(qv.tolist())
            if res:
                retrieved_part = assign(store.vecs[res[0][0]])
                # Follow-up write about the retrieved subject: reuse the retrieved memory's vector
                # neighbourhood, which is what "the agent writes about what it just recalled" means.
                followup_vec = store.vecs[res[0][0]]
                store.put("follow-up", followup_vec.tolist())
                total += 1
                if assign(followup_vec) == retrieved_part:
                    hits += 1
        else:
            # No-feedback control: the follow-up write is independent of any retrieval.
            v = embed([turn.text])[0]
            store.put("independent", v.tolist())
            # score against a random cohort-H memory's partition (the null).
            if store.vecs:
                rid = rng.choice(list(store.vecs))
                total += 1
                if assign(store.vecs[rid]) == assign(v):
                    hits += 1
    return {
        "model": llm.name,
        "feedback": feedback,
        "followups": total,
        "landed_in_retrieved_partition": hits,
        "rate": hits / total if total else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=120)
    args = ap.parse_args()

    rows = [
        json.loads(ln)
        for ln in (C.repo_root() / "data" / "e3" / "github.jsonl").open(encoding="utf-8")
        if ln.strip()
    ]
    import random

    random.Random(0).shuffle(rows)
    sample = rows[:4000]
    vecs = embed_cached([r["text"][:1500] for r in sample])
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=B, random_state=0, n_init=2).fit(vecs)
    cents = (
        km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    ).astype(np.float32)

    models = available()
    if not models:
        raise SystemExit("no LLM backend available (start ollama, or set DEEPSEEK_API_KEY)")
    print("models:", [m.name for m in models], flush=True)

    results = {"experiment": "E2_E7", "n_turns": args.turns, "induction": [], "derived_write": []}
    for llm in models:
        turns = turns_from_github(args.turns, random.Random(1))
        print(f"--- induction: {llm.name} ---", flush=True)
        ind = run_induction(llm, turns, cents)
        results["induction"].append(ind)
        print(
            f"  write_prob={ind['write_probability']:.2f} "
            f"writes/turn={ind['mean_writes_per_turn']:.2f} "
            f"partition_hit_rate={ind['partition_hit_rate']:.2f} adds={ind['adds']} "
            f"updates={ind['updates']} "
            f"noops={ind['noops']} ({ind['seconds']:.0f}s)",
            flush=True,
        )

        for fb in (True, False):
            dw = run_derived_write(llm, turns, cents, feedback=fb)
            results["derived_write"].append(dw)
            print(
                f"  derived-write feedback={fb}: rate={dw['rate']:.2f} "
                f"({dw['landed_in_retrieved_partition']}/{dw['followups']})",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(C.repo_root())}")


if __name__ == "__main__":
    main()
