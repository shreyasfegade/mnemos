# Mnemos

Code, data and results for the paper **Historical Confidentiality for Encrypted Agent Memory:
Dual-Epoch Security and the Channels That Defeat It** ([`paper/main.pdf`](paper/main.pdf),
supplementary material in [`paper/supplement.pdf`](paper/supplement.pdf)).

AI agents write their own long-term memory and retrieve it by semantic search. The paper asks what
an adversary who obtains that storage midway (a synchronized folder, a backup, repeated disk images)
learns about memories written *before* the compromise. Encryption, snapshot security and history
independence do not answer that question. The paper defines dual-epoch security, shows four channels
that reach earlier records (structure transfer, retained versions, refusals, and derived writes),
proves that against a self-writing agent a store is secure exactly when what it reveals is
independent of what the agent does, and evaluates three designs.

## Layout

```
crates/                 Rust implementation
  mnemos-core           shared types, parameters, the leakage event bus
  mnemos-crypto         AEAD, key hierarchy, PRF, locked plaintext arena
  mnemos-keystore       hardware key custody (Windows CNG/TPM), gated software fallback
  mnemos-store          history-independent slab, fixed-length WAL, manifest
  mnemos-index          encrypted IVF index with deletion tokens
  mnemos-engine         Mnemos-RS: record store and index
  mnemos-v2             Mnemos-P: one sealed partition file, placement by content
  mnemos-mem            Mnemos-M: memory-resident index, random placement, clocked sweep flush
  mnemos-baselines      plaintext, naive, Sophos and Diana baselines
  mnemos-ffi            Python bindings (PyO3)
  mnemos-cli, -bench    command-line tool and Criterion benchmarks
python/
  mnemos/               bindings wrapper, embeddings, MCP server
  eval/attacks/         the attacks, which see only the adversary's event stream
  eval/exp2/            the paper's experiments (E1-E33), tables, numbers and verifier
  eval/e8/              the pgvector and deployed-store harness (runs under WSL)
  eval/results/         every result the paper reports, as JSON
data/                   corpus scripts, the agent-written memories and the issue corpus
docs/leakage-profile.md what each operation reveals; a test checks the engine against it
paper/                  LaTeX sources, generated tables, and the built PDFs
params.toml             every tunable parameter, read by the code and the paper
```

## Requirements

- Rust (stable, see `rust-toolchain.toml`) and Python 3.11+.
- [`just`](https://github.com/casey/just) to run the recipes below (optional; each recipe is a
  short command you can run directly).
- For the paper: a TeX distribution with `pdflatex` and `bibtex`.
- Some experiments need extra tools: Ollama with local models (the agent and library experiments),
  WSL with PostgreSQL, pgvector and gocryptfs (the deployed-store experiments), restic and Syncthing
  (the version-retention experiments). Each script says what it needs at the top.

## Reproducing the paper

```bash
just setup        # Rust toolchain, Python environment, the Python extension module
just test         # Rust and Python test suites, including the leakage-profile and audit tests
just verify       # check every stored result against the range the paper claims for it
just paper        # regenerate tables, numbers and figures from the results, then build the PDFs
```

`just verify` reads `python/eval/results/` and needs no experiment to run. Every number in the
paper's prose is a macro generated from those files (`paper/tables/numbers.tex`), so a rebuilt PDF
cannot quote a number the results do not contain.

To rerun experiments rather than check stored results, `just exp2` runs the ones that need no
language model or WSL, and `just overnight` runs the full set in parallel lanes (about 2.5 hours
on the machine described in the paper). The E-numbers match the paper and its supplementary
material, which maps each experiment to its table or figure.

### Registered predictions

Several experiments were registered before they ran: the prediction and the check that tests it
were committed to `python/eval/exp2/verify_r3.py` or `verify_r4.py` first, so the commit history
dates each one. `python -m eval.exp2.registered_table` reads the original ranges back from those
commits and produces the supplement's table of every registered outcome, failed ones included.

## Data

`data/fetch_d2.py` downloads the BEIR domains (corpus D2) and `data/generate_data.py` builds the
synthetic procedural-memory corpus (D3). `data/e3/build_corpora.py` builds the timestamped issue
corpus and the other e3 corpora; the issue corpus and the memories a local model wrote from it
(`data/e3/github.jsonl`, `data/e3/agent_memories.jsonl`) are committed because the second cannot
be regenerated exactly.

## Security notes

The software keystore exists for CI and is compiled out by default; it also refuses to start
without `MNEMOS_I_ACCEPT_INSECURE=1`. Results produced with it are not security results. The
threat model and its assumptions are in Section 3 of the paper.

## Licence

Dual-licensed under [Apache-2.0](LICENSE-APACHE) and [MIT](LICENSE-MIT), at your option.
