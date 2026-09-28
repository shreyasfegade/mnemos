<div align="center">

# Mnemos

**Searchable encrypted memory for self-writing agents.**

*An attacker who can make an agent write can learn where things live —
and that knowledge reaches backwards into memories it never saw being written.*

<img src="paper/figures/fig1_transfer.png" alt="Structure transfer: earlier and later cohorts are exposed equally under a content-placed layout" width="100%">

**A content-placed encrypted store exposes memories written before a compromise exactly as much as
those written after it.** Being written first is not protection — a transfer lemma says so — unless
the agent's subject matter has drifted. The protection is a layout that carries no topic labels at
all.

[Paper](paper/main.pdf) · [Start here](ONBOARDING.md) · [Spec](the specification) · [Architecture](ARCHITECTURE.md) · [Threat model](THREAT-MODEL.md) · [Leakage profile](docs/leakage-profile.md) · [Status](the project notes)

`Status: revised for IEEE TDSC after external review; Mnemos v2 implemented` · `Started 2026-08-12`

</div>

---

## The 90-second version

An autonomous agent writes its own memory, continuously, from tasks its environment hands it. That
memory is a confidentiality surface nobody has modelled: everything the agent has ever figured out,
sitting on disk, searched constantly.

Encrypt it and you still have to *search* it, and searchable encryption leaks patterns. The
standard defence for continuously-written encrypted search is **forward privacy** — and it does
what it claims. It is also beside the point.

**Because the agent's environment can make it write.** Feed it a SQL task and it authors a SQL
memory, which lands in whichever encrypted partition holds SQL. Partitions are contiguous file
regions, so *growth is visible with no key and no cryptanalysis* — forward privacy constrains token
chains, not `stat()`. Batching writes into epochs hides which record in a flush was yours, but not
for long: repeat the injection and your partition shows up in **every** flush while co-flushed
organic writes vary, so intersecting the touched sets recovers it.

Now the part that matters. That map is not limited to what the attacker induced. **Partitions are
shared with everything written earlier**, so a map learned entirely after the compromise applies
backwards to memories the attacker never saw being written. A **transfer lemma** proves the two
cohorts are exposed *equally* under any content-placed layout, so the value of being written early
is zero — **provided the layout tells the adversary which partition holds a given earlier record.**
A growing layout does; a sealed fixed-size one does not, and that distinction turns out to carry the
argument. A real language-model agent writes a memory for **81%** of adversary-supplied tasks.

Drift looks like it protects the early cohort — advantage **0.31** under an i.i.d. split, **0.11**
under a chronological one — but it does not. The adversary chooses what it induces, and one that
induces with *archival* content reaches **+0.41** against **−0.07** for one using current content.
The fall was a property of the attacker's choice, not of the data.

Three channels reach the earlier cohort even when the layout withholds membership. **Retained file
versions** reconstruct pre-compromise writes: history independence constrains one representation
against the operations that made it, and says nothing about a *sequence* of them. A **refused
write** discloses that a topic has filled, and with it the earlier cohort's load. And an agent that
acts on what it recalls leaks through the **occurrence** of a write, which makes deduplication a
membership oracle for facts stored before the compromise.

That last one is a **separation, not an impossibility**: it is open for any store whose operations
depend on the content or occurrence of what the agent does, and closed for one that is
content-oblivious and clock-driven. So the repo ships a **design space**, not a construction:
**Mnemos-RS** (record store + index), **Mnemos-P** (sealed partition file, placement by content),
and **Mnemos-M** (same file, placement at random, search entirely in locked memory, clocked
flushes). Mnemos-M closes all three channels and is *faster* than searching on disk; what bounds it
is resident memory, and Mnemos-P covers the range above that bound with its residual leakage stated
and measured.

---

## What's in the box

| | |
|---|---|
| 🦀 **Rust crypto core** | XChaCha20-Poly1305 content encryption, HKDF key hierarchy, GGM puncturable PRF, and a locked 2 MiB arena where plaintext lives for microseconds and is zeroised on drop. Rust because the threat model *requires* that guarantee, and a GC'd runtime cannot make it true. |
| 🔐 **Hardware key custody** | Root key never leaves the TPM 2.0 / Secure Enclave. Windows CNG, Linux tpm2-tss, macOS Security.framework. The software fallback is compile-gated and refuses to run without an explicit "I accept this is insecure" env var. |
| 🔎 **Encrypted vector search** | IVF over int8-quantised embeddings. Every query reads exactly the same number of slots regardless of what you asked or how much matched — constant-volume by construction, so volumetric attacks have nothing to chew on. |
| 🧨 **The attacks, in-tree** | The structure-transfer attack (A4), IKK, the Cash et al. count attack, the snapshot attack (A3), and a volumetric attack aimed **at Mnemos itself**. Baselines ship in-tree too, so Figure 1 is `python -m eval.experiments.e4_transfer` away. |
| 📊 **The demo** | Two panels, one agent, one COMPROMISE button. On the left, a forward-private baseline gets de-anonymised in front of you and the agent's past reassembles itself. On the right, nothing resolves. A slider for the epoch schedule visibly widens and narrows the gap. |
| 🔌 **A real backend** | MCP server. A real agent stores its real memories in it and completes a real task suite. |

---

## Quick start

> Requires Rust (stable) and Python 3.11+. One-time: `rustup` from <https://rustup.rs>.

```bash
just setup && just test
```

```bash
just demo
```

```bash
just verify
```

```bash
just repro
```

`just repro` runs the load-bearing experiments on the public corpus (D2) and regenerates every
figure. It does not merely run them: **every headline number is checked against the range the
paper claims for it**, each range carrying a written justification, and the run exits non-zero if
any number falls outside. `just verify` re-checks the results already on disk in about a second.
The full sweep is `just repro-full`.

---

## Repository map

```
ONBOARDING.md         start here if you are joining or reviewing — 10 minutes, honest
the project notes             what is built vs specified, open debt, and a defect appendix
the specification            the spec — final, post-vet, and binding
ARCHITECTURE.md       how it is built, and which threat every mechanism answers
THREAT-MODEL.md       the τ-arrival adversary, dual-epoch security, the theorems
the evaluation plan         the evaluation *plan*, kept unedited, with a plan-vs-built table on top
params.toml           every tunable, in one place, read by code *and* docs *and* paper
docs/
  leakage-profile.md  the table CI asserts the implementation against
  findings/           F2–F6, dated, including the ones that were our own fault
  adr/                every locked decision, one file each
  glossary.md
crates/               Rust: core · crypto · keystore · store · index · engine · baselines · ffi · cli · bench
python/
  mnemos/             bindings, embeddings, MCP server
  eval/               attacks, experiments, figures, and repro.py (the range-checking harness)
demo/                 the live side-by-side compromise visualisation
paper/
  main.pdf            the paper
  sections/           one file per section
  tables/             GENERATED from criterion's output — never edited by hand
  figures/            generated by `just figures`
```

---

## Research

This repository is the artifact for a paper in preparation.

**Contribution.** We show that an adversary who can induce a self-writing agent to author memories
recovers the semantic structure of its encrypted store without any cryptanalysis; that this
structure applies backwards to memories written before the compromise, leaving pre-compromise data
94% as exposed as data written under total surveillance; that batching writes is insufficient to
stop it; and that constant-size partitions reduce it to chance. We formalise the required
properties as dual-epoch security and instantiate them in a working agent memory backend.

Findings are recorded as they land in [`docs/findings/`](docs/findings/).

**Positioning.** Closest neighbours are Compass (OSDI 2025) and Opal (2026), which build encrypted
searchable semantic memory but assume either a static corpus or an *uncompromised* enclave, and
give a single uniform guarantee with no pre/post-compromise distinction. Breach-resistant
structured encryption (PoPETs 2019) separates snapshot from persistent *adversaries*; we separate
two *cohorts of data* under one adversary. The agent-memory-poisoning literature (MINJA,
AgentPoison, MemoryGraft) attacks integrity; we protect confidentiality.

Target: PoPETs 2027.2, then a Q1 journal. Full citation set in [`paper/refs.bib`](paper/refs.bib).

---

## Design notes worth reading

- **`update` returns a new id.** Updates are `delete + put`, never in-place, because an in-place
  rewrite is a same-slot write — exactly the correlation signal a live adversary wants. A visible
  API cost paid for a security property.
- **The attacks cannot cheat.** They import `mnemos.trace`, never `mnemos.engine`, so they see
  precisely what the threat model says the adversary sees.
- **The leakage table is a test.** `tests/leakage_profile.rs` asserts the emitted event set matches
  `docs/leakage-profile.md` row by row. Leak something undocumented and the build fails.
- **History independence is property-tested.** For any two insertion orders of the same set, the
  final slab bytes must agree modulo nonces and dummy fill — a machine-checked instance of the
  paper's core hypothesis. It is *conditional*: a partition that overflows spills into a
  neighbour, and which records spill depends on arrival order. The observable form still holds
  unconditionally, because every partition is padded and every slot sealed.
- **The event stream is checked against the disk, not against itself.**
  `tests/channel_enumeration.rs` snapshots the store byte-for-byte, performs one operation, diffs
  the snapshot, and fails if any changed byte range lacks an event describing it. Both findings
  F5 and F6 were channels that emitted nothing, and a test suite that reads events cannot see
  those. Each test carries a negative control asserting it can still fail.

---

## Non-goals

Not a new cryptographic primitive · not a general-purpose vector database · not hiding corpus
existence or gross size · not defending against memory poisoning (that is an integrity problem;
we cite it, we do not solve it) · not multi-host · not requiring a TEE — *not* needing one is a
differentiator.

---

## Licence

Dual-licensed under [Apache-2.0](LICENSE-APACHE) and [MIT](LICENSE-MIT), at your option.
