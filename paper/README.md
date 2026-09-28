# The paper

**Title:** Structure Transfer Attacks and Dual-Epoch Security for Encrypted Agent Memory
**Target:** IEEE Transactions on Dependable and Secure Computing (TDSC).

## Building

```bash
cd paper && latexmk -pdf main.tex
```

The manuscript uses the IEEEtran Computer Society journal class in single-column review layout
and the `IEEEtran` bibliography style. `refs.bib` lists only cited works.

A Word version for review is kept at `dist/main.docx`. It is exported from the same LaTeX source
and is not an independent draft; edit the LaTeX and re-export.

## Structure

| File | Section |
|---|---|
| `sections/00-abstract.tex` | Abstract |
| `sections/01-introduction.tex` | 1 Introduction |
| `sections/02-background.tex` | 2 Background and Related Work |
| `sections/03-threat-model.tex` | 3 System and Threat Model |
| `sections/04-structure-transfer.tex` | 4 The Structure-Transfer Attack |
| `sections/05-dual-epoch.tex` | 5 Dual-Epoch Security (definition, Theorem 1, separations) |
| `sections/06-mnemos.tex` | 6 Mnemos |
| `sections/07-evaluation.tex` | 7 Evaluation |
| `sections/08-discussion.tex` | 8 Discussion (validation, residual leakage, limitations, ethics) |
| `sections/09-conclusion.tex` | 9 Conclusion |
| `sections/10-appendix-proof.tex` | Appendix A: proof of Theorem 1 |

## Generated inputs

- `tables/perf.tex` and `tables/write.tex` are produced by
  `python -m eval.figures.emit_perf_table` from `target/criterion`. Do not edit them by hand.
- `figures/fig1_transfer.pdf` and `figures/fig2_threat_model.pdf` are produced by
  `python -m eval.figures.fig1_transfer` and `python -m eval.figures.fig2_threat_model`.
  Figure 1 plots the actual number of induced writes (injections per topic times the number of
  topics), not the `x` label recorded in the E4 results file.

## Correspondence between results and experiments

| Paper element | Experiment | Results file |
|---|---|---|
| Table 6, Fig. 2 | E4 (simulation) | `e4_transfer_bge_d2-beir.json`, `e4_transfer_bge.json` |
| Table 7 | E4 epoch sweep | same files, `axis = "epoch_writes"` |
| Section 7.4 | E3 sensitivity sweep | `e3_sensitivity.json` |
| Table 8 | E9 (compiled engine) | `e9_transfer_live.json` |
| Section 7.6, volume | E5 | `e5_volumetric.json` |
| Table 9 | E10 | `e10_probe_spread.json` |
| Tables 10 and 11 | Criterion benchmarks | `target/criterion` |
| Table 12 | E7 | `e7_agent_utility.json` |

The previous PoPETs-formatted draft is preserved in `dist/mnemos-paper-source.zip`.
