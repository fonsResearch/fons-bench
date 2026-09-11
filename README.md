# fons-bench

**Confidence signals from protein-ligand co-folding models do not tell you when the model is wrong on unfamiliar targets.** Calibrate an accept/reject threshold the standard way, on a random split, and it meets its 10% error target overall while delivering 36% error on the cases with the least similarity to anything in the training window. Restoring the guarantee there is not a matter of accepting fewer predictions: at a 10% target the low-similarity regime is uncertifiable outright, and honest group-conditional calibration responds by rejecting essentially everything.

That matters because low similarity is the regime drug discovery operates in. A benchmark number computed on the full distribution is dominated by cases the model has effectively already seen: 27.5% of the post-2021 systems have a ligand identical to one released before the model's training cutoff, and those supply 34% of all correct predictions.

## What this is

A measurement instrument, not a model. It takes the [Runs N' Poses](https://github.com/plinder-org/runs-n-poses) release (2,600 protein-ligand systems released after the co-folding training cutoffs, with predictions from eight methods) and asks one question:

> Given a model's confidence signal, at what accept/reject threshold can we guarantee the error rate among accepted predictions stays below a target level, and does that guarantee survive distribution shift?

Everything is derived from one file, the master table, one row per `(group_key, model, seed, sample)`. It and every analysis output are build products: this repository holds the code that produces them, and the pipeline below writes them into `results/`.

## Use

The analysis code is in this repository and is not published as a package. Clone and run it
with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Score your own predictions against the calibrated thresholds. No GPU, and no benchmark
download: the thresholds live in `src/fons_bench/artifacts/calibration.json`.

```sh
uv run fons-bench evaluate --predictions my_preds.csv --model af3 --alpha 0.1
```

`my_preds.csv` needs `system_id` and `signal` (higher = more confident). Add `pocket_similarity`
(SuCOS-pocket similarity, 0-100, to the nearest structure released before your training cutoff)
to get the group-conditional rule; without it only the marginal threshold applies, and the
marginal threshold is exactly what this benchmark shows you should not trust. Add `correct`
(0/1) to get realised error back.

Regenerate every figure from the master table:

```sh
uv run fons-bench reproduce
```

## Findings

| | |
|---|---|
| Marginal calibration, pooled | 0.102 realised error against a 0.100 target |
| Marginal calibration, lowest-similarity bin | 0.363 realised error; violates in 86% of splits |
| Group-conditional, same bin | accepts nothing in 95% of splits at alpha 0.1 |
| Best confidence signal | a chain-pair ipTM variant; normalised excess AURC 0.20 to 0.61 |
| Accuracy on identical-ligand cases | 0.841, against 0.615 on everything else |

The kill condition was written and timestamped before any curve was computed. It did not
fire: all seven models failed it. The pipeline below regenerates it along with every report.

## Reproducing

The master table itself is not committed (65 MB). It rebuilds byte-identically in about
11 seconds once the source data is fetched:

```sh
uv sync
uv run python scripts/01_fetch_data.py      # Zenodo fetch + sha256 verify
uv run python -m fons_bench.cli build-table # assemble master.parquet
uv run python scripts/04_risk_coverage.py   # risk-coverage curves, AURC, kill check
uv run python scripts/05_calibrate.py       # the three conformal calibrations
uv run python scripts/06_alpha_sweep.py     # cost of the guarantee vs target error rate
uv run python scripts/08_phase4.py          # matched abstention, leakage decomposition
uv run python scripts/12_robustness_ligand_axis.py  # same result on the ligand axis
uv run python scripts/07_figures.py         # all figures
```

Every number in the reports traces to one of these scripts. No notebooks in the critical path.

## Data and licensing

Derived from the Runs N' Poses release (Zenodo record 18366081, concept DOI
[10.5281/zenodo.14794785](https://doi.org/10.5281/zenodo.14794785)). Rows for `af3` and
`af3_no_template` derive from AlphaFold 3 output and carry
[Google's AlphaFold 3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md)
(non-commercial). Code is licensed separately; see `LICENSE`.
