# EUREKA benchmarks

This repository holds the benchmark scripts used to evaluate EUREKA (Edge Utility for
REwiring and Knowledge Aware graph representation) against a set of paper faithful
baseline implementations, on the same datasets, splits and seeds.

Two scripts are included:

- `eureka_benchmark.py`: runs EUREKA itself. It compares two configurations,
  the original graph with original features, against the EUREKA rewired graph with
  the learned representation, across several downstream classifiers and seeds.
- `benchmark_others.py`: runs seven baseline methods (IDGL, GADC, LPkG, GRAPHITE,
  DHGR, FoSR, ComFy), each using only the classifier its own paper proposes, on the
  same datasets and splits.

Both scripts share the same dataset loading logic, write results incrementally to a
JSONL file so an interrupted run can resume, and produce summary tables and plots.

## What EUREKA does

Given a graph with adjacency A and node features X, EUREKA builds a structural
representation S with a FUSE style modularity power iteration, a complementary
residual feature representation H orthogonal to S, and combines them into
E_rep = [S | alpha*H]. A small network f_theta scores each edge using a four view
descriptor (structural, residual feature, and two neighbourhood distribution views),
and the adjacency is reweighted by that score. Low utility edges are pruned. The
target for training f_theta is representation agreement between the two endpoints,
not label homophily, so a heterophilic edge can still be kept if it is consistent
under the learned representation. The rewired graph is then re-embedded to produce
the final S*, H* and E* used by the downstream classifier.

## Datasets

`eureka_benchmark.py` evaluates on:

Actor, Squirrel-F, Chameleon-F, Roman-empire, Amazon-ratings, Tolokers, and four
synthetic graphs generated in-script (HSBM-MED, STRUC-HET, FEAT-HET, MIXED-SIG).

`benchmark_others.py` evaluates on the same real datasets plus the same four
synthetic ones.

The synthetic datasets are generated directly in each script and need no external
data. The real datasets are downloaded automatically through PyTorch Geometric on
first use.

## Something to take from outside this repo

Two of the components here rely on external code that is not included in this
repository and needs to be added separately before results are fully faithful:

1. **The real FUSE embedder.** In `eureka_benchmark.py`, the `fuse()` function
   computes the structural representation S with a self-contained modularity power
   iteration (Krylov solver, falling back to an explicit power iteration, falling
   back to a dense solve). This is a workable stand-in, but the intended embedder is
   `fsgb.embedders.fuse.FUSE`, an external package. To use the real embedder, replace
   the body of `fuse()` with a call to
   `fsgb.embedders.fuse.FUSE(dim=dim, ...).fit_transform(gd)` on a `GraphData` object
   built from the edge index and weights, and return its orthonormal output. The same
   applies to `residual_augment()`, which mirrors `fsgb.residual_augment`.

2. **The official DHGR repository.** In `benchmark_others.py`, `run_dhgr()` and the
   code around it try to import `GraphLearner.ModelHandler` from the official DHGR
   codebase (https://github.com/wendongbi/DHGR). This repository is not vendored
   here. Clone it and either set the `DHGR_ROOT` environment variable to its path, or
   place a folder named `DHGR` next to these scripts (the loader also checks one and
   two levels up). If DHGR is not found, the DHGR baseline is skipped automatically
   and every other method still runs.

## The shared `common` package

Both scripts import a local `common` package (`common.metrics`, `common.progress`,
`common.checkpoint`, `common.reporting`, `common.plotting`) that is expected to sit
next to them but is not part of this upload. It should provide, at minimum:

- `metrics.accuracy_and_macro_f1(preds, labels)`
- `progress`: a tracker with `tick_done(elapsed, note=...)`, `tick_skipped(...)` and
  `finish()`
- `checkpoint`: a results store with `append(record)` and `exists(**keys)`, backed by
  the JSONL results file
- `reporting.write_all_csvs(...)`, `reporting.print_dataset_tables(...)`,
  `reporting.print_average_table(...)`
- `plotting.generate_all_figures(...)`

Add this package to the repository before running either script.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torch-geometric numpy scipy scikit-learn networkx pyyaml tqdm
```

Match the torch and torch-geometric versions to your CUDA setup if running on GPU.

## Usage

Run EUREKA:

```bash
python eureka_benchmark.py
python eureka_benchmark.py --resume --device cuda
python eureka_benchmark.py --datasets Actor Chameleon-F --classifiers gcn h2gcn
python eureka_benchmark.py --smoke_test
```

Run the baselines:

```bash
python benchmark_others.py
python benchmark_others.py --datasets Actor Squirrel-F --methods idgl gadc
python benchmark_others.py --smoke_test
python benchmark_others.py --seeds 42 0 1 --device cuda
```

Both scripts resume by default: reruning the same command skips any
(dataset, method, classifier, seed) combination already recorded in the results
file. Pass `--no_resume` to force a clean rerun.

## Output

`eureka_benchmark.py` writes to `eureka_benchmark_results/`:

- `results.jsonl`: one record per (dataset, method, classifier, seed) run
- `graph_quality.csv`: graph quality metrics (homophily, edge purity, spectral
  smoothness, conductance, and more) for the original and rewired graphs
- `tables/` and `plots/`: per-dataset summaries, an average-over-datasets table, and
  figures, refreshed after every dataset

`benchmark_others.py` writes to `others_benchmark_results/`:

- `results_faithful.jsonl`: one record per (dataset, method, seed) run
- `summary_faithful.csv`: mean and standard deviation per (dataset, method)
- `splits/`: the exact train, validation and test masks used for each dataset, saved
  once per dataset so a reported result can be traced back to its split

## Notes

- Every downstream classifier is trained with a fixed seed per run, and both scripts
  seed Python, NumPy, PyTorch and cuDNN for reproducibility.
- `--smoke_test` shortens epoch counts and restricts seeds so a full pipeline check
  runs quickly before committing to a full sweep.
- Failed runs (for example a baseline that errors on a particular dataset) are logged
  and skipped rather than stopping the whole sweep, and are retried automatically on
  the next resume since nothing was written for them.
