# LinkPlace

LinkPlace is the reproducibility release for connectivity-aware macro
placement. The release uses two public method names:

- **LinkPlace-C** (`linkplace-c`): decomposes the macro netlist into connected
  components and trains/places large components sequentially.
- **LinkPlace-M** (`linkplace-m`): trains one monolithic PPO policy over the
  same selected macros and ordering rules.

The code in this directory is a clean publication staging tree. It does not
contain private server paths, raw benchmarks, trained checkpoints, or the full
18 GB experiment archive.

## Repository layout

```text
linkplace/              Core method, models, evaluator, and CLI
tools/                  Queue, aggregation, dataset-import, and QA utilities
tests/                  Unit and PPO-equivalence tests
configs/                Machine-readable formal protocol
docs/                   Protocol, dataset, and reproduction notes
artifacts/tables/       Small paper-facing CSV/JSON artifacts only
environment/            Environment reconstruction notes and lock-file slots
```

## Installation

Create a clean Python environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Install a CUDA-enabled PyTorch build that matches your driver before running
formal GPU experiments. Exact server package exports, when available, belong
under `environment/` and take precedence over the broad development bounds in
`pyproject.toml`.

## Quick checks

```bash
python -m unittest discover -s tests -v
python -m linkplace.runner --help
```

After preparing benchmark caches (see `docs/DATASETS.md`):

```bash
python -m linkplace.runner --cache-root datasets/cache inspect adaptec1
```

<!-- README_RESULTS:BEGIN -->
## Detailed experimental results

This section is generated from the versioned CSV files under [`artifacts/tables/`](artifacts/tables/) by [`tools/render_readme_results.py`](tools/render_readme_results.py). It intentionally exposes more numerical detail than the compact paper tables. All reported placements use the same CompRes legality/HPWL evaluator; lower HPWL and RUDY are better. `±` denotes the sample standard deviation across seeds. Absolute values should be compared within a circuit, not across unrelated benchmarks with different coordinate scales.

### Coverage and protocol

| Result family | Public records | Protocol | Terminal outcome |
|---|---:|---|---|
| LinkPlace-C main | 85 | 17 circuits × seeds 999–1003 | 85/85 complete legal layouts |
| Dual-grid ablation | 240 | 8 ISPD2005 × 2 grids × 3 variants × 5 seeds | 188/240 legal; All-greedy failures retained |
| LinkPlace-M ICCAD2015 | 40 | 8 circuits × seeds 999–1003 | 40/40 complete legal layouts |
| Official baselines | 24 | 3 methods × 8 ISPD2005 × seed 1000 | 22/24 legal; 2 technical failures retained |

Formal PPO runs use a 448 × 448 action grid, 1,000 episodes, and seeds 999–1003. The grid ablation additionally evaluates 224 × 224. Official MaskPlace, WireMask-EA, and EfficientPlace rows are single-seed (`1000`) results and are not presented as five-seed statistics.

### ISPD2005 method comparison

![ISPD2005 normalized HPWL comparison](assets/results/ispd2005_hpwl_relative.svg)

Exact CompRes MacroHPWL values below are scaled by `1e5`. LinkPlace rows are five-seed mean ± sample standard deviation; official baselines are seed 1000 only.

| Circuit | LinkPlace-C | LinkPlace-M | Δ M vs C | MaskPlace | WireMask-EA | EfficientPlace |
|---|---:|---:|---:|---:|---:|---:|
| adaptec1 | 5.723 ± 0.021 | 5.770 ± 0.042 | +0.83% | 10.333 | 7.456 | 7.097 |
| adaptec2 | 32.025 ± 2.267 | 28.659 ± 0.316 | -10.51% | 153.845 | 92.863 | 43.918 |
| adaptec3 | 51.129 ± 0.713 | 52.614 ± 0.217 | +2.90% | 116.168 | 70.157 | 56.942 |
| adaptec4 | 43.738 ± 0.202 | 45.536 ± 0.351 | +4.11% | 92.545 | *technical failed* | 57.086 |
| bigblue1 | 2.126 ± 0.029 | 2.119 ± 0.023 | -0.33% | 4.056 | 2.659 | *technical failed* |
| bigblue2 | 8.039 ± 0.071 | 8.245 ± 0.138 | +2.56% | 17.438 | 11.271 | 10.993 |
| bigblue3 | 31.246 ± 0.642 | 32.369 ± 1.351 | +3.59% | 156.090 | 82.069 | 63.876 |
| bigblue4 | 53.347 ± 0.535 | 52.891 ± 0.828 | -0.86% | 114.111 | 82.304 | 88.510 |

### Grid-resolution ablation

![Grid ablation](assets/results/grid_ablation.svg)

<details>
<summary>Exact dual-grid HPWL and All-greedy success counts</summary>

| Circuit | LinkPlace-C 448 | LinkPlace-C 224 | LinkPlace-M 448 | LinkPlace-M 224 | All-greedy legal 448 | All-greedy legal 224 |
|---|---:|---:|---:|---:|---:|---:|
| adaptec1 | 5.723 ± 0.021 | 5.942 ± 0.035 | 5.770 ± 0.042 | 6.017 ± 0.017 | 1/5 | 1/5 |
| adaptec2 | 32.025 ± 2.267 | 35.951 ± 3.199 | 28.659 ± 0.316 | 31.445 ± 1.071 | 0/5 | 0/5 |
| adaptec3 | 51.129 ± 0.713 | 52.735 ± 0.292 | 52.614 ± 0.217 | 54.369 ± 0.739 | 0/5 | 0/5 |
| adaptec4 | 43.738 ± 0.202 | 46.702 ± 0.367 | 45.536 ± 0.351 | 48.224 ± 0.415 | 3/5 | 1/5 |
| bigblue1 | 2.126 ± 0.029 | 2.266 ± 0.024 | 2.119 ± 0.023 | 2.247 ± 0.017 | 5/5 | 4/5 |
| bigblue2 | 8.039 ± 0.071 | 8.071 ± 0.044 | 8.245 ± 0.138 | 8.301 ± 0.076 | 0/5 | 0/5 |
| bigblue3 | 31.246 ± 0.642 | 35.335 ± 1.927 | 32.369 ± 1.351 | 46.223 ± 15.127 | 1/5 | 3/5 |
| bigblue4 | 53.347 ± 0.535 | 59.812 ± 0.971 | 52.891 ± 0.828 | 59.621 ± 1.004 | 4/5 | 5/5 |

</details>

### Ariane and ICCAD2015-derived instances

The public LinkPlace-C artifact contains all five Ariane seeds: MacroHPWL `7.228 ± 0.110` × `1e5`, peak RUDY `98.8939 ± 12.3974`, mean runtime `3.89 h`, and best seed `1001`. The paper also reports the LinkPlace-M Ariane summary, but its seed-level CSV is not present in this public snapshot; the README therefore does not synthesize unversioned per-seed values.

For the eight ICCAD2015-derived macro-only instances, the table gives unrounded five-seed statistics scaled by `1e8`.

| Circuit | LinkPlace-C HPWL | LinkPlace-M HPWL | Δ M vs C | C runtime | M runtime |
|---|---:|---:|---:|---:|---:|
| superblue1 | 1.129 ± 0.139 | 1.037 ± 0.001 | -8.12% | 0.11 h | 2.16 h |
| superblue3 | 2.551 ± 0.008 | 2.613 ± 0.012 | +2.40% | 0.28 h | 2.16 h |
| superblue4 | 1.575 ± 0.001 | 1.551 ± 0.012 | -1.55% | 0.20 h | 2.17 h |
| superblue5 | 7.223 ± 0.024 | 7.258 ± 0.088 | +0.49% | 0.58 h | 2.18 h |
| superblue7 | 2.403 ± 0.016 | 2.427 ± 0.003 | +0.97% | 0.31 h | 2.17 h |
| superblue10 | 0.887 ± 0.001 | 0.887 ± 0.002 | +0.03% | 0.05 h | 2.16 h |
| superblue16 | 2.113 ± 0.136 | 2.046 ± 0.023 | -3.16% | 0.47 h | 2.19 h |
| superblue18 | 0.824 ± 0.008 | 0.840 ± 0.008 | +1.89% | 0.15 h | 2.15 h |

### Seed stability and RUDY trade-offs

![Five-seed stability heatmap](assets/results/seed_stability.svg)

The heatmap exposes every public seed instead of only mean ± standard deviation. The normalization is performed independently within each method/circuit pair, so it measures stochastic spread rather than cross-circuit quality.

![RUDY relative changes](assets/results/rudy_relative_delta.svg)

<details>
<summary>Exact matched RUDY means for LinkPlace-C and LinkPlace-M</summary>

| Circuit | C peak | M peak | C top-5% mean | M top-5% mean |
|---|---:|---:|---:|---:|
| adaptec1 | 0.666249 | 0.645991 | 0.0821768 | 0.0821225 |
| adaptec2 | 0.732489 | 0.877377 | 0.14394 | 0.141186 |
| adaptec3 | 0.578448 | 0.56465 | 0.0877631 | 0.0879957 |
| adaptec4 | 0.394727 | 0.385773 | 0.0759508 | 0.0792122 |
| bigblue1 | 0.137881 | 0.137724 | 0.0307165 | 0.0303788 |
| bigblue2 | 0.0719219 | 0.0750122 | 0.0214889 | 0.0216774 |
| bigblue3 | 0.818337 | 0.80503 | 0.0746362 | 0.076791 |
| bigblue4 | 0.457117 | 0.407608 | 0.0794019 | 0.0800462 |
| superblue1 | 0.000231756 | 0.00023442 | 8.84153e-06 | 8.23751e-06 |
| superblue3 | 0.000551671 | 0.000479434 | 1.55328e-05 | 1.58482e-05 |
| superblue4 | 0.000680785 | 0.000821537 | 2.90048e-05 | 2.85279e-05 |
| superblue5 | 0.000603314 | 0.000616577 | 4.24537e-05 | 4.25915e-05 |
| superblue7 | 0.000780209 | 0.000743269 | 3.51724e-05 | 3.57444e-05 |
| superblue10 | 0.000257661 | 0.000250823 | 7.57888e-06 | 7.58675e-06 |
| superblue16 | 0.000326553 | 0.000379617 | 3.70305e-05 | 3.59234e-05 |
| superblue18 | 0.000354221 | 0.00042407 | 2.1565e-05 | 2.19391e-05 |

</details>

### Machine-readable records

- [LinkPlace-C: 85 per-seed records](artifacts/tables/main_seed_results.csv) and [17-circuit summary](artifacts/tables/main_mean_std.csv)
- [Dual-grid: 240 per-seed records](artifacts/tables/grid_ablation_five_seed_results.csv) and [48 summary rows](artifacts/tables/grid_ablation_five_seed_mean_std.csv)
- [LinkPlace-M ICCAD2015: 40 per-seed records](artifacts/tables/linkplace_m_iccad2015_seed_results.csv) and [8-circuit summary](artifacts/tables/linkplace_m_iccad2015_mean_std.csv)
- [Official baselines: 24 seed records](artifacts/tables/baseline_seed_results.csv), including WireMask-EA/adaptec4 and EfficientPlace/bigblue1 technical failures

Regenerate and verify the README section and all SVGs with:

```bash
python tools/render_readme_results.py --write
python tools/render_readme_results.py --check
```
<!-- README_RESULTS:END -->

## Formal reproduction

The complete settings and result schema are frozen in
[`docs/FORMAL_PROTOCOL.md`](docs/FORMAL_PROTOCOL.md) and
[`configs/formal.json`](configs/formal.json). Formal runs use seeds
999-1003, 1000 episodes, and a 448 x 448 grid; the grid ablation additionally
uses 224 x 224.

Set environment variables only when paths differ from the defaults:

```bash
export LINKPLACE_PYTHON=/path/to/python
export LINKPLACE_RESULT_ROOT=/path/to/outputs/formal
```

The aggregation tools accept archived `ablation/monolithic` results as a
legacy alias but expose them as LinkPlace-M. New executions never write the old
method name.

## Paper artifacts

Only small, reviewable tables and manifests should be committed. Raw result
trees, checkpoints, TensorBoard files, datasets, and third-party build trees
are ignored. Large immutable artifacts should be published separately (for
example, an institutional repository or Zenodo) with SHA-256 manifests and a
versioned DOI.

## Release limitations

This public snapshot retains the following limitations:

1. Resolve the MaskPlace-derived code identified in
   `THIRD_PARTY_NOTICES.md`. The audited MaskPlace revision has no explicit
   license, so public redistribution requires written permission or independent
   replacement code before choosing a repository-wide license.
2. Add the final author list, paper title, and DOI to a valid `CITATION.cff`
   when the corresponding metadata is available.
3. Verify benchmark redistribution terms; the current release intentionally
   excludes benchmark data.

No license is implied until a `LICENSE` file is added by the authors.
