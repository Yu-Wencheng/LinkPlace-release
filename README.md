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

## Publication blockers

Before making this repository public:

1. Resolve the MaskPlace-derived code identified in
   `THIRD_PARTY_NOTICES.md`. The audited MaskPlace revision has no explicit
   license, so public redistribution requires written permission or independent
   replacement code before choosing a repository-wide license.
2. Add the final author list, repository URL, paper title, and DOI to a valid
   `CITATION.cff`.
3. Verify benchmark redistribution terms; the current release intentionally
   excludes benchmark data.

No license is implied until a `LICENSE` file is added by the authors.
