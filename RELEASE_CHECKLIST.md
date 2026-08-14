# Public release checklist

## Completed in this staging tree

- [x] Repository directory is named `LinkPlace-release`.
- [x] Public method names are LinkPlace, LinkPlace-C, and LinkPlace-M.
- [x] Core code uses the `linkplace` Python package.
- [x] New results use `linkplace-c` and `linkplace-m` identifiers.
- [x] Aggregators retain read-only compatibility with archived `monolithic`
  result directories.
- [x] Private server paths and credentials are absent.
- [x] Raw datasets, checkpoints, TensorBoard events, third-party builds, and
  full result trees are excluded by `.gitignore`.
- [x] Small paper tables have machine-local path columns removed.
- [x] `SOURCE_MANIFEST.sha256` records the lightweight release contents.
- [x] Python source compiles and JSON/TOML metadata parses.
- [x] Author and repository metadata are recorded in `CITATION.cff`.
- [x] Exact Ariane LinkPlace-M five-seed records are published as path-free CSV.
- [x] A path-sanitized environment capture tool is provided.
- [x] Git is initialized and the public remote is configured.

## Required before a licensed archival release

- [ ] Obtain written redistribution permission from the MaskPlace copyright
  holders, or replace the identified MaskPlace-derived files with independently
  implemented and verified code.
- [ ] After resolving that dependency, choose and add a compatible `LICENSE`.
- [ ] Add the final paper DOI to `CITATION.cff` when assigned.
- [ ] Run `tools/export_environment.py` in the formal server environment and
  commit the generated text snapshot under `environment/`.
- [ ] Run the unit suite in a fresh Python 3.10 environment with dependencies.
- [ ] Run one documented CUDA smoke test with a legally obtained benchmark.
- [ ] After the redistribution terms are resolved, publish the immutable
  artifact archive and record its DOI and SHA-256 manifest in the README.
- [ ] Resolve the code/paper differences documented in `RELEASE_AUDIT.md`
  before declaring the source a faithful final-method release.
