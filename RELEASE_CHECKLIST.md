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

## Required before making the repository public

- [ ] Obtain written redistribution permission from the MaskPlace copyright
  holders, or replace the identified MaskPlace-derived files with independently
  implemented and verified code.
- [ ] After resolving that dependency, choose and add a compatible `LICENSE`.
- [ ] Add valid author, paper, repository, and DOI fields to `CITATION.cff`.
- [ ] Add exact server environment exports under `environment/`.
- [ ] Add exact Ariane LinkPlace-M per-seed public artifacts; the local paper
  snapshot currently contains only the rounded paper value.
- [ ] Run the unit suite in a fresh Python 3.10 environment with dependencies.
- [ ] Run one documented CUDA smoke test with a legally obtained benchmark.
- [ ] Publish the full immutable artifact archive separately and record its DOI
  and SHA-256 manifest in the README.
- [ ] Initialize Git only after the above review, then inspect the first commit
  for large files and sensitive data before pushing.
