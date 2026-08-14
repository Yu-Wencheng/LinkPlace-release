# Public result artifacts

This directory contains paper-facing CSV/JSON summaries and compressed final
placement artifacts. Machine-local result paths are removed and the public
method names LinkPlace-C and LinkPlace-M replace their historical internal
labels.

The best legal mixed-size placements for the completed ISPD2005 cases are
available under [`mixed_size_best/`](mixed_size_best/). Each package contains
the final exported placement and its LinkPlace macro initialization; benchmark
inputs remain external.

The tables are evidence summaries, not substitutes for the full raw archive.
The eventual archival deposit should include per-run `result.json`, training
CSV, checkpoints, layouts, and a SHA-256 manifest under a versioned DOI.

Regenerate the sanitized tables from a local artifact snapshot with:

```bash
python tools/prepare_public_artifacts.py \
  --source /path/to/paper_outputs/tables \
  --output artifacts/tables
```

The exact five Ariane LinkPlace-M seed rows were not present in the local table
snapshot used to stage this directory. They must be exported from the retained
raw archive before publication. The rounded value printed in the paper is not
silently substituted for per-seed evidence.
