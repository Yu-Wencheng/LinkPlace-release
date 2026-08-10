# Public result artifacts

This directory contains only small paper-facing CSV/JSON files. Machine-local
result paths are removed and the public method names LinkPlace-C and LinkPlace-M
replace their historical internal labels.

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
