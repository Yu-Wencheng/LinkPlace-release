# Dataset preparation

LinkPlace consumes compressed, self-contained JSON caches in `datasets/cache`.
The public repository does not redistribute ISPD2005, ICCAD2015, Ariane, or
DREAMPlace benchmark files.

The historical cache importer can read an external checkout that provides the
`codeplace.bookshelf`, `codeplace.iccad2015`, and `codeplace.ariane` parsers:

```bash
python tools/import_paper_datasets.py \
  --source-root /path/to/external-baseline-checkout \
  --output-root datasets/cache
```

Each generated cache is accompanied by a manifest containing source hashes.
Review those manifests before moving caches between machines. Do not commit the
cache files unless the benchmark licenses explicitly permit redistribution.

Expected cache keys include:

- `adaptec1`-`adaptec4`, `bigblue1`, `bigblue3`;
- `bigblue2-1024`, `bigblue4-1024`;
- `superblue1-512`, `superblue3-512`, `superblue4-512`, `superblue5-512`,
  `superblue7-512`, `superblue10-512`, `superblue16-512`, `superblue18-512`;
- `ariane`.
