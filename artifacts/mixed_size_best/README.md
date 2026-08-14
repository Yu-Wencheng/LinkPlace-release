# Best legal mixed-size placements

This directory contains the best legal full-design placement obtained for each
completed ISPD2005 mixed-size case.  The selection criterion is the final
physical-coordinate MacroHPWL after DREAMPlace 4.1.0 standard-cell placement.

Each circuit archive contains:

- `<circuit>.gp.pl`: the final exported full-design placement;
- `linkplace_macro_layout.json`: the fixed LinkPlace macro initialization used
  by the mixed-size flow.

The public summary in [`summary.csv`](summary.csv) records the LinkPlace
variant, final MacroHPWL, legality status, and source-artifact hashes.
[`SHA256SUMS`](SHA256SUMS) provides hashes for the downloadable archives.
ISPD2005 benchmark inputs are not redistributed; obtain them from the official
benchmark source when loading these placements.
