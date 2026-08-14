# Release consistency audit

Audit date: 2026-08-15

The public source currently preserves the implementation used to create the
archived experiment records. Two source/paper differences must be resolved
before a final reproducibility tag is described as implementing the manuscript
verbatim:

1. `linkplace.graph.ordered_small_components` orders small components by total
   area, macro count, and identifier, while the LinkPlace-M path orders every
   component by macro count, area, and identifier. Consequently, LinkPlace-C
   and LinkPlace-M can traverse small components in different orders even
   though they share the same component-internal connectivity rule.
2. `linkplace.runner.train_component` derives a relative placement from the
   minimum-HPWL placement epoch and then applies a legal rigid translation.
   The LinkPlace-M path calls this routine on an empty initial canvas, whereas
   the current manuscript states that LinkPlace-M applies no whitespace-aware
   translation.

Changing either execution path could change the reported placements and would
require a new formal experiment set. The release therefore records these
differences instead of silently modifying the implementation or the archived
results.

The separate MaskPlace provenance and redistribution issue is documented in
`docs/MASKPLACE_PROVENANCE.md`.
