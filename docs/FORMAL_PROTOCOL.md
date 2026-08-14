# Formal experiment protocol

This document freezes the protocol used for the paper-facing LinkPlace runs.
`LinkPlace-C` is the connectivity-decomposed method and `LinkPlace-M` is its
single-policy (monolithic) counterpart. The command-line identifiers are
`linkplace-c` and `linkplace-m`.

## Shared PPO settings

- Seeds: `999`, `1000`, `1001`, `1002`, `1003`.
- Formal action grid: `448 x 448`; grid ablations additionally use `224 x 224`.
- Training budget: 1000 placement epochs per trained policy. Each placement
  epoch generates one complete placement trajectory.
- Actor learning rate: `1e-3`; critic learning rate: `1e-4`.
- Discount: `0.95`; PPO clip: `0.2`; minibatch: `64`.
- Ten PPO update epochs and one update per five placement epochs.
- Deterministic PyTorch/CUDA settings and fixed per-run seed.

The square canvas extent, one physical/grid ratio, empty-canvas mask, wire
mask, reward, Actor, critic, and PPO update equations are unchanged between
LinkPlace-C and LinkPlace-M. The 448-grid implementation preserves the same
placement semantics as the reference implementation; its transition buffer is
preallocated to avoid duplicate full-resolution state tensors.

## LinkPlace-C

Every connected component has one deterministic macro order. The first macro
has maximum area. Each later macro must share a net with the already placed set
and maximizes the number of distinct shared nets, with area and macro name as
deterministic tie-breakers.

Large components (at least 20 macros) run first, ordered by macro count, total
macro area, and minimum macro name. Each component receives its own PPO policy
and is trained on the already occupied canvas. The legal placement epoch with
minimum macro-only HPWL is retained and rigidly translated to maximize the remaining blank
rectangle; seeded randomness only resolves exact ties.

Small components are then placed greedily by total area, macro count, and
minimum macro name. The fallback places individual macros using incremental
HPWL, remaining blank rectangle, and seeded tie-breaking. If a macro still has
no legal position, that attempt is retained as a real failure.

## LinkPlace-M and all-greedy controls

LinkPlace-M concatenates the same deterministic component orders and trains one
PPO policy for all selected macros. All-greedy preserves the same component
ordering but replaces trained placement with the greedy placer. All-greedy
failures are terminal observations and are included in success-count reporting.

## Benchmark scope

- ISPD2005: adaptec1-4 and bigblue1-4.
- ICCAD2015-derived: superblue1, 3, 4, 5, 7, 10, 16, and 18.
- Ariane: included for the LinkPlace-C main suite and the five-seed LinkPlace-M
  comparison.

Benchmark files are not redistributed by this repository. See
`docs/DATASETS.md` for expected inputs and cache generation.

## Result layout

The default root is `outputs/formal`; set `LINKPLACE_RESULT_ROOT` to override it.

- `main/<benchmark>/seed-<seed>/`: LinkPlace-C run and all retained attempts.
- `ablation/linkplace-m/<benchmark>/seed-<seed>/`: LinkPlace-M runs.
- `ablation/all-greedy/<benchmark>/seed-<seed>/`: all-greedy controls.
- `grid-ablation/grid-224/`: corresponding 224-grid runs.
- `dreamplace/`: optional fixed-macro DREAMPlace post-processing.
- `paper_outputs/tables/`: per-seed and mean/std CSV summaries.

Readers may also point the aggregation scripts at archived results that use the
legacy directory name `ablation/monolithic`; this alias is read-only and new
runs always use `ablation/linkplace-m`.

## Example commands

```bash
python -m linkplace.runner --cache-root datasets/cache inspect adaptec1

python -m linkplace.runner --cache-root datasets/cache \
  --result-root outputs/formal run-main adaptec1 \
  --seed 999 --episodes 1000 --grid 448 --device cuda

python -m linkplace.runner --cache-root datasets/cache \
  --result-root outputs/formal run-ablation linkplace-m adaptec1 \
  --seed 999 --episodes 1000 --grid 448 --device cuda

python tools/aggregate_paper_results.py --result-root outputs/formal
python tools/aggregate_multiseed_extension.py --result-root outputs/formal
```

The command-line option remains named `--episodes` for compatibility with the
archived experiment scripts; its value is the number of placement epochs, not
the number of PPO update epochs.

Paper-level conclusions require complete expected seed coverage. A healthy
process, a partial queue, or one successful seed is not a complete result.
