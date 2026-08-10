# MaskPlace provenance audit

Audit date: 2026-08-10

Upstream repository: <https://github.com/laiyao1/maskplace>

Pinned revision used by the experiment infrastructure:
`9c7b72fa825eb989a974931fed9114674954c50c`.

## License finding

The pinned revision contains no `LICENSE`, `COPYING`, or equivalent file, and
the GitHub repository metadata exposes no detected license. This is a release
blocker, not permission to select an arbitrary license.

## File evidence

| File | SHA-256 |
|---|---|
| MaskPlace `maskplace/prim.py` | `eb1ad1fd544b4b1fd2f5def92b9564295a34dc15dadd838bf8f709cee7c8dd5a` |
| LinkPlace `linkplace/evaluation/prim.py` | `eb1ad1fd544b4b1fd2f5def92b9564295a34dc15dadd838bf8f709cee7c8dd5a` |
| MaskPlace `maskplace/comp_res.py` | `37caa1136cc8366effd6ef118fee06a95f92a5182e0ea5d4b20d0e9bb290973b` |
| LinkPlace `linkplace/evaluation/comp_res.py` | `c5055241fa9ac997c36870d0fb48243e0944f183dbabb8392f34aa1269445819` |
| MaskPlace `maskplace/PPO2.py` | `4a12ba5fe2cefd0c1fe86bec4693c7c33873e8188a9db01e869687687f97a33b` |
| LinkPlace `linkplace/models/networks.py` | `12ff93e87bc022a0970d7061417bce218f70ac34b37fc30e85433bb79a311084` |
| MaskPlace `maskplace/place_env/place_env.py` | `2a07dcc822fdcd2a29a93c44f1f97d07c25ba0ae8626068438e59da5548a3161` |
| LinkPlace `linkplace/legacy/fast_env.py` | `64e541244ac1df818ecb0fde61bd32f659ec88750a786d34dccd1fb72f5bd170` |

`prim.py` is byte-identical. A line-based comparison shows that LinkPlace's
`comp_res.py` extends the upstream implementation, while the network and legacy
environment files retain substantial corresponding structure despite later
modifications.

## Acceptable release paths

1. Obtain written permission that explicitly allows redistribution and state
   the granted terms in this repository; or
2. replace the affected files with independently implemented code, retain the
   MaskPlace academic citation, document the new provenance, and rerun the
   equivalence/regression checks before release.

This file records technical provenance and is not legal advice.
