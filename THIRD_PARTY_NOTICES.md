# Third-party and data notices

This staging tree contains or interfaces with code whose provenance must be
reviewed before assigning a repository-wide license.

## MaskPlace-derived code

The PPO backbone and related evaluator/environment code were developed with
reference to MaskPlace:

- Upstream repository: <https://github.com/laiyao1/maskplace>
- Audited revision: `9c7b72fa825eb989a974931fed9114674954c50c`
- Paper: *MaskPlace: Fast Chip Placement via Reinforced Visual Representation
  Learning* (NeurIPS 2022).

The audited revision has no `LICENSE` file and GitHub reports no detected
license. The following release files require particular attention:

- `linkplace/evaluation/prim.py` is byte-identical to upstream
  `maskplace/prim.py` at the audited revision.
- `linkplace/evaluation/comp_res.py` is an extended derivative of upstream
  `maskplace/comp_res.py`.
- `linkplace/models/networks.py`, `linkplace/models/ppo.py`, and
  `linkplace/legacy/fast_env.py` preserve substantial PPO/environment structure
  derived from or developed against MaskPlace.

Because the upstream repository provides no explicit redistribution license,
these files must not be published under a guessed open-source license. Public
release requires either written permission from the MaskPlace copyright
holders or independently implemented replacements whose provenance is
documented and whose behavior is verified against the frozen protocol.

## Other external components

- CompRes HPWL evaluation semantics are retained to match the paper evaluator.
- Optional DREAMPlace 4.1.0 integration. DREAMPlace itself is not vendored.
- Optional external parsers used only to build self-contained benchmark caches.

ISPD2005, ICCAD2015, Ariane, and DREAMPlace datasets are not included. Users
must obtain them from their authorized sources and comply with their licenses.

Before public release, record the upstream repository URL, revision, license or
written permission, and local modifications for every inherited component.
This notice does not grant rights beyond those provided by the respective
copyright holders.
