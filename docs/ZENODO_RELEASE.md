# Zenodo release preparation

Zenodo can archive a GitHub release after the repository is enabled in the
owner's Zenodo GitHub settings. `CITATION.cff` provides the current software
citation metadata. `docs/zenodo-metadata-draft.json` records the corresponding
Zenodo fields but is intentionally not named `.zenodo.json` yet.

The final archive is blocked by the unresolved redistribution terms described
in `docs/MASKPLACE_PROVENANCE.md`. Zenodo requires one license to apply to all
files in an open deposit; publishing before that issue is resolved could attach
an inaccurate license to inherited files.

After the rights issue is resolved:

1. add the selected repository-wide license;
2. add its Zenodo license identifier to the metadata draft and copy the file to
   `.zenodo.json`;
3. add the final paper DOI to both citation metadata files;
4. enable `Yu-Wencheng/LinkPlace-release` in Zenodo's GitHub integration;
5. create the final GitHub release tag; and
6. record the resulting version DOI and concept DOI in the README.

The GitHub release must be created only after the repository is enabled in
Zenodo so that the release is archived automatically.
