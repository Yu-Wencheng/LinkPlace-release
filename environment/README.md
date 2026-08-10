# Environment reconstruction

`pyproject.toml` describes the portable Python-level dependencies. Formal GPU
reproduction additionally requires a CUDA-enabled PyTorch build compatible
with the host NVIDIA driver.

For an exact source machine snapshot, place the following generated files in
this directory before tagging a release:

```bash
python -m pip freeze --all > environment/pip-freeze.txt
conda list --explicit > environment/conda-explicit.txt
conda env export --no-builds > environment/conda-environment.yml
nvidia-smi > environment/nvidia-smi.txt
python -m torch.utils.collect_env > environment/torch-collect-env.txt
```

Do not commit a packed Conda environment: it is large, platform-specific, and
may contain absolute prefixes. Publish any binary environment archive as a
separate checksummed release asset. The text specifications above are the
auditable source of truth.
