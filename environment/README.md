# Environment reconstruction

`pyproject.toml` describes the portable Python-level dependencies. Formal GPU
reproduction additionally requires a CUDA-enabled PyTorch build compatible
with the host NVIDIA driver.

For an exact source-machine snapshot, run the provided capture script with the
same Python interpreter and activated environment used for the formal runs:

```bash
python tools/export_environment.py --output environment
```

The script records Python packages, Conda metadata when available, the NVIDIA
driver/GPU snapshot, PyTorch's environment report, platform metadata, and a
SHA-256 manifest. It removes the Conda `prefix:` field and does not record the
value of `CUDA_VISIBLE_DEVICES`.

Do not commit a packed Conda environment: it is large, platform-specific, and
may contain absolute prefixes. Publish any binary environment archive as a
separate checksummed release asset. The text specifications above are the
auditable source of truth.
