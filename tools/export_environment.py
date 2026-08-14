#!/usr/bin/env python3
"""Capture a reviewable, path-sanitized environment snapshot.

Run this script with the same Python interpreter used for the formal runs.
It writes text metadata only; it never packs the environment itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout.rstrip() + "\n"


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def optional_command(output: Path, command: list[str]) -> None:
    if shutil.which(command[0]) is None:
        write(output, f"UNAVAILABLE: {command[0]} was not found on PATH.\n")
        return
    code, text = run(command)
    if code != 0:
        write(output, f"COMMAND FAILED ({code}): {' '.join(command)}\n{text}")
        return
    write(output, text)


def sanitize_conda_yaml(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.startswith("prefix:")
    ).rstrip() + "\n"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("environment"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    system = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in os.environ,
    }
    write(output / "system.json", json.dumps(system, indent=2, sort_keys=True) + "\n")

    code, pip_freeze = run([sys.executable, "-m", "pip", "freeze", "--all"])
    if code != 0:
        raise SystemExit(f"pip freeze failed with exit code {code}:\n{pip_freeze}")
    write(output / "pip-freeze.txt", pip_freeze)

    code, torch_env = run([sys.executable, "-m", "torch.utils.collect_env"])
    if code == 0:
        write(output / "torch-collect-env.txt", torch_env)
    else:
        write(
            output / "torch-collect-env.txt",
            f"COMMAND FAILED ({code}): python -m torch.utils.collect_env\n{torch_env}",
        )

    optional_command(output / "nvidia-smi.txt", ["nvidia-smi"])
    optional_command(output / "conda-explicit.txt", ["conda", "list", "--explicit"])
    if shutil.which("conda") is None:
        write(output / "conda-environment.yml", "UNAVAILABLE: conda was not found on PATH.\n")
    else:
        code, conda_yaml = run(["conda", "env", "export", "--no-builds"])
        if code == 0:
            write(output / "conda-environment.yml", sanitize_conda_yaml(conda_yaml))
        else:
            write(
                output / "conda-environment.yml",
                f"COMMAND FAILED ({code}): conda env export --no-builds\n{conda_yaml}",
            )

    names = (
        "system.json",
        "pip-freeze.txt",
        "torch-collect-env.txt",
        "nvidia-smi.txt",
        "conda-explicit.txt",
        "conda-environment.yml",
    )
    records = [f"{digest(output / name)}  {name}" for name in names]
    write(output / "environment-manifest.sha256", "\n".join(records) + "\n")
    print(f"captured {len(names)} environment files in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
