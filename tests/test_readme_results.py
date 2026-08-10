from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReadmeResultArtifactTest(unittest.TestCase):
    def test_generated_readme_results_are_current(self) -> None:
        subprocess.run(
            [sys.executable, "tools/render_readme_results.py", "--check"],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
