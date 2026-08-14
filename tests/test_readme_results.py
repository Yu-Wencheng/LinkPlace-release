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

    def test_paper_results_precede_server_only_details(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        paper = readme.index("## Results reported in the paper")
        extra = readme.index("## Additional server results not shown in the paper")
        self.assertLess(paper, extra)
        self.assertIn("LinkPlace-C", readme[paper:extra])
        self.assertIn("LinkPlace-M", readme[paper:extra])
        self.assertIn("placement grid", readme[paper:extra])
        self.assertIn("evaluation grid", readme[paper:extra])

    def test_readme_paper_figures_are_versioned(self) -> None:
        for name in (
            "ispd2005_convergence.png",
            "linkplace_component_layouts.png",
            "dreamplace_final_layouts.png",
        ):
            path = ROOT / "assets" / "paper" / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 20_000, name)

    def test_mixed_size_publication_omits_seed_identifiers(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        start = readme.index("### Best legal mixed-size placements")
        end = readme.index("## Additional server results not shown in the paper")
        self.assertNotIn("seed", readme[start:end].lower())


if __name__ == "__main__":
    unittest.main()
