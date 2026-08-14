from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "tables"


class ReleaseMetadataTest(unittest.TestCase):
    def test_citation_has_current_public_identity(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("Connectivity-Aware Reinforcement Learning for Macro Placement", citation)
        self.assertIn("https://github.com/Yu-Wencheng/LinkPlace-release", citation)
        self.assertIn("family-names: Yu", citation)
        self.assertIn("version: 0.1.0", citation)
        self.assertNotIn("license:", citation)

    def test_zenodo_metadata_is_a_nonactivating_draft(self) -> None:
        draft = json.loads(
            (ROOT / "docs" / "zenodo-metadata-draft.json").read_text(encoding="utf-8")
        )
        self.assertEqual(draft["upload_type"], "software")
        self.assertEqual(draft["version"], "0.1.0")
        self.assertEqual(len(draft["creators"]), 5)
        self.assertFalse((ROOT / ".zenodo.json").exists())

    def test_ariane_linkplace_m_seed_statistics(self) -> None:
        with (TABLES / "linkplace_m_ariane_seed_results.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        with (TABLES / "linkplace_m_ariane_mean_std.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            summary = next(csv.DictReader(stream))

        self.assertEqual({int(row["seed"]) for row in rows}, {999, 1000, 1001, 1002, 1003})
        self.assertTrue(all(row["status"] == "complete" for row in rows))
        self.assertTrue(all(row["legal"] == "True" for row in rows))
        values = [float(row["comp_res_hpwl"]) for row in rows]
        self.assertAlmostEqual(statistics.mean(values), float(summary["comp_res_hpwl_mean"]))
        self.assertAlmostEqual(statistics.stdev(values), float(summary["comp_res_hpwl_std"]))
        public_csv = (TABLES / "linkplace_m_ariane_seed_results.csv").read_text()
        self.assertNotIn("result_path", public_csv)
        self.assertNotIn("source_path", public_csv)

    def test_environment_capture_tool_has_stable_cli(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/export_environment.py", "--help"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertIn("--output", completed.stdout)


if __name__ == "__main__":
    unittest.main()
