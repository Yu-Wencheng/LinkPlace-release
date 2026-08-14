from __future__ import annotations

import csv
import hashlib
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "mixed_size_best"
EXPECTED_CIRCUITS = {
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue3",
}


class PublishedMixedSizeResultTest(unittest.TestCase):
    def test_summary_and_archives_are_complete(self) -> None:
        with (ARTIFACTS / "summary.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual({row["circuit"] for row in rows}, EXPECTED_CIRCUITS)
        self.assertEqual(len(rows), len(EXPECTED_CIRCUITS))
        for row in rows:
            self.assertNotIn("macro_seed", row)
            self.assertEqual(row["legal"], "true")
            self.assertGreater(int(row["final_full_design_hpwl"]), 0)
            archive = ARTIFACTS / row["package"]
            self.assertTrue(archive.is_file(), archive)
            with zipfile.ZipFile(archive) as package:
                names = set(package.namelist())
                self.assertEqual(
                    names,
                    {f'{row["circuit"]}.gp.pl', "linkplace_macro_layout.json"},
                )
                placement_hash = hashlib.sha256(
                    package.read(f'{row["circuit"]}.gp.pl')
                ).hexdigest()
                macro_hash = hashlib.sha256(
                    package.read("linkplace_macro_layout.json")
                ).hexdigest()
            self.assertEqual(placement_hash, row["placement_sha256"])
            self.assertEqual(macro_hash, row["macro_layout_sha256"])

    def test_archive_checksums(self) -> None:
        recorded = {}
        for line in (ARTIFACTS / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            recorded[name] = digest

        self.assertEqual(set(recorded), {f"{name}.zip" for name in EXPECTED_CIRCUITS})
        for name, expected in recorded.items():
            actual = hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)


if __name__ == "__main__":
    unittest.main()
