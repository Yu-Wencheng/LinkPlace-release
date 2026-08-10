from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.run_dreamplace_mixed import evaluate_full_design


class DreamplaceEvaluationTests(unittest.TestCase):
    def test_full_design_hpwl_and_rudy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = root / "tiny.nodes"
            nets = root / "tiny.nets"
            pl = root / "tiny.pl"
            scl = root / "tiny.scl"
            nodes.write_text(
                "UCLA nodes 1.0\nNumNodes : 2\nNumTerminals : 0\n"
                "a 10 10\nb 10 10\n",
                encoding="utf-8",
            )
            nets.write_text(
                "UCLA nets 1.0\nNumNets : 1\nNumPins : 2\n"
                "NetDegree : 2 n0\n"
                "a I : 0 0\nb I : 0 0\n",
                encoding="utf-8",
            )
            pl.write_text(
                "UCLA pl 1.0\na 0 0 : N\nb 20 0 : N\n",
                encoding="utf-8",
            )
            scl.write_text(
                "UCLA scl 1.0\nNumRows : 1\nCoreRow Horizontal\n"
                " Coordinate : 0\n Height : 10\n Sitewidth : 1\n"
                " SubrowOrigin : 0 NumSites : 100\nEnd\n",
                encoding="utf-8",
            )
            metrics, demand = evaluate_full_design(nodes, nets, scl, pl, grid=10)
            self.assertEqual(metrics["hpwl"], 20.0)
            self.assertEqual(metrics["evaluated_nets"], 1)
            self.assertEqual(demand.shape, (10, 10))
            self.assertGreater(metrics["rudy_peak"], 0.0)


if __name__ == "__main__":
    unittest.main()
