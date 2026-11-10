import json
import tempfile
import unittest
from pathlib import Path
from construction_quality_gate import audit_source


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.good = dict(sample_id="a", status="ok", correct=True, accepted=True)
        self.write("construct", [self.good, dict(sample_id="b", status="rejected")])
        self.write("transfer", [self.good]); self.write("filter", [self.good])
        self.write("cot_filtered", [dict(sample_id="a", rationale="observed")])

    def write(self, name, rows):
        (self.root / (name + ".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in rows))

    def test_complete_and_latest_retry(self):
        self.write("construct", [dict(self.good, status="error"), self.good, dict(sample_id="b", status="rejected")])
        self.assertEqual(audit_source(self.root, 2, 1)["filter"]["accepted"], 1)

    def test_missing_transfer_record(self):
        self.write("transfer", [])
        with self.assertRaisesRegex(ValueError, "coverage"):
            audit_source(self.root, 2, 1)

    def test_unresolved_error(self):
        self.write("construct", [self.good, dict(sample_id="b", status="error")])
        with self.assertRaisesRegex(ValueError, "operational"):
            audit_source(self.root, 2, 1)

    def test_duplicate_export(self):
        self.write("cot_filtered", [self.good, self.good])
        with self.assertRaisesRegex(ValueError, "export"):
            audit_source(self.root, 2, 1)

    def test_yield_regression(self):
        with self.assertRaisesRegex(ValueError, "yield"):
            audit_source(self.root, 2, 2)
