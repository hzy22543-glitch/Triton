"""Ensure corrupted evidence cannot silently produce a successful report."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_results import audit_run


class EvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'evidence'
        shutil.copytree(ROOT / 'results/runs/20261114_235210_573287', self.path)

    def test_real_record_passes(self):
        _, rows = audit_run(self.path)
        self.assertEqual(len(rows), 4)

    def test_changed_raw_timing_is_rejected(self):
        p = self.path / 'raw.json'
        rows = json.loads(p.read_text())
        rows[0]['raw_block_seconds'] = [v*2 for v in rows[0]['raw_block_seconds']]
        p.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'statistic mismatch'):
            audit_run(self.path)

    def test_missing_timing_group_is_rejected(self):
        p = self.path / 'raw.json'
        p.write_text(json.dumps(json.loads(p.read_text())[1:]))
        with self.assertRaisesRegex(ValueError, 'timing groups'):
            audit_run(self.path)

    def test_modified_source_is_rejected(self):
        p = self.path / 'source/w10_softmax.py'
        p.write_bytes(p.read_bytes() + b'\n# modified\n')
        with self.assertRaisesRegex(ValueError, 'source hash'):
            audit_run(self.path)

    def test_failed_correctness_is_rejected(self):
        p = self.path / 'correctness.json'
        rows = json.loads(p.read_text())
        rows[0]['status'] = 'FAIL'
        p.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'correctness check'):
            audit_run(self.path)

    def test_unknown_suite_is_rejected(self):
        p = self.path / 'environment.json'
        env = json.loads(p.read_text())
        env['args']['suite'] = 'typo'
        p.write_text(json.dumps(env))
        with self.assertRaisesRegex(ValueError, 'unknown suite'):
            audit_run(self.path)

    def test_unknown_dtype_is_rejected(self):
        p = self.path / 'environment.json'
        env = json.loads(p.read_text())
        env['args']['dtype'] = 'fp64'
        p.write_text(json.dumps(env))
        with self.assertRaisesRegex(ValueError, 'unknown dtype'):
            audit_run(self.path)


if __name__ == '__main__':
    unittest.main()
