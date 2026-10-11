"""Run the read-only preflight script in subprocesses using temporary folders."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "00_Check_Environment.py"


class PreflightTests(unittest.TestCase):
    def run_check(self, root, data_dir, *arguments):
        config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
        config.update(data_dir=str(data_dir), protocol_dir=str(root / "protocol"),
                      cache_dir=str(root / "cache"), results_dir=str(root / "results"))
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config_path), *map(str, arguments)],
                                cwd=root, env=environment, capture_output=True, text=True, timeout=180)
        return result, sorted(path.name for path in root.iterdir())

    def test_completed_preprocessing_passes_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            (data / "images").mkdir(parents=True)
            (data / "clean_manifest.csv").write_text("image_id\n1\n", encoding="utf-8")
            (data / "audit_summary.json").write_text(
                json.dumps({"complete": True, "clean_images": 1, "clean_products": 1}), encoding="utf-8")
            result, files = self.run_check(root, data)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[PASS] Preprocessed data", result.stdout)
            self.assertIn("0 FAIL", result.stdout)
            # Only the config written by this test remains; no outputs or write-test files.
            self.assertEqual(files, ["config.json", "data"])

    def test_missing_zip_and_incomplete_data_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, _ = self.run_check(root, root / "absent", "--zip-path", root / "missing.zip")
            self.assertEqual(result.returncode, 1)
            self.assertIn("[FAIL] SOP ZIP", result.stdout)
            partial = root / "partial"
            partial.mkdir()
            (partial / "leftover.txt").write_text("x", encoding="utf-8")
            result, _ = self.run_check(root, partial)
            self.assertEqual(result.returncode, 1)
            self.assertIn("nonempty but has no audit_summary.json", result.stdout)


if __name__ == "__main__":
    unittest.main()
