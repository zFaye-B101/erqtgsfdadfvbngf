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
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        config.update(data_dir=str(data_dir), protocol_dir=str(root / "protocol"),
                      cache_dir=str(root / "cache"), results_dir=str(root / "results"),
                      experiment_dir=str(ROOT / "experiment_configs"))
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

    def test_unknown_group_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, _ = self.run_check(root, root / "absent", "--group", "UNKNOWN")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Unknown or ambiguous group", result.stdout)

    def test_groups_reported_and_path_estimate_uses_names(self):
        import importlib.util
        sys.path.insert(0, str(ROOT))
        from utils.sop_config import discover_experiments, load_config
        spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
        preflight = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preflight)
        found = discover_experiments(ROOT / "experiment_configs")
        self.assertEqual(preflight.select_groups(list(found), "F1.2"), ["F1.2_Traditional_Content_HSV_Gradient192"])
        content = load_config(ROOT / "config.json", experiment_path=found["F1.2_Traditional_Content_HSV_Gradient192"][0])
        self.assertEqual(preflight.extractors_of(content), {"hsv_hog_content"})
        name = preflight.select_groups(list(found), "F4")[0]
        config = load_config(ROOT / "config.json", experiment_path=found[name][0])
        expected = (Path(config["results_dir"]) / config["run_name"] / name / "supcon" / "outer_10"
                    / "SupCon_ResNet18_GlobalLocal2560" / "tau_0.0125" / "inner_9" / "training_loss.json")
        self.assertGreaterEqual(preflight.estimate_longest_output_path({name: config}), len(str(expected)))


if __name__ == "__main__":
    unittest.main()
