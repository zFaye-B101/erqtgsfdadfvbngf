"""Exercise the real CLI in subprocesses using synthetic data only."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "02_SOP_Experiments.py"


class CLITests(unittest.TestCase):
    def run_cli(self, *arguments):
        with tempfile.TemporaryDirectory() as cache:
            environment = dict(os.environ, MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE="1")
            return subprocess.run(
                [sys.executable, str(ENTRYPOINT), *map(str, arguments)],
                cwd=ROOT, env=environment, capture_output=True, text=True,
                timeout=180,
            )

    def test_help_runs_argument_parser(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertIn("--method {baselines,supcon,all}", result.stdout)

    def test_unknown_group_is_rejected(self):
        result = self.run_cli("--group", "UNKNOWN", "--method", "baselines")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Unknown or ambiguous group", result.stderr)

    def test_both_methods_produce_smoke_results(self):
        import numpy as np
        import pandas as pd
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            rows = []
            # Enough products per category for two outer/inner folds and calibration.
            for category in range(3):
                for product in range(40):
                    sku = category * 40 + product
                    pixels = np.random.default_rng(sku).integers(20, 230, (224, 224, 3))
                    for view in range(4):
                        image_id = len(rows) + 1
                        relative = f"{image_id}.jpg"
                        Image.fromarray((pixels + view).astype(np.uint8)).save(data / relative)
                        rows.append(dict(
                            image_id=image_id, class_id=sku, super_class_id=category,
                            source_split="train", original_width=224, original_height=224,
                            processed_relative_path=relative,
                        ))
            with (data / "clean_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (data / "audit_summary.json").write_text(
                json.dumps({"clean_images": len(rows), "complete": True}), encoding="utf-8",
            )

            config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            config.update(run_name="cli_test", experiment_dir=str(ROOT / "experiment_configs"), run_official_test=False)
            for key, name in (("data_dir", "data"), ("protocol_dir", "protocol"),
                              ("cache_dir", "cache"), ("results_dir", "results")):
                config[key] = str(root / name)
            config["supcon"].update(hidden_dim=16, representation_dim=8, projection_dim=8, device="cpu")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            for method in ("baselines", "supcon"):
                with self.subTest(method=method):
                    result = self.run_cli(
                        "--config", config_path, "--group", "F1", "--method", method,
                        "--smoke-test", "--skip-official-test",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("/ smoke", result.stdout)
                    output = root / "results_smoke" / "cli_test" / "F1_Traditional_HSV48" / method
                    summary = pd.read_csv(output / "summary_mean_sd.csv")
                    self.assertTrue(summary["mode"].eq("smoke").all())
                    self.assertTrue(summary.outer_fold_count.eq(2).all())
                    self.assertEqual(set(summary.condition), set(config["conditions"]))
                    self.assertEqual(set(summary.algorithm), {"LR", "SVM"} if method == "baselines" else {"SupCon"})
            combined = root / "results_smoke" / "cli_test" / "F1_Traditional_HSV48" / "combined_summary_mean_sd.csv"
            self.assertEqual(set(pd.read_csv(combined).algorithm), {"LR", "SVM", "SupCon"})


if __name__ == "__main__":
    unittest.main()
