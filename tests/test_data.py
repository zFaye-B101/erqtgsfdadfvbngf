import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
from sop_project.data import SOPDataset, configured_path


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "Stanford_Online_Products"
        (self.root / "items").mkdir(parents=True)
        header = "image_id class_id super_class_id path\n"
        (self.root / "Ebay_train.txt").write_text(header + "1 1 1 items/a.jpg\n2 1 1 items/b.jpg\n", encoding="utf-8")
        (self.root / "Ebay_test.txt").write_text(header + "3 2 1 items/c.jpg\n", encoding="utf-8")
        for name, size in (("a", (100, 100)), ("b", (120, 100)), ("c", (99, 120))):
            Image.new("RGB", size).save(self.root / "items" / (name + ".jpg"))

    def archive(self):
        path = self.base / "sample.zip"
        with ZipFile(path, "w") as archive:
            for item in self.root.rglob("*"):
                if item.is_file():
                    archive.write(item, item.relative_to(self.base).as_posix())
        return path

    def test_zip_folder_and_parent_equivalent(self):
        reports = []
        for path in (self.root, self.base, self.archive()):
            with SOPDataset(path) as data:
                rows = data.load_index()
                report, eligible = data.audit(rows, limit=None)
                reports.append(report)
                self.assertEqual([r["image_id"] for r in eligible], [1, 2])
                self.assertEqual(data.read_image(rows[0]).size, (100, 100))
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0], reports[2])
        self.assertEqual(reports[0]["scope"], "full_dataset")

    def test_smoke_check_is_not_full_audit(self):
        with SOPDataset(self.root) as data:
            report, _ = data.audit(data.load_index(), limit=1)
        self.assertEqual(report["scope"], "prefix_smoke_test_only")
        self.assertEqual(report["checked_images"], 1)

    def test_missing_and_corrupt_images_reported(self):
        (self.root / "items/a.jpg").unlink()
        (self.root / "items/b.jpg").write_bytes(b"not an image")
        with SOPDataset(self.root) as data:
            report, eligible = data.audit(data.load_index(), limit=None)
        self.assertEqual(len(report["unreadable"]), 2)
        self.assertEqual(eligible, [])

    def test_duplicate_ids_rejected(self):
        p = self.root / "Ebay_test.txt"
        p.write_text(p.read_text().replace("3 2 1", "1 2 1"))
        with SOPDataset(self.root) as data, self.assertRaisesRegex(ValueError, "Duplicate"):
            data.load_index()

    def test_product_overlap_rejected(self):
        p = self.root / "Ebay_test.txt"
        p.write_text(p.read_text().replace("3 2 1", "3 1 1"))
        with SOPDataset(self.root) as data, self.assertRaisesRegex(ValueError, "both"):
            data.load_index()

    def test_unsafe_paths_rejected(self):
        for source in (self.root, self.archive()):
            with SOPDataset(source) as data:
                for path in ("../secret", "/etc/passwd", "C:/secret", "..\\secret"):
                    with self.assertRaises(ValueError), data.open_file(path):
                        pass

    def test_config_and_environment_override(self):
        cfg = self.base / "config.local.json"
        cfg.write_text(json.dumps({"data_path": str(self.root)}))
        with patch.dict(os.environ, {"SOP_PATH": ""}):
            self.assertEqual(configured_path(cfg), self.root)
        with patch.dict(os.environ, {"SOP_PATH": str(self.base)}):
            self.assertEqual(configured_path(cfg), self.base)

    def test_bad_limits_and_missing_paths_rejected(self):
        with self.assertRaises(FileNotFoundError):
            SOPDataset(self.base / "absent")
        with SOPDataset(self.root) as data, self.assertRaises(ValueError):
            data.audit(data.load_index(), limit=0)


if __name__ == "__main__":
    unittest.main()
