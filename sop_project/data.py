"""SOP metadata and image access for an original ZIP or extracted directory."""
import argparse
from collections import Counter
from contextlib import contextmanager
from io import TextIOWrapper
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, is_zipfile


def configured_path(config_path=None):
    value = os.environ.get("SOP_PATH", "").strip()
    if not value:
        config = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config.local.json"
        if config.exists():
            value = json.loads(config.read_text(encoding="utf-8"))["data_path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Set SOP_PATH or data_path in config.local.json to your local ZIP/folder.")
    return Path(value.strip().strip('\"').strip("'")).expanduser()


class SOPDataset:
    def __init__(self, path):
        self.path = Path(str(path).strip().strip('\"').strip("'")).expanduser()
        self.archive = None
        self.prefix = ""
        if self.path.is_file() and is_zipfile(self.path):
            self.archive = ZipFile(self.path)
            names = set(self.archive.namelist())
            prefixes = [name[:-len("Ebay_train.txt")] for name in names
                        if PurePosixPath(name).name == "Ebay_train.txt"]
            prefixes = [prefix for prefix in prefixes if prefix + "Ebay_test.txt" in names]
            if len(prefixes) != 1:
                self.close()
                raise ValueError("Expected exactly one pair of Ebay_train.txt and Ebay_test.txt in ZIP.")
            self.prefix = prefixes[0]
        elif self.path.is_dir():
            candidates = [self.path, self.path / "Stanford_Online_Products"]
            roots = [p for p in candidates if all((p / f"Ebay_{s}.txt").is_file() for s in ("train", "test"))]
            if len(roots) != 1:
                raise ValueError("Choose the folder containing Ebay_train.txt and Ebay_test.txt, or its parent.")
            self.path = roots[0].resolve()
        else:
            raise FileNotFoundError("Dataset path is not an existing SOP ZIP or directory: " + str(self.path))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.archive is not None:
            self.archive.close()

    @contextmanager
    def open_file(self, relative):
        relative = str(relative)
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
            raise ValueError("Unsafe dataset-relative path: " + relative)
        if self.archive is not None:
            with self.archive.open(self.prefix + relative) as stream:
                yield stream
        else:
            target = (self.path / relative).resolve()
            if self.path not in target.parents:
                raise ValueError("Image path escapes the dataset directory.")
            with target.open("rb") as stream:
                yield stream

    def load_index(self):
        images = []
        ids = set()
        product_split = {}
        product_category = {}
        for split in ("train", "test"):
            with self.open_file(f"Ebay_{split}.txt") as stream:
                text = TextIOWrapper(stream, encoding="utf-8-sig")
                if text.readline().split() != ["image_id", "class_id", "super_class_id", "path"]:
                    raise ValueError("Unexpected SOP index header in " + split)
                for number, line in enumerate(text, 2):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 4:
                        raise ValueError(f"Malformed {split} index row {number}")
                    image_id, product, category = map(int, fields[:3])
                    if image_id in ids:
                        raise ValueError("Duplicate image ID: " + str(image_id))
                    if product in product_split and product_split[product] != split:
                        raise ValueError("Product appears in both original train and test: " + str(product))
                    if product in product_category and product_category[product] != category:
                        raise ValueError("Inconsistent category for product: " + str(product))
                    ids.add(image_id)
                    product_split[product] = split
                    product_category[product] = category
                    images.append(dict(image_id=image_id, class_id=product,
                                       super_class_id=category, path=fields[3], source_split=split))
        if not images:
            raise ValueError("SOP indices contain no images.")
        return images

    def read_image(self, row):
        from PIL import Image
        with self.open_file(row["path"]) as stream:
            with Image.open(stream) as picture:
                return picture.convert("RGB")

    def audit(self, images, limit=20, min_size=100):
        from PIL import Image
        if min_size <= 0 or (limit is not None and limit <= 0):
            raise ValueError("min_size and limit must be positive (or limit=None for full audit).")
        rows = images if limit is None else images[:limit]
        eligible = []
        undersized = []
        unreadable = []
        for row in rows:
            try:
                with self.open_file(row["path"]) as stream:
                    with Image.open(stream) as picture:
                        width, height = picture.size
                        picture.load()
            except (OSError, KeyError, ValueError) as exc:
                unreadable.append({"image_id": row["image_id"], "error": str(exc)})
                continue
            if min(width, height) >= min_size:
                eligible.append(row)
            else:
                undersized.append({"image_id": row["image_id"], "width": width, "height": height})
        counts = Counter(row["class_id"] for row in eligible)
        report = {
            "scope": "full_dataset" if len(rows) == len(images) else "prefix_smoke_test_only",
            "indexed_images": len(images),
            "indexed_products": len({r["class_id"] for r in images}),
            "checked_images": len(rows), "min_original_size": min_size,
            "eligible_images_in_checked_subset": len(eligible),
            "products_with_two_eligible_in_checked_subset": sum(n >= 2 for n in counts.values()),
            "undersized": undersized, "unreadable": unreadable,
        }
        return report, eligible


def main():
    parser = argparse.ArgumentParser(description="Read-only SOP check, not model training.")
    parser.add_argument("--data", help="SOP ZIP or extracted folder; otherwise use local configuration")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full", action="store_true", help="Decode and check every image; may take time")
    args = parser.parse_args()
    with SOPDataset(args.data or configured_path()) as dataset:
        rows = dataset.load_index()
        report, _ = dataset.audit(rows, limit=None if args.full else args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["unreadable"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
