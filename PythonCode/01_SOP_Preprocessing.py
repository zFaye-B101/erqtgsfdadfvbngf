"""Clean original SOP images and export reusable inputs for experiments."""

from pathlib import Path, PurePosixPath
import argparse
from io import BytesIO
from zipfile import ZipFile
import json
import time

import pandas as pd
from PIL import Image

cwd = Path(__file__).resolve().parent
if not (cwd / "utils" / "sop_features.py").is_file():
    raise FileNotFoundError("Place utils/sop_features.py beside this script.")

from utils.sop_features import resize_for_model


def main():
    parser = argparse.ArgumentParser(description="Clean and export SOP images once.")
    parser.add_argument("--config", type=Path, default=cwd / "config.json")
    parser.add_argument("--zip-path", type=Path, default=None,
                        help="Input SOP ZIP; otherwise search beside the script and its parents.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New preprocessing output directory (default: beside this script).")
    args = parser.parse_args()

    print("Pillow:", Image.__version__)

    def find_sop_zip():
        for folder in (cwd, *cwd.parents):
            candidate = folder / "Stanford_Online_Products.zip"
            if candidate.is_file():
                return candidate
        entered = input("Enter the full path to Stanford_Online_Products.zip: ").strip().strip('"')
        candidate = Path(entered).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError("SOP ZIP not found: " + str(candidate))
        return candidate.resolve()

    ZIP_PATH = args.zip_path.expanduser().resolve() if args.zip_path else find_sop_zip()
    if not ZIP_PATH.is_file():
        raise FileNotFoundError("SOP ZIP not found: " + str(ZIP_PATH))
    from utils.sop_config import load_config
    settings = load_config(args.config)
    OUTPUT_DIR = args.output_dir.expanduser().resolve() if args.output_dir else Path(settings["data_dir"])
    MIN_ORIGINAL_SIDE = 100
    PROCESSED_SIZE = 224

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            "Output directory already exists and is not empty. To avoid mixing old images, set OUTPUT_DIR to a new folder name, for example processed_sop_run2."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("SOP ZIP:", ZIP_PATH)
    print("Output directory:", OUTPUT_DIR)

    with ZipFile(ZIP_PATH, "r") as zf:
        names = set(zf.namelist())
        train_members = [n for n in names if PurePosixPath(n).name == "Ebay_train.txt"]
        test_members = [n for n in names if PurePosixPath(n).name == "Ebay_test.txt"]
        if len(train_members) != 1 or len(test_members) != 1:
            raise ValueError("The ZIP must contain exactly one Ebay_train.txt and one Ebay_test.txt.")

        train_member, test_member = train_members[0], test_members[0]
        train_prefix = PurePosixPath(train_member).parent.as_posix()
        test_prefix = PurePosixPath(test_member).parent.as_posix()
        if train_prefix != test_prefix:
            raise ValueError("The train and test indexes are not in the same data directory.")
        ZIP_PREFIX = "" if train_prefix == "." else train_prefix + "/"

        train_df = pd.read_csv(BytesIO(zf.read(train_member)), sep=r"\s+", engine="python")
        test_df = pd.read_csv(BytesIO(zf.read(test_member)), sep=r"\s+", engine="python")

    required_columns = ["image_id", "class_id", "super_class_id", "path"]
    for split_name, frame in [("train", train_df), ("test", test_df)]:
        if list(frame.columns) != required_columns:
            raise ValueError(split_name + " index column names or order do not match expectations.")
        frame["source_split"] = split_name

    index_df = pd.concat([train_df, test_df], ignore_index=True)
    if index_df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values found.")
    if (index_df.groupby("class_id")["source_split"].nunique() > 1).any():
        raise ValueError("A product appears in both official train and test splits. Stop and inspect the labels.")
    if (index_df.groupby("class_id")["super_class_id"].nunique() > 1).any():
        raise ValueError("A product is associated with multiple super_class_id values.")

    print("Indexed images:", len(index_df))
    print("Products:", index_df["class_id"].nunique())
    print(index_df.groupby("source_split").agg(
        images=("image_id", "size"), products=("class_id", "nunique")
    ))

    # Inspect every original image before resizing; resizing does not qualify small originals.
    audit_rows = []
    start_time = time.time()

    with ZipFile(ZIP_PATH, "r") as zf:
        for i, row in enumerate(index_df.itertuples(index=False), start=0):
            relative = PurePosixPath(str(row.path))
            if relative.is_absolute() or ".." in relative.parts:
                audit_rows.append({
                    "image_id": int(row.image_id), "original_width": None,
                    "original_height": None, "status": "excluded",
                    "reason": "unsafe_metadata_path"
                })
                continue

            member = ZIP_PREFIX + relative.as_posix()
            if member not in names:
                audit_rows.append({
                    "image_id": int(row.image_id), "original_width": None,
                    "original_height": None, "status": "excluded",
                    "reason": "missing_file_in_zip"
                })
                continue

            try:
                raw_bytes = zf.read(member)
                with Image.open(BytesIO(raw_bytes)) as opened:
                    width, height = opened.size
                    opened.load()

                if width < MIN_ORIGINAL_SIDE or height < MIN_ORIGINAL_SIDE:
                    audit_rows.append({
                        "image_id": int(row.image_id), "original_width": width,
                        "original_height": height, "status": "excluded",
                        "reason": "original_smaller_than_100"
                    })
                else:
                    audit_rows.append({
                        "image_id": int(row.image_id), "original_width": width,
                        "original_height": height, "status": "eligible", "reason": ""
                    })
            except Exception as exc:
                audit_rows.append({
                    "image_id": int(row.image_id), "original_width": None,
                    "original_height": None, "status": "excluded",
                    "reason": "unreadable: " + type(exc).__name__
                })

            if (i + 1) % 5000 == 0:
                elapsed = round((time.time() - start_time) / 60, 1)
                print("Checked", i + 1, "/", len(index_df), "images; elapsed", elapsed, "minutes")

    audit_df = pd.DataFrame(audit_rows)
    audit_df = audit_df.set_index("image_id").reindex(index_df["image_id"]).reset_index()
    audit_df = pd.concat(
        [index_df.reset_index(drop=True), audit_df.drop(columns=["image_id"])],
        axis=1
    )
    print("Original-image audit complete; elapsed minutes:", round((time.time() - start_time) / 60, 1))
    print(audit_df["reason"].value_counts(dropna=False))

    # Remove products with only one image left after the original-image checks.
    eligible_mask = audit_df["status"].eq("eligible")
    eligible_counts = audit_df.loc[eligible_mask, "class_id"].value_counts()
    single_image_products = set(eligible_counts[eligible_counts == 1].index)

    singleton_mask = eligible_mask & audit_df["class_id"].isin(single_image_products)
    audit_df.loc[singleton_mask, "status"] = "excluded"
    audit_df.loc[singleton_mask, "reason"] = "only_one_eligible_image_for_product"

    clean_manifest = audit_df.loc[audit_df["status"].eq("eligible")].copy().reset_index(drop=True)
    excluded_samples = audit_df.loc[audit_df["status"].eq("excluded")].copy().reset_index(drop=True)
    excluded_products = (
        audit_df.loc[singleton_mask, ["class_id", "super_class_id", "source_split"]]
        .drop_duplicates()
        .sort_values("class_id")
        .reset_index(drop=True)
    )

    print("Passed original-image checks:", int(eligible_mask.sum()))
    print("Products with only one eligible image:", len(single_image_products))
    print("Clean images:", len(clean_manifest))
    print("Clean products:", clean_manifest["class_id"].nunique())
    print(clean_manifest.groupby("source_split").agg(
        images=("image_id", "size"), products=("class_id", "nunique")
    ))

    # Export reusable RGB images; experiment runners extract the selected features.
    processed_paths = []
    start_time = time.time()

    with ZipFile(ZIP_PATH, "r") as zf:
        for i, row in enumerate(clean_manifest.itertuples(index=False), start=0):
            relative = PurePosixPath(str(row.path))
            member = ZIP_PREFIX + relative.as_posix()
            raw_bytes = zf.read(member)

            with Image.open(BytesIO(raw_bytes)) as opened:
                opened.load()
                original_image = opened.convert("RGB")

            processed_image = resize_for_model(original_image, size=PROCESSED_SIZE)
            relative_output = (
                Path("images") / str(row.source_split) / str(row.class_id)
                / (str(row.image_id) + ".jpg")
            )
            output_path = OUTPUT_DIR / relative_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed_image.save(output_path, format="JPEG", quality=95, optimize=True)

            processed_paths.append(relative_output.as_posix())

            if (i + 1) % 5000 == 0:
                elapsed = round((time.time() - start_time) / 60, 1)
                print("Processed", i + 1, "/", len(clean_manifest), "images; elapsed", elapsed, "minutes")

    clean_manifest["processed_relative_path"] = processed_paths
    clean_manifest["processed_width"] = PROCESSED_SIZE
    clean_manifest["processed_height"] = PROCESSED_SIZE
    print("Image conversion complete; elapsed minutes:", round((time.time() - start_time) / 60, 1))

    # Save audit tables and the completed-run summary.
    clean_manifest.to_csv(
        OUTPUT_DIR / "clean_manifest.csv", index=False, encoding="utf-8-sig"
    )
    excluded_samples.to_csv(
        OUTPUT_DIR / "excluded_samples.csv", index=False, encoding="utf-8-sig"
    )
    excluded_products.to_csv(
        OUTPUT_DIR / "excluded_products.csv", index=False, encoding="utf-8-sig"
    )
    audit_df.to_csv(
        OUTPUT_DIR / "full_image_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "preprocessing_version": 2,
        "complete": True,
        "zip_path": str(ZIP_PATH),
        "indexed_images": int(len(index_df)),
        "indexed_products": int(index_df["class_id"].nunique()),
        "clean_images": int(len(clean_manifest)),
        "clean_products": int(clean_manifest["class_id"].nunique()),
        "images_by_official_split": {
            str(k): int(v) for k, v in clean_manifest["source_split"].value_counts().items()
        },
        "excluded_samples_by_reason": {
            str(k): int(v) for k, v in excluded_samples["reason"].value_counts().items()
        },
        "processed_image_size": [PROCESSED_SIZE, PROCESSED_SIZE],
        "new_train_validation_split_created": False
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Complete. Processed image folder:", (OUTPUT_DIR / "images").resolve())


if __name__ == "__main__":
    main()
