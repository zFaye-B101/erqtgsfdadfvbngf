# Data audit notebook

Code-only export. Stored outputs, including dataset pictures, are intentionally excluded. This is not the final model notebook.

## Cell 1

```python
import csv
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

def load_sop_index(zip_path):
    images = []

    with ZipFile(zip_path) as archive:
        for split in ("train", "test"):
            name = f"Stanford_Online_Products/Ebay_{split}.txt"

            with archive.open(name) as raw:
                reader = csv.DictReader(
                    TextIOWrapper(raw, encoding="utf-8"),
                    delimiter=" ",
                    skipinitialspace=True,
                )

                for row in reader:
                    images.append({
                        "image_id": int(row["image_id"]),
                        "class_id": int(row["class_id"]),
                        "super_class_id": int(row["super_class_id"]),
                        "path": row["path"],
                        "source_split": split,
                    })

    return images

zip_path = Path(input("SOP ZIP 的完整路径：").strip()).expanduser()
images = load_sop_index(zip_path)

print("图片数：", len(images))
print("商品数：", len({image["class_id"] for image in images}))
print("第一条：", images[0])
```

## Cell 2

```python
from collections import defaultdict
from IPython.display import Image, display

by_product = defaultdict(list)
for item in images:
    by_product[item["class_id"]].append(item)

anchor, same_product = next(
    (group[0], group[1])
    for group in by_product.values()
    if len(group) >= 2
)

different_product = next(
    item for item in images
    if item["super_class_id"] == anchor["super_class_id"]
    and item["class_id"] != anchor["class_id"]
)

with ZipFile(zip_path) as archive:
    for label, left, right in (
        (1, anchor, same_product),
        (0, anchor, different_product),
    ):
        print(f"标签 {label}：商品 {left['class_id']} 对商品 {right['class_id']}")
        for item in (left, right):
            name = "Stanford_Online_Products/" + item["path"]
            display(Image(data=archive.read(name), width=220))
```

## Cell 3

```python
from random import Random

candidate_ids = sorted(
    product_id
    for product_id, group in by_product.items()
    if product_id != anchor["class_id"]
    and group[0]["super_class_id"] == anchor["super_class_id"]
)

sampled_ids = Random(42).sample(candidate_ids, 8)

with ZipFile(zip_path) as archive:
    print("参照商品：", anchor["class_id"])
    display(Image(
        data=archive.read("Stanford_Online_Products/" + anchor["path"]),
        width=180,
    ))

    for product_id in sampled_ids:
        item = by_product[product_id][0]
        print("不同商品：", product_id)
        display(Image(
            data=archive.read("Stanford_Online_Products/" + item["path"]),
            width=180,
        ))
```

## Cell 4

```python
from PIL import Image as PILImage

eligible_images = []
undersized_images = []

with ZipFile(zip_path) as archive:
    for count, item in enumerate(images, start=1):
        name = "Stanford_Online_Products/" + item["path"]

        with archive.open(name) as image_file:
            with PILImage.open(image_file) as picture:
                width, height = picture.size

        if width >= 100 and height >= 100:
            eligible_images.append(item)
        else:
            undersized_images.append((item["image_id"], width, height))

        if count % 20000 == 0:
            print("已检查：", count)

print("合格原图：", len(eligible_images))
print("尺寸不足：", len(undersized_images))

```

## Cell 5

```python
from collections import Counter

photos_per_product = Counter(
    item["class_id"] for item in eligible_images
)

usable_product_ids = {
    product_id
    for product_id, count in photos_per_product.items()
    if count >= 2
}

print("仍有合格图片的商品：", len(photos_per_product))
print("至少有两张图的商品：", len(usable_product_ids))
print("只剩一张图的商品：", sum(
    count == 1 for count in photos_per_product.values()
))
```

## Cell 6

```python

```

## Cell 7

```python

```
