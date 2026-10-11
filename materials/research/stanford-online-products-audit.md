# Stanford Online Products: local dataset audit

Audited on 2026-09-25 against `official/Project.pdf`, Section 2.b. The archive was downloaded from the authors' FTP URL listed in their [repository](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16#stanford-online-products-dataset):

- Local archive: `research/data/Stanford_Online_Products.zip`
- Download size: 3,083,860,082 bytes (matches the server's reported size)
- SHA-256: `b04fa78210e69fab050ce73939550e1cd0f96890bdd98ec9fee732c3b756ef48`
- ZIP integrity: `unzip -tqq` passed for all entries

## Metadata and images

The archive README defines `image_id`, `class_id` (an index for the product/item), `super_class_id` (one of 12 broad categories), and image path. The filename's `itemID` is the eBay item ID. The two sampled photos with the same filename item ID show different views of the same bicycle; a sampled mug image is from another category.

| Check | Result |
| --- | ---: |
| Metadata rows / image entries | 120,053 / 120,053 |
| Official train / test rows | 59,551 / 60,502 |
| Product classes / broad categories | 22,634 / 12 |
| Train/test class overlap | 0 |
| Missing metadata-referenced images | 0 |
| Image headers unreadable by Pillow | 0 |
| Original images with width >= 100 and height >= 100 | 120,026 |
| Original images below either threshold | 27 |
| Product classes retained after size filtering | 22,634 |
| Retained classes with at least two qualified images | 22,630 |
| Smallest observed width / height | 53 / 63 pixels |

The 27 undersized images must not be enlarged or padded to satisfy the assignment. Excluding them leaves far more than the required 10,000 original images. For a pairwise matching task, also exclude or separately handle the four classes with only one qualified image, because they cannot supply a positive pair.

The authors' archive README says the photographs were collected through the eBay API. Its `LICENSE` text grants rights to the "Software"; it does not explicitly establish unrestricted redistribution rights for the underlying eBay photographs. Cite the original dataset and do not include images in the submitted code ZIP (also prohibited by the assignment instructions).

## Project implication

The dataset passes the assignment's public-access, image-count, and original-size gates. The annotations support same-item versus different-item comparisons and evaluation on unseen product IDs. They should not be described as independently verified SKU identifiers. The [dataset paper](https://cvgl.stanford.edu/papers/song_cvpr16.pdf) already studies hard-negative mining, so the group's research question should target a more specific failure mode or controlled comparison rather than merely ask whether hard negatives help.
