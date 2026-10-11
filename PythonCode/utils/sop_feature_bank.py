"""Pluggable fixed extractors and image-based corruption feature caches.

No SOP labels are used to fit these extractors. ResNet18 uses frozen external
ImageNet weights. Learned representations belong inside model training folds.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .sop_features import image_feature_vector
from .sop_protocol import content_box, corrupt_image, write_json


def choose_device(requested, torch):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda or mps.")
    return requested


class HandcraftedExtractor:
    dimension = 192

    def __init__(self, settings):
        import cv2
        self.metadata = {"extractor": "hsv_hog", "opencv": cv2.__version__,
                         "geometry": "224_square_white_padding", "dimension": self.dimension}

    def __call__(self, images):
        return np.stack([image_feature_vector(image) for image in images])


class ContentHandcraftedExtractor(HandcraftedExtractor):
    """Exclude known padding before HSV statistics and spatial gradient cells."""
    content_only = True

    def __init__(self, settings):
        super().__init__(settings)
        self.metadata.update(extractor='hsv_hog_content', geometry='metadata_content_rectangle_without_padding')


class ResNet18Extractor:
    dimension = 2560

    def __init__(self, settings):
        import torch
        import torchvision
        from torchvision.models import ResNet18_Weights, resnet18
        self.torch = torch
        self.device = choose_device(settings["device"], torch)
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        self.model = torch.nn.Sequential(*list(model.children())[:-2]).to(self.device).eval()
        self.model.requires_grad_(False)
        # Preserve the same padded 224 image geometry used by every extractor.
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.metadata = {"extractor": "resnet18", "weights": "IMAGENET1K_V1",
                         "torch": torch.__version__, "torchvision": torchvision.__version__,
                         "device": self.device, "geometry": "224_square_white_padding_no_crop",
                         "normalization": "ImageNet_mean_std", "dimension": self.dimension,
                         "layout": "global512_then_row_major_2x2_regions2048"}

    def __call__(self, images):
        torch = self.torch
        array = np.stack([np.asarray(image.convert("RGB"), dtype=np.float32) / 255 for image in images])
        x = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(self.device)
        with torch.inference_mode():
            fmap = self.model((x - self.mean) / self.std)
            global_features = fmap.mean(dim=(2, 3))
            regions = torch.nn.functional.adaptive_avg_pool2d(fmap, (2, 2))
            regions = regions.permute(0, 2, 3, 1).reshape(len(images), -1)
            return torch.cat([global_features, regions], dim=1).cpu().numpy().astype(np.float32)


# Add new fixed extractors here; the numbered entry points do not change.
EXTRACTORS = {"hsv_hog": HandcraftedExtractor, "hsv_hog_content": ContentHandcraftedExtractor,
              "resnet18": ResNet18Extractor}
_INSTANCES = {}


def file_digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def dataset_digest(data_dir, manifest):
    """Fingerprint the manifest and file size/mtime; not a cryptographic image audit."""
    digest = hashlib.sha256((Path(data_dir) / "clean_manifest.csv").read_bytes())
    root = Path(data_dir).resolve()
    for relative in manifest.processed_relative_path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes the dataset.")
        stat = path.stat()
        digest.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def extract_cached(frame, data_dir, config, dataset_id, extractor_name, condition, query_mask):
    factory = EXTRACTORS.get(extractor_name)
    if factory is None:
        raise ValueError("Unknown feature extractor: " + extractor_name)
    instance_key = (extractor_name, json.dumps(config["extraction"], sort_keys=True))
    if instance_key not in _INSTANCES:
        _INSTANCES[instance_key] = factory(config["extraction"])
    extractor = _INSTANCES[instance_key]
    identity = {"dataset_id": dataset_id, "image_ids": frame.image_id.tolist(),
                "query_mask": query_mask.tolist(), "seed": config["seed"],
                "condition": condition, "parameters": config["conditions"][condition],
                "condition_index": list(config["conditions"]).index(condition),
                "extractor": extractor.metadata,
                "code": file_digest([__file__, Path(__file__).with_name("sop_features.py"),
                                     Path(__file__).with_name("sop_protocol.py")])}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(config["cache_dir"]) / extractor_name / key
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "features.npy"
    metadata = directory / "metadata.json"
    if target.exists() and metadata.exists():
        if json.loads(metadata.read_text()) != identity:
            raise ValueError("Feature cache metadata mismatch.")
        result = np.load(target, allow_pickle=False)
    else:
        clean = None if condition == "clean" else extract_cached(
            frame, data_dir, config, dataset_id, extractor_name, "clean", query_mask)
        result = np.empty((len(frame), extractor.dimension), dtype=np.float32) if clean is None else clean.copy()
        indexes = np.arange(len(frame)) if condition == "clean" else np.flatnonzero(query_mask)
        batch_size = int(config["extraction"]["batch_size"])
        if batch_size < 1:
            raise ValueError("Feature extraction batch size must be positive.")
        seed = config["seed"] + list(config["conditions"]).index(condition) * 100003
        for start in range(0, len(indexes), batch_size):
            chunk = indexes[start:start + batch_size]
            images = []
            for index in chunk:
                row = frame.iloc[int(index)]
                path = (Path(data_dir) / row.processed_relative_path).resolve()
                if not path.is_relative_to(Path(data_dir).resolve()):
                    raise ValueError("Image path escapes the dataset.")
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                if image.size != (224, 224):
                    raise ValueError("Expected preprocessing script 01's 224 x 224 images.")
                images.append(corrupt_image(image, row, condition, config["conditions"][condition], seed))
                if getattr(extractor, 'content_only', False):
                    images[-1] = images[-1].crop(content_box(row, image.width))
            result[chunk] = extractor(images)
            if (start // batch_size) % 50 == 0:
                print(f"Features {extractor_name}/{condition}: {min(start + batch_size, len(indexes))}/{len(indexes)}", flush=True)
        if not np.isfinite(result).all():
            raise ValueError("Feature extractor returned nonfinite values.")
        temporary = directory / "features.tmp.npy"
        np.save(temporary, result)
        temporary.replace(target)
        write_json(metadata, identity)
    if result.shape != (len(frame), extractor.dimension) or not np.isfinite(result).all():
        raise ValueError("Feature cache has invalid shape or values.")
    return result


def build_banks(frame, data_dir, config, dataset_id, query_mask):
    extracted = {}
    banks = {}
    for name, spec in config["feature_sets"].items():
        blocks = []
        for component in spec.get('components', [spec]):
            extractor_name = component['extractor']
            if extractor_name not in EXTRACTORS:
                raise ValueError('Unknown extractor: ' + extractor_name)
            if extractor_name not in extracted:
                extracted[extractor_name] = {condition: extract_cached(
                    frame, data_dir, config, dataset_id, extractor_name, condition, query_mask)
                    for condition in config['conditions']}
            start, end = component['columns']
            if end > EXTRACTORS[extractor_name].dimension:
                raise ValueError('Feature columns exceed extractor dimension: ' + name)
            block = {}
            for condition, values in extracted[extractor_name].items():
                selected = values[:, start:end]
                if component.get('normalization', 'none') == 'l2':
                    selected = selected / np.maximum(np.linalg.norm(selected, axis=1, keepdims=True), 1e-12)
                block[condition] = selected
            blocks.append(block)
        banks[name] = {condition: np.concatenate([block[condition] for block in blocks], axis=1)
                       for condition in config['conditions']}
    return banks
