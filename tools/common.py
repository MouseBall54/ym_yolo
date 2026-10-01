"""Paths and grayscale data checks shared by the user entry points."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def load_yaml(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream), path


def resolve_path(value, base=ROOT):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def require_gpu(device):
    if str(device) == "cpu":
        return
    if str(device) != "0":
        raise ValueError("This single-GPU example supports device=0 or cpu.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the locked cu130 environment and check the NVIDIA driver.")


def image_files(source):
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image: {source}")
        return [source]
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No images found: {source}")
    return files


def read_gray(path, require_uint8=False):
    """Decode without reducing bit depth; accept one plane with integer values 0..255."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Multi-page TIFF requires an explicit frame choice: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    if image.ndim != 2 or image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError(f"Expected a single-channel uint8/uint16 image: {path}, {image.shape}, {image.dtype}")
    if image.max() > 255:
        raise ValueError(f"Pixel value exceeds the agreed range 0..255: {path} (max={image.max()})")
    if require_uint8 and image.dtype != np.uint8:
        raise ValueError(f"Prepare the 16-bit container as uint8 first: {path}. Use tools.prepare_dataset.")
    return image.astype(np.uint8, copy=False)


def dataset_config(path, task, split=None):
    """Resolve data paths relative to the dataset YAML and validate images and class-ID masks."""
    data, path = load_yaml(path)
    if data.get("channels") != 1:
        raise ValueError("Dataset YAML must declare channels: 1.")
    data["path"] = str(resolve_path(data["path"], path.parent))
    root = Path(data["path"])
    names = data["names"]
    if isinstance(names, dict) and sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs in names must be contiguous from zero.")
    splits = [split] if split else ["train", "val"]
    for key in splits:
        directory = resolve_path(data[key], root)
        if not directory.is_dir():
            raise ValueError(f"Examples use split directories, not text lists: {directory}")
        for file in image_files(directory):
            image = read_gray(file, require_uint8=True)
            relative = file.relative_to(root / "images")
            if task == "semantic":
                mask_path = root / data.get("masks_dir", "masks") / relative.with_suffix(".png")
                with Image.open(mask_path) as mask_image:
                    mask = np.asarray(mask_image)
                if mask.ndim != 2 or mask.shape != image.shape:
                    raise ValueError(f"Mask must be a class-ID plane with the original image size: {mask_path}")
                ids = set(map(int, np.unique(mask)))
                mapping = data.get("label_mapping", {})
                mapped = {mapping.get(i, i) for i in ids}
                valid = set(range(max(len(names), 2))) | {255, "ignore_label", None}
                if not mapped <= valid:
                    raise ValueError(f"Mask IDs do not match names/label_mapping: {mask_path}, {ids}")
                if ids and mapped <= {255, "ignore_label", None}:
                    raise ValueError(f"All pixels are ignored: {mask_path}")
            elif not (root / "labels" / relative.with_suffix(".txt")).exists():
                raise FileNotFoundError(f"Missing detection label for {file}; use an empty txt for background images.")
    return data


def require_one_channel(model):
    if model.model.model[0].conv.in_channels != 1:
        raise ValueError("This entry point requires a trained one-channel model.")
