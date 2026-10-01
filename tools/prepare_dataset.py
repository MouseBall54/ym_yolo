"""Copy images to uint8 grayscale PNG without changing values; keep YOLO labels or semantic mask IDs."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from tools.common import image_files, read_gray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Root containing images/train and images/val")
    parser.add_argument("--output", required=True, type=Path, help="New, empty output directory")
    parser.add_argument("--task", required=True, choices=("detect", "semantic"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be empty; preparation does not overwrite existing datasets.")
    # Validate the complete source before producing any outputs.
    pairs = []
    for file in image_files(args.source / "images"):
        relative = file.relative_to(args.source / "images")
        image = read_gray(file)
        annotation = (
            args.source
            / ("labels" if args.task == "detect" else "masks")
            / relative.with_suffix(".txt" if args.task == "detect" else ".png")
        )
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        if args.task == "semantic":
            with Image.open(annotation) as mask_image:
                mask = np.asarray(mask_image)
            if mask.ndim != 2 or mask.shape != image.shape or mask.max() > 255:
                raise ValueError(f"Mask must store IDs 0..255 and match image dimensions: {annotation}")
        pairs.append((file, relative, annotation))
    targets = [relative.with_suffix(".png") for _, relative, _ in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError("Different source extensions share a stem; rename them before PNG conversion.")
    for file, relative, annotation in pairs:
        output = args.output / "images" / relative.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(read_gray(file)).save(output)
        target = args.output / annotation.relative_to(args.source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(annotation, target)
    print(f"Prepared {len(pairs)} single-channel images: {args.output}")


if __name__ == "__main__":
    main()
