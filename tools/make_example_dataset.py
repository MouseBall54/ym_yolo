"""Generate a tiny, deterministic grayscale detection and semantic dataset. Not an accuracy benchmark."""

import argparse

import cv2
import numpy as np
from PIL import Image

from tools.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "datasets"))
    args = parser.parse_args()
    root = ROOT / args.root
    rng = np.random.default_rng(42)
    for task in ("detect", "semantic"):
        base = root / f"example_{task}"
        for split, count in (("train", 8), ("val", 4)):
            for folder in ("images", "labels" if task == "detect" else "masks"):
                (base / folder / split).mkdir(parents=True, exist_ok=True)
            for i in range(count):
                image = rng.integers(0, 40, (96, 128), dtype=np.uint8)
                x, y = int(rng.integers(12, 65)), int(rng.integers(12, 45))
                w, h = 32, 28
                image[y : y + h, x : x + w] = 190
                stem = f"sample_{i:02d}"
                Image.fromarray(image).save(base / "images" / split / f"{stem}.png")
                if task == "detect":
                    label = f"0 {(x + w / 2) / 128:.6f} {(y + h / 2) / 96:.6f} {w / 128:.6f} {h / 96:.6f}\n"
                    (base / "labels" / split / f"{stem}.txt").write_text(label, encoding="utf-8")
                else:
                    mask = np.zeros_like(image)
                    cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 1, -1)
                    Image.fromarray(mask).save(base / "masks" / split / f"{stem}.png")
    print(f"Created synthetic examples in {root}")


if __name__ == "__main__":
    main()
