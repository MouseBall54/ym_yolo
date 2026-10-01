"""Verify that the channel comparison changes neither pixels nor annotations."""

import cv2
import numpy as np
import pytest
import yaml

from tools.benchmark_channels import prepare_pair, summarize
from tools.common import write_json


@pytest.mark.parametrize("task", ["detect", "semantic"])
def test_paired_dataset(tmp_path, task):
    source = tmp_path / "source"
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    annotation_kind = "labels" if task == "detect" else "masks"
    for split in ("train", "val"):
        image_path = source / "images" / split / "sample.png"
        image_path.parent.mkdir(parents=True)
        cv2.imwrite(str(image_path), gray)
        annotation = source / annotation_kind / split / ("sample.txt" if task == "detect" else "sample.png")
        annotation.parent.mkdir(parents=True)
        if task == "detect":
            annotation.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        else:
            mask = (gray > 128).astype(np.uint8)
            mask[0, 0] = 255
            cv2.imwrite(str(annotation), mask)
    data_path = tmp_path / "dataset.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(source),
                "channels": 1,
                "train": "images/train",
                "val": "images/val",
                "names": {0: "background", 1: "target"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "paired"
    manifest = prepare_pair(data_path, task, output)
    assert manifest["counts"] == {"train": 1, "val": 1}
    for split in ("train", "val"):
        one = cv2.imread(str(output / "1ch/images" / split / "sample.png"), cv2.IMREAD_UNCHANGED)
        three = cv2.imread(str(output / "3ch/images" / split / "sample.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(one, gray)
        np.testing.assert_array_equal(three, np.repeat(gray[..., None], 3, axis=2))
        suffix = ".txt" if task == "detect" else ".png"
        relative = f"{annotation_kind}/{split}/sample{suffix}"
        assert (output / "1ch" / relative).read_bytes() == (source / relative).read_bytes()
        assert (output / "3ch" / relative).read_bytes() == (source / relative).read_bytes()
    assert yaml.safe_load((output / "1ch.yaml").read_text())["channels"] == 1
    assert yaml.safe_load((output / "3ch.yaml").read_text())["channels"] == 3
    with pytest.raises(ValueError, match="Output collision"):
        prepare_pair(data_path, task, output)


def test_summary_rejects_different_actual_batch(tmp_path):
    for channels in (1, 3):
        write_json(
            tmp_path / "detect" / f"repeat1_{channels}ch/measurement.json", {"first_batch_plane_sha256": str(channels)}
        )
    with pytest.raises(RuntimeError, match="pixels/order differ"):
        summarize(tmp_path, ("detect",), 1)
    assert not (tmp_path / "summary.json").exists()
