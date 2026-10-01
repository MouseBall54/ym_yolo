"""Regression checks for input-channel and pixel-ID mistakes that silently corrupt training."""

import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics.nn.tasks import DetectionModel, SemanticSegmentationModel

from tools.common import ROOT, dataset_config, read_gray


def test_16bit_container_preserves_8bit_values(tmp_path):
    pixels = np.array([[0, 1, 127, 254, 255]], dtype=np.uint16)
    file = tmp_path / "gray.png"
    Image.fromarray(pixels).save(file)
    np.testing.assert_array_equal(read_gray(file), pixels.astype(np.uint8))
    with pytest.raises(ValueError, match="Prepare the 16-bit"):
        read_gray(file, require_uint8=True)


def test_out_of_range_is_not_silently_truncated(tmp_path):
    file = tmp_path / "gray.tiff"
    Image.fromarray(np.array([[256, 65535]], dtype=np.uint16)).save(file)
    with pytest.raises(ValueError, match="exceeds"):
        read_gray(file)


def test_rgb_is_not_replicated_or_converted(tmp_path):
    file = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(file)
    with pytest.raises(ValueError, match="single-channel"):
        read_gray(file)


def test_rgb_weight_sum_preserves_replicated_gray_response():
    from torch.nn.functional import conv2d

    gray = torch.rand(1, 1, 12, 12)
    weight = torch.rand(4, 3, 3, 3)
    torch.testing.assert_close(conv2d(gray.repeat(1, 3, 1, 1), weight), conv2d(gray, weight.sum(1, keepdim=True)))


@pytest.mark.parametrize(
    "model_class,yaml_name",
    [
        (DetectionModel, "yolo26n-gray.yaml"),
        (SemanticSegmentationModel, "yolo26n-gray-sem.yaml"),
    ],
)
def test_real_models_accept_only_one_channel(model_class, yaml_name):
    model = model_class(str(ROOT / "configs" / "models" / yaml_name), ch=1, nc=2, verbose=False).eval()
    assert model.model[0].conv.in_channels == 1
    with torch.no_grad():
        model(torch.zeros(1, 1, 64, 64))
    with pytest.raises(RuntimeError, match="channels"):
        model(torch.zeros(1, 3, 64, 64))


def test_semantic_mask_ids_are_not_normalized(tmp_path):
    import yaml

    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "masks" / split).mkdir(parents=True)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(tmp_path / "images" / split / "a.png")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[0] = 1
        mask[1] = 255
        Image.fromarray(mask).save(tmp_path / "masks" / split / "a.png")
    config = tmp_path / "data.yaml"
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "channels": 1,
        "masks_dir": "masks",
        "names": {0: "background", 1: "target"},
    }
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    dataset_config(config, "semantic")
    with Image.open(tmp_path / "masks" / "train" / "a.png") as image:
        assert set(np.unique(image)) == {0, 1, 255}
