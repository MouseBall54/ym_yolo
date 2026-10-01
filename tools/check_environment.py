"""Verify real CUDA execution and the editable Ultralytics import."""

import torch
import torchvision
import ultralytics

from tools.common import require_gpu


def main():
    require_gpu(0)
    tensor = torch.rand(128, 128, device="cuda")
    result = tensor @ tensor.T
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    print(f"torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA={torch.version.cuda}")
    print(f"GPU={torch.cuda.get_device_name(0)}, architecture={torch.cuda.get_device_capability(0)}")
    print(f"ultralytics={ultralytics.__version__}: {ultralytics.__file__}")
    print("PASS: real GPU matrix multiplication")


if __name__ == "__main__":
    main()
