"""Export one-channel checkpoints to ONNX or TensorRT while keeping artifacts outside training runs."""

import argparse
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO

from tools.common import ROOT, require_gpu, require_one_channel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--format", required=True, choices=("onnx", "engine"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fp16", action="store_true", help="TensorRT FP16; ONNX defaults to portable FP32")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    require_gpu(args.device)
    if args.format == "engine":
        if args.device == "cpu":
            raise ValueError("TensorRT engine building requires device=0.")
        import tensorrt  # noqa: F401 -- fail before creating files if the optional extra is missing
    if args.fp16 and args.format != "engine":
        raise ValueError("This example uses FP16 for TensorRT; use FP32 ONNX with CPU Runtime.")
    original = YOLO(str(args.model))
    require_one_channel(original)
    output = args.output or ROOT / "results" / "exports" / original.task / args.model.parent.parent.name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / args.model.name
    if checkpoint.resolve() != args.model.resolve():
        shutil.copy2(args.model, checkpoint)
    model = YOLO(str(checkpoint))
    exported = str(
        model.export(
            format=args.format,
            device=args.device,
            imgsz=args.imgsz,
            batch=1,
            dynamic=args.dynamic,
            quantize=16 if args.fp16 else 32,
            simplify=False,
            opset=18,
            **({"workspace": 1} if args.format == "engine" else {}),
        )
    )
    graph_path = Path(exported) if args.format == "onnx" else checkpoint.with_suffix(".onnx")
    graph = onnx.load(graph_path)
    onnx.checker.check_model(graph)
    shape = graph.graph.input[0].type.tensor_type.shape.dim
    if shape[1].dim_value != 1:
        raise ValueError("Exported graph does not have one input channel.")
    write_json(
        output / f"{args.format}_manifest.json",
        {
            "source": str(args.model.resolve()),
            "export": exported,
            "task": original.task,
            "channels": 1,
            "input_range": [0, 1],
            "normalization": "uint8 / 255",
            "imgsz": args.imgsz,
            "dynamic": args.dynamic,
            "fp16": args.fp16,
            "engine_scope": "TensorRT engines must be built for the target GPU, platform and TensorRT version.",
        },
    )
    print(f"PASS: one-channel {args.format} export: {exported}")


if __name__ == "__main__":
    main()
