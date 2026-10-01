"""Save detection JSON/overlays or semantic class-ID PNGs/overlays from grayscale images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from tools.common import ROOT, image_files, read_gray, require_gpu, require_one_channel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("detect", "semantic"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0", help="Use cpu for the installed CPU ONNX Runtime")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    require_gpu(args.device)
    model = YOLO(args.model, task=args.task)
    if Path(args.model).suffix == ".pt":
        require_one_channel(model)
    output = args.output or ROOT / "results" / "predictions" / args.task
    output.mkdir(parents=True, exist_ok=True)
    for file in image_files(args.source):
        relative = file.relative_to(args.source) if args.source.is_dir() else Path(file.name)
        # Include the source extension to avoid a.png / a.tif output collisions.
        destination = output / relative.parent / relative.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = read_gray(file)
        result = model.predict(image[..., None], imgsz=args.imgsz, device=args.device, conf=args.conf, verbose=False)[0]
        if model.predictor.model.channels != 1:
            raise ValueError("Exported model must also have exactly one input channel.")
        if args.task == "detect":
            rows = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                rows.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf.item()),
                        "xyxy": box.xyxy[0].tolist(),
                    }
                )
            write_json(str(destination) + ".json", {"source": str(file.resolve()), "detections": rows})
        else:
            mask = result.semantic_mask.data.cpu().numpy().astype(np.uint8)
            Image.fromarray(mask).save(str(destination) + ".mask.png")
            ids, counts = np.unique(mask, return_counts=True)
            write_json(
                str(destination) + ".json",
                {"source": str(file.resolve()), "class_pixel_counts": dict(zip(map(str, ids), map(int, counts)))},
            )
        result.save(filename=str(destination) + ".overlay.jpg")
    print(f"Saved results: {output.resolve()}")


if __name__ == "__main__":
    main()
