"""Evaluate detection mAP or semantic mIoU/pixel accuracy on a labeled split."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from tools.common import ROOT, dataset_config, require_gpu, require_one_channel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", required=True, choices=("detect", "semantic"))
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="evaluation")
    args = parser.parse_args()
    require_gpu(args.device)
    data = dataset_config(args.data, args.task, split=args.split)
    resolved = ROOT / "results" / "configs" / f"evaluate_{args.task}.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    model = YOLO(args.model, task=args.task)
    if Path(args.model).suffix == ".pt":
        require_one_channel(model)
    metrics = model.val(
        data=str(resolved),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=2,
        workers=0,
        project=str(ROOT / "results" / "evaluations" / args.task),
        name=args.name,
        save_json=True,
    )
    write_json(metrics.save_dir / "metrics.json", metrics.results_dict)
    print(f"Saved metrics: {metrics.save_dir}")


if __name__ == "__main__":
    main()
