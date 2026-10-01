"""Train a native one-channel YOLO26 detection or semantic model from a user YAML."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer

from tools.common import ROOT, dataset_config, load_yaml, require_gpu, resolve_path, write_json


def audit_input(trainer):
    """Check the actual training batch and first convolution, then record channel savings."""
    if trainer.model.model[0].conv.in_channels != 1:
        raise ValueError("Training model must have one input channel.")
    batch = next(iter(trainer.train_loader))
    image = batch["img"]
    if image.ndim != 4 or image.shape[1] != 1 or image.min() < 0 or image.max() > 255:
        raise ValueError(f"Expected training input [B,1,H,W] in 0..255, got {tuple(image.shape)}")
    report = {
        "batch_shape": list(image.shape),
        "batch_dtype": str(image.dtype),
        "input_elements": image.numel(),
        "replicated_rgb_elements": image.numel() * 3,
        "input_element_ratio": 1 / 3,
        "first_conv_channels": trainer.model.model[0].conv.in_channels,
        "gpu": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else "cpu",
        "note": "Input storage and first-convolution MACs are reduced, not total model cost by 3x.",
    }
    write_json(trainer.save_dir / "channel_audit.json", report)
    print(json.dumps(report, indent=2))


def transfer_rgb_stem(trainer):
    """Preserve the RGB response to replicated grayscale when initializing from a local checkpoint."""
    source_model = YOLO(str(trainer.args.pretrained)).model
    source = source_model.model[0].conv.weight
    target = trainer.model.model[0].conv.weight
    if source.shape[1] == 3 and target.shape[1] == 1 and source.shape[0] == target.shape[0]:
        with torch.no_grad():
            gray_weight = source.sum(dim=1, keepdim=True)
            target.copy_(gray_weight)
            trainer.ema.ema.model[0].conv.weight.copy_(gray_weight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Dataset YAML override, relative to the current working directory")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--name")
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path, help="Resume from last.pt with its stored training arguments")
    args = parser.parse_args()
    config, config_path = load_yaml(args.config)
    if args.resume and not args.data:
        config["data"] = YOLO(str(args.resume.resolve())).model.args["data"]
    if args.data:
        config["data"] = str(Path(args.data).resolve())
    task = config["task"]
    if task not in {"detect", "semantic"}:
        raise ValueError("Use detect or semantic (pixel masks), not instance segment.")
    for key in ("epochs", "imgsz", "name", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    require_gpu(config.get("device", 0))
    config["model"] = str(resolve_path(config["model"], config_path.parent))
    config["data"] = str(resolve_path(config["data"], config_path.parent))
    data = dataset_config(config["data"], task)
    project = resolve_path(config.pop("project", f"results/training/{task}"))
    resolved_data = ROOT / "results" / "configs" / f"{config_path.stem}_data.yaml"
    resolved_data.parent.mkdir(parents=True, exist_ok=True)
    resolved_data.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config["data"] = str(resolved_data)
    config["project"] = str(project)
    weights = config.pop("weights", None)
    if args.resume:
        if any(getattr(args, key) is not None for key in ("epochs", "imgsz", "name")):
            raise ValueError("Resume uses stored epochs/imgsz/name; use a fresh run to change them.")
        config = {"resume": str(args.resume.resolve()), "device": config.get("device", 0)}
    elif weights:
        config["pretrained"] = str(resolve_path(weights))
    trainer_class = DetectionTrainer if task == "detect" else SemanticSegmentationTrainer
    # Native trainers construct the very first model with ch=data['channels']; YOLO(yaml) alone defaults to RGB.
    trainer = trainer_class(overrides=config)
    if weights and not args.resume:
        trainer.add_callback("on_train_start", transfer_rgb_stem)
    trainer.add_callback("on_train_start", audit_input)
    trainer.train()


if __name__ == "__main__":
    main()
