"""Train matched grayscale/RGB copies in fresh processes and measure CUDA training peaks."""

import argparse
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer
from ultralytics.nn.tasks import DetectionModel, SemanticSegmentationModel

from tools.common import ROOT, dataset_config, image_files, read_gray, write_json


def prepare_pair(data_path, task, output):
    """Copy the same pixels/annotations; RGB is exactly three identical grayscale planes."""
    data = dataset_config(data_path, task)
    source = Path(data["path"])
    digest = hashlib.sha256()
    counts = {}
    for split in ("train", "val"):
        files = image_files(source / data[split])
        counts[split] = len(files)
        for file in files:
            relative = file.relative_to(source / "images")
            gray = read_gray(file, require_uint8=True)
            annotation = source / (data.get("masks_dir", "masks") if task == "semantic" else "labels")
            annotation /= relative.with_suffix(".png" if task == "semantic" else ".txt")
            digest.update(str(relative).encode())
            digest.update(str(gray.shape).encode())
            digest.update(gray.tobytes())
            digest.update(annotation.read_bytes())
            for channels in (1, 3):
                root = output / f"{channels}ch"
                image_path = root / "images" / relative.with_suffix(".png")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if image_path.exists():
                    raise ValueError(f"Output collision; image stems must be unique within each folder: {image_path}")
                pixels = gray if channels == 1 else np.repeat(gray[..., None], 3, axis=2)
                encoded = cv2.imencode(".png", pixels)[1]
                encoded.tofile(image_path)
                decoded = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(decoded, pixels)
                label_path = (
                    root
                    / ("masks" if task == "semantic" else "labels")
                    / relative.with_suffix(".png" if task == "semantic" else ".txt")
                )
                label_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(annotation, label_path)
    for channels in (1, 3):
        paired = {**data, "path": str(output / f"{channels}ch"), "channels": channels}
        paired.update(train="images/train", val="images/val")
        if task == "semantic":
            paired["masks_dir"] = "masks"
        (output / f"{channels}ch.yaml").write_text(yaml.safe_dump(paired, sort_keys=False), encoding="utf-8")
    return {"counts": counts, "pixel_annotation_sha256": digest.hexdigest(), "source": str(source)}


def worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    trainer_class = DetectionTrainer if job["task"] == "detect" else SemanticSegmentationTrainer
    options = {
        "task": job["task"],
        "model": job["model"],
        "data": job["data"],
        "project": job["directory"],
        "name": "training",
        "device": 0,
        "epochs": job["epochs"],
        "imgsz": job["imgsz"],
        "batch": job["batch"],
        "workers": 0,
        "cache": False,
        "pretrained": False,
        "amp": job["amp"],
        "seed": 42,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "nbs": job["batch"],
        "warmup_epochs": 0.0,
        "save": False,
        "val": False,
        "plots": False,
        "exist_ok": False,
        "multi_scale": 0.0,
        "close_mosaic": 0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "bgr": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    }
    trainer = trainer_class(overrides=options)
    records = []
    report = {
        "job": job,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
    }

    def initialize(trainer):
        state = torch.load(job["initial_state"], map_location="cpu", weights_only=True)
        if job["channels"] == 1:
            state["model.0.conv.weight"] = state["model.0.conv.weight"].sum(dim=1, keepdim=True)
        trainer.model.load_state_dict(state, strict=True)
        trainer.ema.ema.load_state_dict(state, strict=True)
        report["parameters"] = sum(p.numel() for p in trainer.model.parameters())
        report["actual_amp"] = bool(trainer.amp)
        if bool(trainer.amp) != job["amp"]:
            raise RuntimeError("AMP changed from the requested setting; comparison is invalid.")
        # Restore the same RNG state after channel-dependent model construction.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()

        def capture(module, inputs):
            image = inputs[0]
            if image.shape[1] != job["channels"]:
                raise RuntimeError(f"Unexpected actual input: {image.shape}")
            if "input_shape" not in report:
                report["input_shape"] = list(image.shape)
                report["input_dtype"] = str(image.dtype)
                report["input_bytes"] = image.numel() * image.element_size()
                planes = image.detach().float().cpu().numpy()
                if job["channels"] == 3:
                    np.testing.assert_array_equal(planes[:, 0], planes[:, 1])
                    np.testing.assert_array_equal(planes[:, 0], planes[:, 2])
                report["first_batch_plane_sha256"] = hashlib.sha256(planes[:, 0].tobytes()).hexdigest()

        trainer.model.model[0].conv.register_forward_pre_hook(capture)

    def epoch_start(trainer):
        if trainer.batch_size != job["batch"]:
            raise RuntimeError("Batch changed (possibly OOM recovery); comparison is invalid.")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        report["epoch_started"] = time.perf_counter()

    def epoch_end(trainer):
        torch.cuda.synchronize()
        records.append(
            {
                "epoch": trainer.epoch + 1,
                "seconds": time.perf_counter() - report.pop("epoch_started"),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "loss": {k: float(v) for k, v in trainer.label_loss_items(trainer.tloss).items()},
            }
        )
        write_json(Path(job["directory"]) / "epochs.json", records)

    trainer.add_callback("on_train_start", initialize)
    trainer.add_callback("on_train_epoch_start", epoch_start)
    trainer.add_callback("on_train_epoch_end", epoch_end)
    trainer.train()
    report.update(epochs=records, validation={k: float(v) for k, v in trainer.metrics.items()})
    report["peak_allocated_mib"] = max(r["peak_allocated_mib"] for r in records)
    report["peak_reserved_mib"] = max(r["peak_reserved_mib"] for r in records)
    report["steady_epoch_seconds"] = statistics.median(r["seconds"] for r in records[1:])
    write_json(Path(job["directory"]) / "measurement.json", report)


def summarize(output, tasks, repeats):
    summary = {}
    for task in tasks:
        measurements = {
            c: [
                json.loads((output / task / f"repeat{r}_{c}ch" / "measurement.json").read_text(encoding="utf-8"))
                for r in range(1, repeats + 1)
            ]
            for c in (1, 3)
        }
        for one, three in zip(measurements[1], measurements[3]):
            if one["first_batch_plane_sha256"] != three["first_batch_plane_sha256"]:
                raise RuntimeError("The actual first training batch pixels/order differ between cases.")
        keys = ("peak_allocated_mib", "peak_reserved_mib", "steady_epoch_seconds", "input_bytes", "parameters")
        summary[task] = {
            f"{c}ch": {key: statistics.median(m[key] for m in measurements[c]) for key in keys} for c in (1, 3)
        }
        summary[task]["one_over_three"] = {key: summary[task]["1ch"][key] / summary[task]["3ch"][key] for key in keys}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("detect", "semantic", "both"), default="both")
    parser.add_argument("--detect-data", default="configs/datasets/example_detect.yaml")
    parser.add_argument("--semantic-data", default="configs/datasets/example_semantic.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config)
        return
    if args.epochs < 2 or args.repeats < 1 or args.batch < 1:
        parser.error("Need epochs >= 2, repeats >= 1, batch >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU.")
    output = (
        args.output or ROOT / "results/benchmarks" / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = ("detect", "semantic") if args.task == "both" else (args.task,)
    manifest = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "datasets": {}}
    for task in tasks:
        folder = output / task
        folder.mkdir()
        data_path = Path(getattr(args, f"{task}_data")).resolve()
        manifest["datasets"][task] = prepare_pair(data_path, task, folder)
        model_path = ROOT / "configs/models" / ("yolo26n-gray.yaml" if task == "detect" else "yolo26n-gray-sem.yaml")
        data = dataset_config(data_path, task)
        torch.manual_seed(42)
        model_class = DetectionModel if task == "detect" else SemanticSegmentationModel
        model = model_class(str(model_path), ch=3, nc=len(data["names"]), verbose=False)
        assert model.model[0].conv.in_channels == 3
        initial_state = folder / "initial_rgb_state.pt"
        torch.save(model.state_dict(), initial_state)
        del model
        for repeat in range(1, args.repeats + 1):
            # Alternate order to reduce systematic temperature/background-load bias.
            for channels in (1, 3) if repeat % 2 else (3, 1):
                directory = folder / f"repeat{repeat}_{channels}ch"
                directory.mkdir()
                job = {
                    "task": task,
                    "model": str(model_path),
                    "data": str(folder / f"{channels}ch.yaml"),
                    "initial_state": str(initial_state),
                    "directory": str(directory),
                    "channels": channels,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "amp": args.amp,
                }
                job_path = directory / "job.json"
                write_json(job_path, job)
                print(f"Training {task}, repeat {repeat}/{args.repeats}, {channels}ch: {directory}", flush=True)
                with (directory / "training.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [sys.executable, "-m", "tools.benchmark_channels", "--worker-config", str(job_path)],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                measured = json.loads((directory / "measurement.json").read_text(encoding="utf-8"))
                print(
                    {k: measured[k] for k in ("peak_allocated_mib", "peak_reserved_mib", "steady_epoch_seconds")},
                    flush=True,
                )
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summarize(output, tasks, args.repeats), indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
