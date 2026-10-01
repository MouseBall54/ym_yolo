Training: `training/{detect,semantic}/RUN/` (best.pt, last.pt, results.csv, plots, channel_audit.json).
Prediction: `predictions/{detect,semantic}/` (JSON, overlays, semantic class-ID PNGs).
Evaluation: `evaluations/{detect,semantic}/RUN/` (metrics.json and upstream reports).
Exports: `exports/{detect,semantic}/RUN/` (ONNX/engine and preprocessing manifest).
Runtime data configurations: `configs/` (resolved absolute dataset paths).
