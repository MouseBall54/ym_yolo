Store real data here; images and annotations are excluded from Git. Images must be one-channel uint8 in 0..255.

The real-data folders have already been created:
- `detection/images/{train,val}/` and `detection/labels/{train,val}/`.
- `segmentation/images/{train,val}/` and `segmentation/masks/{train,val}/`.

The default training YAMLs point to these folders. Replace placeholder class names in
`configs/datasets/detection.yaml` and `configs/datasets/semantic.yaml` to match the real annotation IDs.
Synthetic examples are stored separately under `example_detect/` and `example_semantic/`.

Detection: `NAME/images/{train,val}/*` and `NAME/labels/{train,val}/*.txt` (YOLO boxes).
Semantic: `NAME/images/{train,val}/*` and `NAME/masks/{train,val}/*.png` (integer class IDs).

Images and labels/masks must share the same relative stem. Semantic masks must have the original image dimensions.
The example semantic mask uses 0=background, 1=target, 255=ignore. Do not normalize class IDs.
