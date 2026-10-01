# 실제 검증 기록 — 2026-09-30 (Asia/Seoul)

환경: Windows, RTX 2080 SUPER 8GB, NVIDIA 591.86, Python 3.11.13,
torch 2.11.0+cu130, torchvision 0.26.0+cu130, Ultralytics 8.4.168 (editable), TensorRT 10.13.3.9.

| 확인 | 결과 | 증거 |
|---|---|---|
| 실제 CUDA 행렬 곱 | PASS | `tools.check_environment` 실행, architecture (7,5) |
| Detection GPU 학습 1 epoch | PASS | `results/training/detect/smoke/` |
| Semantic GPU 학습 1 epoch | PASS | `results/training/semantic/smoke/` |
| 실제 학습 입력 및 첫 합성곱 | PASS | 두 run의 `channel_audit.json`: `[2,1,128,128]`, uint8, in_channels=1 |
| Detection/semantic PT 추론 | PASS | `results/predictions/detect/`, `semantic/` |
| Detection/semantic 정답 평가 | PASS | `results/evaluations/{detect,semantic}/evaluation/metrics.json` |
| Detection/semantic ONNX export | PASS | `results/exports/{detect,semantic}/smoke/best.onnx`, ONNX checker |
| Detection/semantic ONNX CPU 추론 | PASS | `results/predictions/onnx_detect/`, `onnx_semantic/` |
| Detection/semantic TensorRT FP16 build | PASS | `results/exports/{detect,semantic}/smoke/best.engine` |
| Detection/semantic TensorRT GPU 추론 | PASS | `results/predictions/engine_detect/`, `engine_semantic/` |
| Semantic mask 형상/클래스 값 | PASS | PT/ONNX/TensorRT 각 4개, 원본 `(96,128)`, ID `{0,1}` 범위 |
| 회귀 검증 | PASS | `pytest tests -q`: 7 passed |
| 사용자 코드 lint | PASS | `ruff check tools training inference exporting tests` |
| lock 일치 / 설치 script 구문 | PASS | `uv lock --check`, PowerShell parser |
| 원본 Ultralytics 변경 | 없음 | `git -C vendor/ultralytics status --short` 출력 없음 |

예제 명령은 README에 있으며, 실제 데이터 경로는 미리 생성된 `datasets/detection/`과
`datasets/segmentation/`입니다. 합성 데이터는 train 8장 / val 4장이고 검증 입력 해상도는 128px입니다.
실데이터 640px/다른 batch/클래스 및 정확도는 아직 검증하지 않았습니다.

합성 데이터 1 epoch 결과에서 detection mAP는 0이고 semantic은 배경 위주로 예측합니다.
이 모델들은 파이프라인 검증용 가중치이며 사용자 데이터에 쓸 성능이 확보된 모델이 아닙니다.
총 학습 메모리·시간이 1/3이 되었다는 측정도 하지 않았습니다.
1/3은 동일 dtype의 입력 원소 수와 첫 convolution MACs에 해당합니다.

TensorRT detection 최초 build는 약 250초, semantic은 약 118초 소요되었습니다.
Detection 최초 build 후 manifest를 기록하는 단계에서 Path JSON 직렬화 오류가 있었고,
export 코드를 문자열 변환으로 수정했습니다. 이미 정상 생성된 detection 엔진의 metadata와 실제 추론을
확인한 뒤 manifest를 복구했습니다. 수정된 export 코드로 semantic build/manifest/추론을 모두 통과했습니다.
최초 기록은 `results/validation/engine_detect.log`, semantic 기록은 `engine_semantic.log`에 남아 있습니다.

Pretrained RGB 커널 합산의 수학적 동등성은 회귀 검증에 포함했습니다.
사용자 pretrained 파일의 실제 fine-tuning, 중단 학습 resume, dynamic shape export는 이번 실행에서 확인하지 않았습니다.
TensorRT engine은 현재 GPU/Windows/TensorRT 조합에서만 실행을 확인했으며 다른 대상에서 다시 빌드해야 합니다.

## 추가 검증 — 2026-10-01: 동일 이미지셋 1ch / 3ch 실제 학습

`tools.benchmark_channels`로 detection과 semantic을 각각 1ch / 3ch, 5 epochs × 3회씩
총 12개의 독립 GPU 프로세스에서 학습 완료했습니다. 예제 train 8 / val 4, imgsz 640, batch 2, AMP.
위의 2026-09-30 기록 당시 미측정이었던 리소스를 이번에 비교했습니다.

- 픽셀/annotation 동일, 실제 첫 batch 평면 SHA-256 동일: PASS.
- batch 2 및 실제 AMP 활성화 유지: PASS.
- 최대 PyTorch 텐서 할당량 중앙값: detection 390.42 → 375.10 MiB (-3.92%), semantic 391.34 → 375.77 MiB (-3.98%).
- 캐시 포함 최대 예약량 중앙값: detection 412 → 400 MiB, semantic 418 → 406 MiB (약 -2.9%).
- 입력 float32 텐서: 9.375 → 3.125 MiB (정확히 1/3). 전체 학습 VRAM 1/3 절감은 확인되지 않음.
- 단위 검증: 10 passed. lint: PASS.

원본 결과: `results/benchmarks/channels_20261001/`.
반복별 결과와 손실/평가 차이 및 해석: [비교 보고서](docs/channel-benchmark-20261001.md).
측정은 학습 구간의 PyTorch allocator 대상이며 GPU 전체 사용량/다른 앱/context/validation을 포함하지 않습니다.
이 합성 데이터·짧은 학습으로 실데이터 정확도 유지 여부를 판단하지 않습니다.
