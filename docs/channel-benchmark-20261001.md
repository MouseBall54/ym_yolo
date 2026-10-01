# 동일 이미지셋 1채널 / 3채널 실제 학습 비교 — 2026-10-01

입력 텐서는 정확히 1/3이지만, 이번 조건에서 최대 학습 텐서 메모리는 약 4% 줄었습니다.
전체 학습 VRAM이 1/3이 된다는 주장은 이 실험 결과와 맞지 않습니다.

![실제 측정 비교](benchmarks/channels_20261001.png)

## 실험 조건

- Windows / RTX 2080 SUPER 8GB / NVIDIA 591.86.
- Python 3.11.13 / torch 2.11.0+cu130 / Ultralytics 8.4.168.
- YOLO26n detection, YOLO26n semantic segmentation을 각각 비교.
- 작업별 기존 합성 예제 train 8장 / val 4장. 사용자 실제 데이터 폴더는 아직 비어 있음.
- imgsz 640, batch 2, AMP 실제 활성화, AdamW, seed 42, workers 0, cache 없음.
- 각 경우 5 epochs × 3회, 총 12개의 독립 프로세스에서 순차 학습.
- 1채널 파일의 픽셀을 그대로 세 번 복제하여 3채널 PNG 생성. 라벨/semantic mask는 바이트 그대로 복사.
  PNG 재해독 후 원본 픽셀 일치, 실제 첫 학습 batch의 평면 SHA-256 일치를 검사.
- 색상/기하/혼합 증강, multi-scale, warmup은 모두 끔. 반복마다 1ch/3ch 실행 순서를 번갈아 배치.
- 작업마다 공통 RGB 초기 state를 사용. 1ch 첫 convolution은 RGB 커널 합산, 나머지 층과 EMA 초기값은 동일.
- 학습 epoch 시작/끝에 CUDA 동기화. epoch마다 peak 통계를 초기화하고 validation 전에 측정 종료.
  학습 메모리는 5 epochs 중 최대값, 시간은 초기 준비 비용이 큰 첫 epoch를 제외한 2~5 epoch 중앙값.
  아래 표는 3회 실행 중앙값. validation, checkpoint 저장, 프로세스 시작 시간은 제외.

## 측정 결과

| 항목 | Detection 3ch | Detection 1ch | 감소 | Semantic 3ch | Semantic 1ch | 감소 |
|---|---:|---:|---:|---:|---:|---:|
| 최대 텐서 할당량 (MiB) | 390.42 | 375.10 | 3.92% | 391.34 | 375.77 | 3.98% |
| 캐시 포함 최대 예약량 (MiB) | 412.00 | 400.00 | 2.91% | 418.00 | 406.00 | 2.87% |
| 학습 epoch 시간 (초) | 0.3529 | 0.3350 | 5.07% | 0.7998 | 0.7778 | 2.75% |
| 첫 convolution 입력 float32 (MiB) | 9.375 | 3.125 | 66.67% | 9.375 | 3.125 | 66.67% |
| 학습 모델 파라미터 | 2,504,190 | 2,503,902 | 0.0115% | 1,630,692 | 1,630,404 | 0.0177% |

실제 입력 형상은 `[2,3,640,640]` / `[2,1,640,640]`이며 두 경우 모두 첫 convolution 직전 float32입니다.
AMP 내부 convolution 연산의 정밀도와 입력 텐서 dtype은 별개입니다.

최대 텐서 할당량의 3회 범위는 detection 1ch **375.10~385.66 MiB**, 3ch **390.42~404.42 MiB**,
semantic 1ch **375.77~388.14 MiB**, 3ch **391.34~403.71 MiB**입니다.
실행별 메모리·시간 변동이 있으므로 작은 시간 차이를 모든 데이터/장비의 고정 속도 개선으로 해석하면 안 됩니다.

`max_memory_allocated()`는 PyTorch 텐서가 점유한 메모리,
`max_memory_reserved()`는 PyTorch 캐시 allocator가 관리하는 메모리의 최대값입니다.
GPU 전체 사용량, CUDA context, allocator 밖의 라이브러리 메모리, 다른 앱 VRAM까지 합친 값이 아닙니다.
정의: [PyTorch allocated](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html),
[PyTorch reserved](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html).

첫 층의 입력 채널만 줄여도 첫 층 출력부터는 같은 feature channel 구조를 사용합니다.
이번 nano 모델에서 줄어든 파라미터는 288개뿐입니다. 이후 activation, gradient, optimizer state의 대부분은 유지됩니다.
따라서 입력만 1/3이라는 사실로 총 메모리·시간도 1/3이라고 추론할 수 없습니다.

## 손실과 평가 지표

| 작업 / 지표 | 3ch | 1ch |
|---|---:|---:|
| Detection 마지막 box loss | 1.88029 | 2.21736 |
| Detection 마지막 cls loss | 12.19173 | 12.94770 |
| Detection 마지막 l1 loss | 0.02703 | 0.03367 |
| Detection val mAP50-95 | 0.001777 | 0.001677 |
| Semantic 마지막 CE loss | 0.31396 | 0.30437 |
| Semantic 마지막 Dice loss | 0.37457 | 0.36939 |
| Semantic 마지막 aux loss | 0.12519 | 0.12007 |
| Semantic val mIoU | 0.463542 | 0.463542 |
| Semantic val pixel accuracy | 0.927083 | 0.927083 |

같은 채널 조건의 손실과 평가 지표는 3회 반복에서 기록 정밀도 내 동일했습니다.
하지만 1ch와 3ch의 학습 손실까지 같지는 않습니다. 커널 합산은 복제 grayscale 입력에 대한 **초기 응답**을 보존합니다.
이후 RGB 첫 층은 세 커널을 각각 업데이트하고 1ch는 한 커널을 업데이트하므로, 같은 AdamW 설정에서도
유효 커널의 업데이트가 달라집니다. AMP 부동소수점 연산의 차이도 있을 수 있습니다.

이 예제는 작은 합성 데이터로 짧게 학습한 모델입니다. Detection 지표는 매우 낮고,
semantic은 두 경우 모두 foreground IoU=0으로 배경 위주 예측입니다.
같은 mIoU만으로 실제 정확도 유지 여부를 판단할 수 없습니다. 실데이터, 충분한 epochs와 여러 seed로 평가해야 합니다.

## 현재 RGB 입력 정책

프로젝트의 `training.train`, `inference.predict`, `tools.prepare_dataset`은 RGB 파일을 자동 변환하지 않습니다.
`tools.common.read_gray()`의 검사에서 오류를 냅니다. 현재는 0~255 값의 단일 채널 입력을 명시적으로 요구합니다.
Ultralytics 원본에는 `channels: 1`이면 grayscale로 읽는 기능이 있지만, 사용자 학습 코드의 입력 검사가 먼저 실행됩니다.

이번 벤치마크의 3ch 케이스만 비교를 위해 native trainer를 직접 사용합니다.
따라서 벤치마크가 3ch를 학습했다고 일반 사용자 학습 경로가 RGB를 허용하게 된 것은 아닙니다.
RGB 세 평면이 같은 파일과 실제 컬러 RGB 파일도 구분해야 합니다. 후자는 grayscale 변환 시 색상 정보를 잃습니다.

## 재현 명령과 기록

```powershell
uv run --no-sync python -m tools.benchmark_channels --epochs 5 --repeats 3 --imgsz 640 --batch 2

# 실제 데이터 저장 후 같은 절차로 비교:
uv run --no-sync python -m tools.benchmark_channels --detect-data configs/datasets/detection.yaml --semantic-data configs/datasets/semantic.yaml

# Detection만 비교 / AMP를 끄고 별도 실험:
uv run --no-sync python -m tools.benchmark_channels --task detect --no-amp
```

매 실행은 `results/benchmarks/<실행시각>/`에 새 비교 데이터셋, 공통 초기 state, 실행별 log·epoch·measurement JSON,
`manifest.json`, `summary.json`을 저장합니다. 원본 데이터는 덮어쓰지 않습니다.
OOM으로 batch가 바뀌거나 AMP가 요청 설정에서 변경되거나 실제 첫 batch 평면이 다르면 비교를 실패시킵니다.

이번 실행 원본: `results/benchmarks/channels_20261001/`.
작은 결과 파일은 문서와 함께 보관했습니다:

- [요약 JSON](benchmarks/channels_20261001_summary.json)
- [설정·데이터 fingerprint](benchmarks/channels_20261001_manifest.json)
- [12회 실행 결과 CSV](benchmarks/channels_20261001_runs.csv)
- [epoch별 메모리·시간·손실 CSV](benchmarks/channels_20261001_epochs.csv)
- [테스트 코드](../tools/benchmark_channels.py)

`pytest tests -q`: 10 passed. 여기에는 실제 비교셋의 픽셀/라벨/mask 불변 검사와 서로 다른 실제 batch를
결과 요약이 거부하는 검사도 포함합니다. 단위 검증과 별도로 위의 GPU 학습 12회를 모두 완료했습니다.
