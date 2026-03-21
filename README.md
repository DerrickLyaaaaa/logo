# LoGo-Fuse

LoGo-Fuse is a training-free 3D point cloud OOD detection codebase built on frozen ULIP-2 features.

This public repository keeps the cleaned mainline only:
- local graph propagation
- global prototype refinement
- negative prototype bank
- scalar local/global fusion

Historical experimental branches such as `gsp_ot`, old OT/OODD scoring, geometry-side fusion branches, and legacy few-shot prototype clustering variants are not part of the intended public workflow.

## Repository Contents

Included:
- `main_logofuse.py`
- `ood_methods/`, `models/`, `utils/`
- `data/` metadata and dataset configs
- `tools/` helper scripts

Not included in GitHub:
- checkpoints (`*.pt`, `*.pth`)
- dataset binaries (`*.dat`)
- runtime caches and logs (`outputs/`, `logs_*`)

## Environment

Python 3.10+ is recommended.

CUDA is required for ULIP-2 evaluation in this repository.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install PyTorch separately for your CUDA version. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Optional acceleration for PointBERT KNN:

```bash
pip install --no-build-isolation git+https://github.com/unlimblue/KNN_CUDA.git
```

If `knn_cuda` is not installed, LoGo-Fuse falls back to the built-in torch KNN implementation.

## Required Assets

### Checkpoint

Download the ULIP-2 checkpoint:
- [ULIP-2 checkpoint](https://drive.google.com/open?id=1Kaf6etUyhA4b9NohYmbsJpVT-bQ3iA9p)

Place it at the repository root as:

```text
ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt
```

or override with:

```bash
export TEST_CKPT_ADDR=/abs/path/to/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt
```

### Dataset `.dat` bundle

Download the dat bundle:
- [LoGo-Fuse DAT bundle](https://drive.google.com/open?id=11rYY2vF2ENqgNsgOQj3FousqAQq_IAmb)

Expected paths:

```text
data/scanobjectnn15_normal_resampled/scanobjectnn15_train_2048pts_fps.dat
data/scanobjectnn15_normal_resampled/scanobjectnn15_test_2048pts_fps.dat

data/shapenetcore54_normal_resampled/shapenetcore54_train_4096pts_fps.dat
data/shapenetcore54_normal_resampled/shapenetcore54_test_4096pts_fps.dat

data/modelnet40_normal_resampled/modelnet40_train_8192pts_fps.dat
data/modelnet40_normal_resampled/modelnet40_test_8192pts_fps.dat
```

The repository already includes the required lightweight metadata such as:
- `data/templates.json`
- `data/labels.json`
- `data/SR`, `data/SN`, `data/MN`
- `*_train.txt`, `*_test.txt`, `*_shape_names.txt`

## Benchmark Tracks

This repository currently documents three evaluation tracks.

### 1. Synthetic: ShapeNetCore54

Zero-shot example (`SN1`):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ShapeNetCore54 --dataset_split SN1 \
  --npoints 4096 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 0
```

Full-shot example (`SN1`):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ShapeNetCore54 --dataset_split SN1 \
  --npoints 4096 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 999999
```

### 2. Synthetic-to-Real: ModelNet support -> ScanObjectNN test

Officially this track is used for `SR1` and `SR2`.

The target/eval dataset remains `ScanObjectNN15`. Synthetic support is injected through a custom train `.dat` via `--scanobject_train_dat`.

The support `.dat` must already be prepared in a **ScanObject-compatible label space**.

Helper script:

```bash
bash tools/run_synth2real_sr12.sh /abs/path/to/modelnet_as_support_train.dat
```

Optional override for the ScanObject test dat:

```bash
SCANOBJECT_TEST_DAT=/abs/path/to/scanobject_test.dat \
  bash tools/run_synth2real_sr12.sh /abs/path/to/modelnet_as_support_train.dat
```

### 3. Real-to-Real: ScanObjectNN15

Zero-shot (`SR1/SR2/SR3`):

```bash
bash tools/run_zeroshot_sr123.sh
```

Full-shot (`SR1/SR2/SR3`):

```bash
bash tools/run_fullshot_sr123.sh
```

Negative-bank sweep (`K_neg`, full-shot):

```bash
bash tools/run_kneg_sweep_fullshot.sh 1 15
```

## Verified Results

These numbers were re-verified with the cleaned mainline configuration.

### ScanObjectNN15 full-shot

- `SR1`: `AUROC 0.930279`, `FPR95 0.306383`
- `SR2`: `AUROC 0.919289`, `FPR95 0.369697`
- `SR3`: `AUROC 0.859724`, `FPR95 0.530387`
- `Macro Average`: `AUROC 0.903097`, `FPR95 0.402156`

### ShapeNetCore54 full-shot

- `SN1`: `AUROC 0.913809`, `FPR95 0.464525`
- `SN2`: `AUROC 0.932384`, `FPR95 0.338373`
- `SN3`: `AUROC 0.963097`, `FPR95 0.248195`
- `Macro Average`: `AUROC 0.936430`, `FPR95 0.350364`

## Outputs

- logs: `logs_*`
- per-run summaries: `results.tsv`, `summary.tsv`
- feature cache: `outputs/feature_cache_logofuse*`

## Repro Notes

- The helper scripts prefer `./.venv/bin/python` when present.
- Full-shot support comes from the packaged train split, or from `--scanobject_train_dat` in the Synth-to-Real track.
- Zero-shot uses the same pipeline with `--shot 0`.
- The public scripts are aligned with the cleaned mainline configuration and do not rely on removed historical flags.
