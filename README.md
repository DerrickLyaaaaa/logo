# LoGo-Fuse

Training-free 3D point cloud OOD detection codebase (LoGo-Fuse).

## 1. What is included

This repository contains:
- Core code: `main_logofuse.py`, `ood_methods/`, `models/`, `utils/`
- Dataset configs and label/split metadata: `data/*.yaml`, `data/SR`, `data/SN`, `data/MN`, `data/templates.json`
- Repro scripts (mainly for ScanObjectNN): `tools/*.sh`

This repository intentionally **does not include**:
- Large model checkpoint (`*.pt`)
- Large dataset binaries (`*.dat`)
- Runtime cache/log outputs

## 2. Environment setup

Recommended: Python 3.10+ and CUDA-enabled PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install PyTorch separately (example CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 3. Required files

### 3.1 Checkpoint

Put ULIP checkpoint at repository root (or pass custom path):

- default expected filename:
  `ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt`

or run with:

```bash
export TEST_CKPT_ADDR=/abs/path/to/your_checkpoint.pt
```

### 3.2 Dataset `.dat` files

Expected locations:

- `data/scanobjectnn15_normal_resampled/scanobjectnn15_train_2048pts_fps.dat`
- `data/scanobjectnn15_normal_resampled/scanobjectnn15_test_2048pts_fps.dat`
- `data/shapenetcore54_normal_resampled/shapenetcore54_train_4096pts_fps.dat`
- `data/shapenetcore54_normal_resampled/shapenetcore54_test_4096pts_fps.dat`
- `data/modelnet40_normal_resampled/modelnet40_train_8192pts_fps.dat`
- `data/modelnet40_normal_resampled/modelnet40_test_8192pts_fps.dat`

Notes:
- `*_normal_resampled` means normalized + fixed-point sampling preprocessed data.
- `.txt` / `shape_names` metadata files are already included in this repo.

## 4. Quick start

### 4.1 ScanObjectNN (recommended reproducibility path)

Zero-shot SR1/SR2/SR3:

```bash
bash tools/run_zeroshot_sr123.sh
```

Full-shot SR1/SR2/SR3:

```bash
bash tools/run_fullshot_sr123.sh
```

K_neg sweep (full-shot):

```bash
bash tools/run_kneg_sweep_fullshot.sh 1 15
```

### 4.2 ShapeNet / ModelNet (direct command examples)

ShapeNet zero-shot (SN1):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ShapeNetCore54 --dataset_split SN1 \
  --npoints 4096 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 0
```

ShapeNet full-shot (SN1):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ShapeNetCore54 --dataset_split SN1 \
  --npoints 4096 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 999999
```

ModelNet zero-shot (MN1):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ModelNet40 --dataset_split MN1 \
  --npoints 8192 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 0
```

ModelNet full-shot (MN1):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ModelNet40 --dataset_split MN1 \
  --npoints 8192 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 999999
```

## 5. Outputs

- Run logs: `logs_*`
- TSV summaries: inside each `logs_*` directory
- Feature cache: `outputs/feature_cache_logofuse`

## 6. Repro tips

- Keep `--fewshot_seed 0` and fixed config for strict reproducibility.
- If metrics drift, clear `outputs/feature_cache_logofuse` and rerun.
- For paper tables, record the exact log directory name (`logs_*_YYYYmmdd_HHMMSS`).
