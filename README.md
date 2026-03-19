# LoGo-Fuse

Training-free 3D point cloud OOD detection codebase (LoGo-Fuse).

## 1) Repository contents

Included:
- Core code: `main_logofuse.py`, `ood_methods/`, `models/`, `utils/`
- Dataset configs and metadata: `data/*.yaml`, `data/SR`, `data/SN`, `data/MN`, `data/templates.json`
- Repro scripts (ScanObjectNN): `tools/*.sh`

Not included in GitHub (too large):
- Checkpoint (`*.pt`)
- Dataset binaries (`*.dat`)
- Runtime outputs (`outputs/`, logs)

## 2) Environment setup

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

## 3) Download required assets from Google Drive

### 3.1 Checkpoint (Google Drive)

Download:
- [ULIP-2 checkpoint](https://drive.google.com/open?id=1Kaf6etUyhA4b9NohYmbsJpVT-bQ3iA9p)

Place it at repository root with this exact filename:

```text
ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt
```

You can also override path via env var:

```bash
export TEST_CKPT_ADDR=/abs/path/to/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt
```

### 3.2 Dataset `.dat` files (Google Drive)

Download:
- [LoGo-Fuse DAT bundle](https://drive.google.com/open?id=11rYY2vF2ENqgNsgOQj3FousqAQq_IAmb)

After download, make sure these files exist at the following paths:

```text
data/scanobjectnn15_normal_resampled/scanobjectnn15_train_2048pts_fps.dat
data/scanobjectnn15_normal_resampled/scanobjectnn15_test_2048pts_fps.dat

data/shapenetcore54_normal_resampled/shapenetcore54_train_4096pts_fps.dat
data/shapenetcore54_normal_resampled/shapenetcore54_test_4096pts_fps.dat

data/modelnet40_normal_resampled/modelnet40_train_8192pts_fps.dat
data/modelnet40_normal_resampled/modelnet40_test_8192pts_fps.dat
```

Notes:
- `*_normal_resampled` = normalized + fixed-point preprocessed point clouds.
- This repo already includes required text metadata (`*_train.txt`, `*_test.txt`, `*_shape_names.txt`).

## 4) Benchmark tracks

This repository is organized into three benchmark tracks:

1. **Synthetic**: ShapeNetCore54 in-domain (SN1/SN2/SN3).
2. **Synthetic-to-Real**: ModelNet support (synthetic) + ScanObjectNN test (real).
3. **Real-to-Real**: ScanObjectNN in-domain (SR1/SR2/SR3).

### 4.1 Synthetic (ShapeNetCore54)

Zero-shot (SN1):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ShapeNetCore54 --dataset_split SN1 \
  --npoints 4096 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 0
```

Full-shot (SN1):

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ShapeNetCore54 --dataset_split SN1 \
  --npoints 4096 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 999999
```

### 4.2 Synthetic-to-Real (ModelNet support -> ScanObjectNN test)

This track must satisfy:

- **Support/Train source = ModelNet (synthetic)**
- **Test target = ScanObjectNN (real)**
- Do **not** switch test to ModelNet.

Use `dataset_name=ScanObjectNN15` for evaluation target, and inject synthetic support via custom train `.dat`:

```bash
python main_logofuse.py \
  --model ULIP_PointBERT --method logofuse --evaluate_3d \
  --dataset_name ScanObjectNN15 --dataset_split SR1 \
  --npoints 2048 --validate_dataset_prompt shapenet_64 \
  --test_ckpt_addr "${TEST_CKPT_ADDR}" --shot 999999 \
  --scanobject_train_dat /abs/path/to/modelnet_as_support_train.dat
```

Important:

- `--scanobject_train_dat` must be a **support dat prepared for ScanObject-compatible label space**.
- Test remains ScanObject default test split (or optional `--scanobject_test_dat` if explicitly overridden).

### 4.3 Real-to-Real (ScanObjectNN)

Zero-shot (SR1/SR2/SR3):

```bash
bash tools/run_zeroshot_sr123.sh
```

Full-shot (SR1/SR2/SR3):

```bash
bash tools/run_fullshot_sr123.sh
```

K_neg sweep (full-shot):

```bash
bash tools/run_kneg_sweep_fullshot.sh 1 15
```

## 5) Outputs

- Run logs: `logs_*`
- Summary TSV: under each `logs_*` folder
- Feature cache: `outputs/feature_cache_logofuse`

## 6) Repro tips

- Use fixed seeds and solver for paper runs:
  - `--seed 0`
  - `--fewshot_seed 0`
  - `--fewshot_weight_solver map`
- Use `--rebuild_feature_cache` to force cache rebuild each run (recommended for strict reproducibility).
- The provided `tools/run_*` scripts already include fixed seed + map solver + cache rebuild.
- Keep each run directory name (`logs_*_YYYYmmdd_HHMMSS`) for paper tables.
