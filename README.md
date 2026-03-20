# Brain Tumor Segmentation — 3D U-Net

**CAP 5516 – Medical Image Computing (Spring 2026)**

3D U-Net trained on the BraTS 2016/2017 dataset (484 subjects, 4 MRI modalities) for multi-class brain tumor segmentation, evaluated with 5-fold cross-validation.

---

## Project Structure

```
BrainTumour/
├── data/
│   ├── dataset.json        # Metadata and file list (484 training subjects)
│   ├── imagesTr/           # 484 training MRI volumes (.nii.gz, 4D: H×W×D×4)
│   ├── labelsTr/           # Corresponding segmentation masks (.nii.gz)
│   └── imagesTs/           # 266 held-out test volumes (no labels)
├── src/
│   ├── utils.py            # Label mapping (ET/TC/WT), logging
│   ├── preprocess.py       # Load → normalize → resample → crop → resize
│   ├── dataset.py          # PyTorch Dataset + 5-fold split loader
│   ├── unet3d.py           # 3D U-Net architecture (~23M params)
│   ├── train.py            # Training loop with 5-fold cross-validation
│   ├── evaluate.py         # Dice + HD95 metrics per fold
│   └── visualize.py        # Side-by-side MRI / GT / Prediction plots
├── results/
│   ├── fold_1/ … fold_5/   # best_model.pth, metrics.json, train.log
│   └── visualizations/     # Saved PNG comparison images
├── reference/              # Assignment PDFs and notes
└── README.md
```

---

## Setup

### Requirements

Python 3.8+ and a CUDA-capable GPU are recommended (CPU training is possible but very slow).

# install with 
 python3 -m venv venv
 source venv/bin/activate

 pip install --upgrade pip
 pip install -r requirements.txt  


## Dataset & Label Mapping

| MRI Index | Modality |
|-----------|----------|
| 0 | FLAIR |
| 1 | T1w |
| 2 | T1gd (gadolinium-enhanced) |
| 3 | T2w |

Raw labels are remapped to three evaluation regions before computing metrics:

| Region | Raw Labels Used |
|--------|----------------|
| ET — Enhancing Tumor | `label == 3` |
| TC — Tumor Core | `label == 2` or `label == 3` |
| WT — Whole Tumor | `label == 1` or `label == 2` or `label == 3` |

---

## Preprocessing (applied per subject at load time)

1. Load 4D NIfTI → shape `(H, W, D, 4)`
2. Z-score normalize each modality using non-zero (brain) voxels
3. Resample to 1 mm isotropic voxel spacing
4. Crop to the bounding box of non-zero brain voxels
5. Resize/pad to fixed spatial size `128 × 128 × 128`
6. *(Training only)* Random flips on all axes + per-modality intensity jitter

---

## Model Architecture

- **Input:** `(B, 4, 128, 128, 128)` — batch of 4-modality volumes
- **Output:** `(B, 4, 128, 128, 128)` — 4-class logits (background + 3 tumor classes)
- Encoder: 4 stages of `[Conv3D → InstanceNorm → ReLU] × 2` + strided-conv downsampling
- Bottleneck at `8 × 8 × 8` spatial resolution
- Decoder: transposed-conv upsampling + skip connections from encoder
- Loss: `0.5 × DiceLoss + 0.5 × CrossEntropyLoss`

---

## Running the Experiments

All commands are run from the **`BrainTumour/`** directory.

### One-shot: run everything with the batch script

```bash
# Run all 5 folds (train → evaluate → visualize)
bash run_experiments.sh

# Run only fold 1
bash run_experiments.sh --fold 1

# Override epochs or batch size
bash run_experiments.sh --epochs 400 --batch_size 2
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fold` | `0` | Fold to run (1–5); `0` runs all five |
| `--epochs` | `200` | Training epochs per fold |
| `--batch_size` | `1` | Batch size |
| `--lr` | `1e-4` | Adam learning rate |
| `--workers` | `4` | DataLoader workers |
| `--n_vis` | `5` | Subjects to visualize per fold |
| `--data_dir` | `data` | Path to folder containing `dataset.json` |
| `--output_dir` | `results` | Root output directory |

The script runs three stages in order: **train → evaluate → visualize**, printing timestamps at each step. Outputs land in `results/`.

---

### Manual commands (run steps individually)

### 1. Train a single fold

```bash
python src/train.py --data_dir data --fold 1
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `.` | Path to the folder containing `dataset.json` |
| `--fold` | `1` | Fold to train (1–5); use `0` to run all folds sequentially |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `1` | Batch size (1–2 for 128³ volumes on 16 GB VRAM) |
| `--lr` | `1e-4` | Adam learning rate |
| `--output_dir` | `results` | Directory where checkpoints and logs are saved |
| `--workers` | `4` | DataLoader worker processes |

Best model per fold is saved to `results/fold_N/best_model.pth`.
Training log is written to `results/fold_N/train.log`.

### 2. Train all 5 folds

```bash
python src/train.py --data_dir data --fold 0 --epochs 200
```

### 3. Evaluate a fold

```bash
python src/evaluate.py --data_dir data --fold 1
```

Prints per-region Dice and HD95, writes `results/fold_N/metrics.json`.

### 4. Evaluate all folds and print the cross-validation table

```bash
python src/evaluate.py --data_dir data --fold 0
```

Output format:

```
=== Cross-Validation Summary ===
Fold    ET Dice   TC Dice   WT Dice   ET HD95   TC HD95   WT HD95
-------------------------------------------------------------------
1         x.xxxx    x.xxxx    x.xxxx    xx.xx     xx.xx     xx.xx
…
Avg       x.xxxx    x.xxxx    x.xxxx    xx.xx     xx.xx     xx.xx
```

Aggregated results saved to `results/cv_results.json`.

### 5. Visualize predictions

```bash
python src/visualize.py --data_dir data --fold 1 --n_subjects 5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_subjects` | `5` | How many validation subjects to plot |
| `--vis_dir` | `results/visualizations` | Output directory for PNG files |
| `--slice_idx` | *(mid-slice)* | Fixed axial slice index (optional) |

Each PNG shows 4 panels: **FLAIR \| T1w \| Ground Truth \| Prediction**.

---

## Expected Results

The table below should be populated after running all 5 folds:

| Fold | ET Dice | TC Dice | WT Dice | ET HD95 | TC HD95 | WT HD95 |
|------|---------|---------|---------|---------|---------|---------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| **Avg** | | | | | | |

---

## Hyperparameters Summary

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| LR Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Batch Size | 1 |
| Epochs | 200 |
| Input Shape | 128 × 128 × 128 |
| Base Channels | 32 |
| Loss | 0.5 × Dice + 0.5 × CE |

---

## References

- Bakas et al., *Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation*, 2018
- Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*, IEEE TMI 2015
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- MONAI Framework: https://docs.monai.io
