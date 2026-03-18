# Brain Tumor Segmentation — 3D U-Net
CAP 5516 – Medical Image Computing (Spring 2026)

## Project Overview
Implement a 3D U-Net for brain tumor segmentation using the Task01_BrainTumour dataset (BraTS 2016/2017 subset, 484 training subjects). Evaluate with 5-fold cross-validation using Dice Score and 95% Hausdorff Distance on three tumor regions.

---

## Dataset

**Location:** `~/Downloads/Machine Learning/BrainMRI/Task01_BrainTumour/`

**Structure:**
```
Task01_BrainTumour/
├── imagesTr/       # 484 training MRI volumes (.nii.gz, 4D: H×W×D×4 modalities)
├── labelsTr/       # Corresponding segmentation masks (.nii.gz)
├── imagesTs/       # 266 test volumes (no labels)
└── dataset.json    # Metadata (modalities, labels, file list)
```

**MRI Modalities (from dataset.json):**
| Index | Modality |
|-------|----------|
| 0 | FLAIR |
| 1 | T1w |
| 2 | T1gd (T1 with gadolinium) |
| 3 | T2w |

**Raw Segmentation Labels:**
| Label | Meaning |
|-------|---------|
| 0 | Background |
| 1 | Edema |
| 2 | Non-enhancing tumor |
| 3 | Enhancing tumor |

**Evaluation Regions (assignment-aligned label mapping):**
| Region | Derived From Raw Labels |
|--------|------------------------|
| ET (Enhancing Tumor) | label == 3 |
| TC (Tumor Core) | label == 2 OR label == 3 |
| WT (Whole Tumor) | label == 1 OR label == 2 OR label == 3 |

Always apply this mapping when computing metrics — never evaluate on raw labels directly.

---

## Project Folder Structure
```
Task01_BrainTumour/
├── src/
│   ├── preprocess.py   # Preprocessing pipeline
│   ├── dataset.py      # PyTorch Dataset class
│   ├── unet3d.py       # 3D U-Net architecture
│   ├── train.py        # Training loop + 5-fold CV
│   ├── evaluate.py     # Metric computation (Dice + HD95)
│   ├── visualize.py    # Side-by-side MRI/GT/Pred plots
│   └── utils.py        # Shared helpers (label mapping, logging)
├── results/
│   ├── fold_1/ … fold_5/   # Checkpoints + per-fold metrics
│   └── visualizations/     # Saved prediction images
├── dataset.json
├── draft.md
└── CLAUDE.md
```

---

## Environment

**Install dependencies:**
```bash
pip install numpy scipy nibabel medpy torch torchvision torchaudio monai tqdm matplotlib
```

Key libraries:
- `nibabel` — load `.nii.gz` files
- `monai` — medical image transforms and utilities
- `medpy` — HD95 metric (`medpy.metric.binary.hd`)
- `torch` — model training

---

## Preprocessing Pipeline
Apply to each subject before feeding to the model:

1. **Load & Stack:** Read all 4 modality volumes from the single `.nii.gz` file (4D tensor) and produce shape `(4, H, W, D)`.
2. **Z-score Normalization:** Per modality, normalize using only non-zero voxels (brain mask): `(x - mean) / std`.
3. **Resample:** Resample to 1 mm isotropic voxel spacing using affine from the NIfTI header.
4. **Crop:** Crop to the bounding box of non-zero brain voxels.
5. **Resize/Pad:** Resize or pad to a fixed shape — default `128×128×128`.
6. **Augmentations (training only):** Random flips (all axes), random rotations (±15°), random intensity shifts.

---

## Model Architecture — 3D U-Net

- **Input:** `(B, 4, 128, 128, 128)` — 4 MRI modalities
- **Output:** `(B, 4, 128, 128, 128)` — 4 class logits (labels 0–3)
- **Encoder:** Repeated blocks of 3D Conv → InstanceNorm → ReLU; downsampling via strided conv (stride=2)
- **Bottleneck:** Deepest feature map
- **Decoder:** Transposed conv upsampling + skip connections from encoder
- **Final layer:** 1×1×1 conv → 4 channels; softmax at inference
- **Loss:** `DiceLoss + CrossEntropyLoss` (combined, equal weight)

---

## Training

**5-Fold Cross-Validation:**
- Split 484 subjects at subject level into 5 stratified folds.
- Each fold: train on 4 folds (~387 subjects), validate on 1 fold (~97 subjects).
- Save best checkpoint per fold based on mean validation Dice (ET+TC+WT).

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 1–2 |
| Epochs | 200–400 |
| LR Scheduler | ReduceLROnPlateau (optional) |

**Run training:**
```bash
python src/train.py --data_dir . --fold 1
```

---

## Evaluation

**Metrics per fold (for ET, TC, WT):**
- Dice Score: `2|P∩G| / (|P|+|G|)`
- 95% Hausdorff Distance:

  ```python
  from medpy.metric import binary
  hd95 = binary.hd(pred_binary, gt_binary, voxelspacing=spacing, connectivity=1)
  ```
  Use the voxel spacing from the resampled volume (1.0, 1.0, 1.0) mm.

**Run evaluation:**
```bash
python src/evaluate.py --fold 1
```

**Results table format:**
| Fold | ET Dice | TC Dice | WT Dice | ET HD95 | TC HD95 | WT HD95 |
|------|---------|---------|---------|---------|---------|---------|
| 1–5  | … | … | … | … | … | … |
| Avg  | … | … | … | … | … | … |

---

## Visualization

Generate side-by-side comparisons (MRI input / Ground Truth / Prediction) per axial slice for selected subjects:

```bash
python src/visualize.py --fold 1
```

Output saved to `results/visualizations/`. Also produce ITK-SNAP screenshots for the report.

---

## Report Requirements (Assignment)
- ITK-SNAP visualization of MRI + segmentation mask overlay
- Implementation details of the 3D U-Net (architecture diagram or description)
- 5-fold results table (Dice + HD95 for ET, TC, WT)
- Qualitative segmentation examples (MRI / GT / Prediction side-by-side)

---

## References
- Bakas et al., BraTS Challenge Papers
- Menze et al., Multimodal Brain Tumor Segmentation Benchmark
- MONAI Documentation: https://docs.monai.io
