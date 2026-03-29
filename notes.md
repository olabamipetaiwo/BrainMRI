# Notes — Brain Tumor Segmentation

## Metrics Reminder
- **Dice Score**: higher is better (0–1, perfect = 1.0)
- **HD95**: lower is better (mm, perfect = 0.0)

---

## Ways to Reduce HD95

### 1. Add a Boundary/Surface Loss *(highest impact)*
HD95 is a boundary metric — Dice+CE loss doesn't optimize for it directly.
Add a surface loss that penalizes predictions far from the GT boundary.

```python
from monai.losses import HausdorffDTLoss
# Combined loss: 0.4*dice + 0.4*ce + 0.2*boundary
```

### 2. Post-processing: Remove Small Disconnected Components
Spurious small blobs far from the main tumor mass inflate HD95 significantly.
Remove connected components below a size threshold (e.g. <50 voxels).
Can drop HD95 by 1–2 mm with zero retraining — implement in `evaluate.py`.

### 3. More Training Epochs
Boundaries sharpen as training continues.
Current run: 15 epochs. Target: 20–30 epochs for further improvement.

### 4. Test-Time Augmentation (TTA)
Average predictions over flipped versions of the input at inference.
Smooths boundary uncertainty. Implement in `evaluate.py`.

```python
# predict on original + 3 axis flips, average softmax outputs
```

### 5. Increase Input Resolution
Current: 128³. Bumping to 160³ or 192³ gives finer boundary detail.
Trade-off: higher GPU memory usage.

### 6. Deeper Decoder / More Skip Connection Channels
Increase `base_ch` from 32 → 48 in `unet3d.py`.
Helps preserve fine-grained boundary detail during upsampling.

---

## Quickest Wins (no retraining needed)
- **#2** — Post-processing connected components
- **#4** — Test-time augmentation

---

## Results Progression (for report)

### Run 1 — Baseline (10 epochs, no post-processing, no TTA)
| Fold | ET Dice | TC Dice | WT Dice | ET HD95 | TC HD95 | WT HD95 |
|------|---------|---------|---------|---------|---------|---------|
| 1    | 0.7240  | 0.7910  | 0.8780  | 5.14    | 5.23    | 4.37    |
| 2    | 0.7510  | 0.8080  | 0.8710  | 4.83    | 6.14    | 4.92    |
| 3    | 0.7430  | 0.8150  | 0.8830  | 4.02    | 4.61    | 3.89    |
| 4    | 0.6980  | 0.7930  | 0.8760  | 5.41    | 5.48    | 4.15    |
| 5    | 0.7190  | 0.7870  | 0.8630  | 5.64    | 5.82    | 5.03    |
| **Avg** | **0.7270** | **0.7988** | **0.8742** | **4.99** | **5.46** | **4.47** |

Techniques: 3D U-Net, DiceLoss + weighted CrossEntropyLoss (ET upweighted 3x), Adam lr=1e-4, AMP, grad clipping.

---

### Run 2 — More Epochs (15 epochs, no post-processing, no TTA)
| Fold | ET Dice | TC Dice | WT Dice | ET HD95 | TC HD95 | WT HD95 |
|------|---------|---------|---------|---------|---------|---------|
| 1    | 0.7131  | 0.7713  | 0.8838  | 3.08    | 4.35    | 3.79    |
| 2    | 0.7645  | 0.7976  | 0.8821  | 2.88    | 4.36    | 3.70    |
| 3    | 0.7304  | 0.7830  | 0.8863  | 4.50    | 4.56    | 4.20    |
| 4    | 0.7117  | 0.7796  | 0.8857  | 5.74    | 6.09    | 4.89    |
| 5    | 0.7319  | 0.7806  | 0.8798  | 2.82    | 3.95    | 3.74    |
| **Avg** | **0.7303** | **0.7824** | **0.8835** | **3.80** | **4.66** | **4.06** |

Changes from Run 1: increased epochs 10 → 15. HD95 improved significantly (ET: 4.99 → 3.80 mm). WT Dice improved. TC Dice dipped slightly.

---

### Run 3 — TTA + Post-processing (15 epochs + TTA + remove small components)
| Fold | ET Dice | TC Dice | WT Dice | ET HD95 | TC HD95 | WT HD95 |
|------|---------|---------|---------|---------|---------|---------|
| 1    | 0.7287  | 0.7768  | 0.8871  | 3.07    | 4.15    | 3.82    |
| 2    | 0.7715  | 0.8030  | 0.8870  | 3.03    | 4.53    | 3.71    |
| 3    | 0.7375  | 0.7875  | 0.8881  | 4.39    | 4.54    | 4.55    |
| 4    | 0.7064  | 0.7849  | 0.8917  | 5.85    | 5.47    | 4.27    |
| 5    | 0.7390  | 0.7859  | 0.8823  | 2.72    | 4.00    | 3.82    |
| **Avg** | **0.7366** | **0.7876** | **0.8872** | **3.81** | **4.54** | **4.04** |

Changes from Run 2: added TTA (average softmax over original + 3 axis flips) and post-processing (remove connected components <50 voxels for all regions). No retraining.

---

## Implemented Improvements

### Post-processing: Remove Small Connected Components *(implemented in `evaluate.py`)*
After argmax prediction, any connected component with fewer than 50 voxels is removed
using `scipy.ndimage.label`. Applied to all three regions (ET, TC, WT).
Rationale: small spurious blobs far from the main tumor mass inflate HD95 significantly
because HD95 measures the worst-case boundary distance.

### Test-Time Augmentation — TTA *(implemented in `evaluate.py`)*
At inference, we predict on the original volume and 3 axis-flipped versions (flip along
H, W, D axes separately), then average the 4 softmax probability maps before taking argmax.
This smooths out boundary uncertainty and reduces prediction noise without any retraining.

```python
probs = softmax(model(img))
for axis in (2, 3, 4):
    flipped = flip(img, axis)
    probs  += flip(softmax(model(flipped)), axis)
probs /= 4.0
pred = argmax(probs)
```

### Training: 15 Epochs (up from 10)
Retraining all 5 folds with 15 epochs improved mean val Dice from 0.7585 → 0.7962.
HD95 also improved on most folds (ET: 4.99 → 3.80 mm avg), though folds 3 & 4 regressed
slightly on HD95.
