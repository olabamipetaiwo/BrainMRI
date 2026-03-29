import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet3d import UNet3D
from dataset import BraTSDataset, load_dataset_splits
from utils import get_logger, map_labels

try:
    from medpy.metric import binary as medpy_binary
    _HAS_MEDPY = True
except ImportError:
    _HAS_MEDPY = False
    print('WARNING: medpy not installed — HD95 will be skipped.')

from scipy.ndimage import label as cc_label


def remove_small_components(pred_bin: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
    pred_bin = pred_bin.copy()
    labeled, n = cc_label(pred_bin)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_size:
            pred_bin[labeled == i] = 0
    return pred_bin


# 
# Metric helpers
# 
def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0   # both empty → perfect agreement
    return float(2 * inter / denom)


def compute_hd95(pred_bin: np.ndarray, gt_bin: np.ndarray,
                 spacing=(1.0, 1.0, 1.0)) -> float:
    if not _HAS_MEDPY:
        return float('nan')
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('nan')   # undefined if one mask is empty
    try:
        return float(medpy_binary.hd95(
            pred_bin, gt_bin, voxelspacing=spacing, connectivity=1))
    except Exception:
        return float('nan')


def compute_sensitivity(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """TP / (TP + FN). Returns 1.0 if GT is empty (nothing to detect)."""
    tp = (pred_bin & gt_bin).sum()
    fn = (~pred_bin & gt_bin).sum()
    denom = tp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


def compute_specificity(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """TN / (TN + FP). Returns 1.0 if the entire volume is GT-positive."""
    tn = (~pred_bin & ~gt_bin).sum()
    fp = (pred_bin & ~gt_bin).sum()
    denom = tn + fp
    if denom == 0:
        return 1.0
    return float(tn / denom)


def compute_asd(pred_bin: np.ndarray, gt_bin: np.ndarray,
                spacing=(1.0, 1.0, 1.0)) -> float:
    """Average Symmetric Surface Distance (mm)."""
    if not _HAS_MEDPY:
        return float('nan')
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('nan')
    try:
        return float(medpy_binary.asd(
            pred_bin, gt_bin, voxelspacing=spacing, connectivity=1))
    except Exception:
        return float('nan')


# 
# Per-fold evaluation
# 
def evaluate_fold(data_dir, fold_idx, splits, device, args):
    fold_dir  = os.path.join(args.output_dir, f'fold_{fold_idx}')
    ckpt_path = os.path.join(fold_dir, 'best_model.pth')
    logger    = get_logger(
        f'eval_{fold_idx}',
        log_file=os.path.join(fold_dir, 'eval.log'),
    )

    if not os.path.exists(ckpt_path):
        logger.error(f'Checkpoint not found: {ckpt_path} — skip fold {fold_idx}')
        return None

    _, val_subjects = splits[fold_idx - 1]
    val_ds = BraTSDataset(data_dir, val_subjects, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = UNet3D(in_channels=4, num_classes=4, base_ch=32).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f'Loaded fold {fold_idx} checkpoint (epoch {ckpt["epoch"]})')

    bucket = {r: {'dice': [], 'hd95': [], 'sensitivity': [], 'specificity': [], 'asd': []}
              for r in ('ET', 'TC', 'WT')}

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            img = images.to(device)

            # Test-Time Augmentation: average softmax over original + 3 axis flips
            probs = torch.softmax(model(img), dim=1)
            for axis in (2, 3, 4):
                flipped = torch.flip(img, dims=[axis])
                probs  += torch.flip(torch.softmax(model(flipped), dim=1), dims=[axis])
            probs /= 4.0

            pred     = probs.argmax(dim=1)[0].cpu().numpy()   # (H,W,D)
            gt       = labels[0].numpy()

            pred_map = map_labels(pred)
            gt_map   = map_labels(gt)

            for region in ('ET', 'TC', 'WT'):
                pm = pred_map[region].astype(bool)
                pm = remove_small_components(pm, min_size=50)
                gm = gt_map[region].astype(bool)
                bucket[region]['dice'].append(compute_dice(pm, gm))
                bucket[region]['hd95'].append(compute_hd95(pm, gm))
                bucket[region]['sensitivity'].append(compute_sensitivity(pm, gm))
                bucket[region]['specificity'].append(compute_specificity(pm, gm))
                bucket[region]['asd'].append(compute_asd(pm, gm))

            if (i + 1) % 20 == 0:
                logger.info(f'  {i+1}/{len(val_loader)} subjects done')

    # Summary
    summary = {}
    header = f'{"Region":>6}  {"Dice":>8}  {"HD95":>8}  {"Sens":>8}  {"Spec":>8}  {"ASD":>8}'
    logger.info(f'\nFold {fold_idx} — {len(val_subjects)} val subjects')
    logger.info(header)
    logger.info('-' * len(header))

    for region in ('ET', 'TC', 'WT'):
        def _mean(key):
            vals = [v for v in bucket[region][key] if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float('nan')

        mean_dice = _mean('dice')
        mean_hd95 = _mean('hd95')
        mean_sens = _mean('sensitivity')
        mean_spec = _mean('specificity')
        mean_asd  = _mean('asd')
        summary[region] = {
            'dice': mean_dice, 'hd95': mean_hd95,
            'sensitivity': mean_sens, 'specificity': mean_spec, 'asd': mean_asd,
        }
        logger.info(f'{region:>6}  {mean_dice:>8.4f}  {mean_hd95:>8.2f}'
                    f'  {mean_sens:>8.4f}  {mean_spec:>8.4f}  {mean_asd:>8.2f}')

    out_path = os.path.join(fold_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'Metrics saved → {out_path}')

    return summary


# 
# Entry point
# 
def parse_args():
    p = argparse.ArgumentParser(description='Evaluate 3D U-Net on BraTS folds')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--fold',       type=int, default=1,
                   help='Fold index (1-5); 0 = all folds')
    p.add_argument('--output_dir', default='results')
    return p.parse_args()


def print_cv_table(all_results):
    cols = ('ET', 'TC', 'WT')
    metrics = ('dice', 'hd95', 'sensitivity', 'specificity', 'asd')
    fmt     = {'dice': '.4f', 'hd95': '.2f', 'sensitivity': '.4f', 'specificity': '.4f', 'asd': '.2f'}
    labels  = {'dice': 'Dice', 'hd95': 'HD95', 'sensitivity': 'Sens', 'specificity': 'Spec', 'asd': 'ASD'}

    header = f'{"Fold":<5}'
    for m in metrics:
        for c in cols:
            header += f'  {c+" "+labels[m]:>12}'
    print('\n=== Cross-Validation Summary ===')
    print(header)
    print('-' * len(header))

    agg = {r: {m: [] for m in metrics} for r in cols}
    for fold, res in sorted(all_results.items()):
        row = f'{fold:<5}'
        for m in metrics:
            for r in cols:
                row += f'  {res[r][m]:>12{fmt[m]}}'
                agg[r][m].append(res[r][m])
        print(row)

    print('-' * len(header))
    row = f'{"Avg":<5}'
    for m in metrics:
        for r in cols:
            vals = [v for v in agg[r][m] if not np.isnan(v)]
            row += f'  {np.mean(vals):>12{fmt[m]}}'
    print(row)


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = load_dataset_splits(args.data_dir)

    folds = range(1, 6) if args.fold == 0 else [args.fold]
    all_results = {}
    for fold in folds:
        result = evaluate_fold(args.data_dir, fold, splits, device, args)
        if result:
            all_results[fold] = result

    if len(all_results) > 1:
        print_cv_table(all_results)

        agg_path = os.path.join(args.output_dir, 'cv_results.json')
        with open(agg_path, 'w') as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
        print(f'\nAggregated results saved → {agg_path}')


if __name__ == '__main__':
    main()
