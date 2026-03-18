import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import preprocess_subject


class BraTSDataset(Dataset):
    def __init__(self, data_dir, subjects, augment=False,
                 target_shape=(128, 128, 128)):
        """
        Args:
            data_dir     : path to Task01_BrainTumour/
            subjects     : list of {'image': rel_path, 'label': rel_path}
            augment      : apply training augmentations
            target_shape : spatial size after preprocessing
        """
        self.data_dir = data_dir
        self.subjects = subjects
        self.augment = augment
        self.target_shape = target_shape

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        img_path = os.path.join(self.data_dir, s['image'].lstrip('./'))
        lbl_path = os.path.join(self.data_dir, s['label'].lstrip('./'))

        image, label = preprocess_subject(img_path, lbl_path, self.target_shape)

        image = torch.from_numpy(image)                    # (4, H, W, D)
        label = torch.from_numpy(label.astype(np.int64))   # (H, W, D)

        if self.augment:
            image, label = self._augment(image, label)

        return image, label

    def _augment(self, image, label):
        """Random flips (all axes) and per-modality intensity jitter."""
        # Random flips along H / W / D  (image dims 1,2,3; label dims 0,1,2)
        for axis in range(3):
            if np.random.random() > 0.5:
                image = torch.flip(image, [axis + 1])
                label = torch.flip(label, [axis])

        # Per-modality intensity shift + scale
        for m in range(image.shape[0]):
            shift = float(np.random.uniform(-0.1, 0.1))
            scale = float(np.random.uniform(0.9, 1.1))
            image[m] = image[m] * scale + shift

        return image, label


def load_dataset_splits(data_dir, n_folds=5, seed=42):
    """Parse dataset.json and return 5-fold subject splits.

    Returns:
        list of (train_subjects, val_subjects) — one tuple per fold
    """
    with open(os.path.join(data_dir, 'dataset.json')) as f:
        meta = json.load(f)

    subjects = meta['training']   # list of {'image': ..., 'label': ...}
    n = len(subjects)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    splits = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        splits.append(
            ([subjects[i] for i in train_idx],
             [subjects[i] for i in val_idx])
        )

    return splits
