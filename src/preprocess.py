import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def load_volume(path):
    """Load a NIfTI file.

    Returns:
        data     : float32 ndarray (H, W, D, 4) for 4D images or (H, W, D)
        affine   : 4×4 affine matrix
        spacing  : voxel spacing (sx, sy, sz) in mm
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    spacing = tuple(np.abs(np.diag(affine)[:3]).tolist())
    return data, affine, spacing


def zscore_normalize(volume):
    """Per-modality Z-score using only non-zero (brain) voxels.

    Accepts (H, W, D, 4) and returns same shape.
    """
    out = np.zeros_like(volume)
    n_mod = volume.shape[-1]
    for m in range(n_mod):
        mod = volume[..., m]
        mask = mod != 0
        if mask.sum() == 0:
            continue
        mean = mod[mask].mean()
        std = mod[mask].std()
        if std < 1e-8:
            std = 1.0
        out[..., m] = (mod - mean) / std
    return out


def resample_volume(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0),
                    is_label=False):
    """Resample to target voxel spacing using affine-derived scale factors.

    Handles both (H,W,D,4) image volumes and (H,W,D) label volumes.
    """
    zoom_factors = [cs / ts for cs, ts in zip(current_spacing, target_spacing)]
    if volume.ndim == 4:
        zoom_factors = zoom_factors + [1.0]   # keep modality axis untouched
    order = 0 if is_label else 1
    return zoom(volume, zoom_factors, order=order, prefilter=(order > 1))


def crop_to_brain(volume, label=None):
    """Crop spatial dims to bounding box of non-zero voxels.

    volume shape: (H, W, D, 4)
    label  shape: (H, W, D)  [optional]

    Returns cropped volume (and label if provided).
    """
    brain_mask = volume.sum(axis=-1) != 0   # (H, W, D)
    coords = np.where(brain_mask)
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) + 1 for c in coords]
    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))

    cropped_vol = volume[slices]
    if label is not None:
        return cropped_vol, label[slices]
    return cropped_vol


def resize_or_pad(volume, target_shape=(128, 128, 128), is_label=False):
    """Zoom then exact-pad/crop to target_shape.

    volume: (H, W, D, C) or (H, W, D)
    Returns volume with spatial dims == target_shape.
    """
    spatial = volume.shape[:3]
    zoom_factors = [t / s for t, s in zip(target_shape, spatial)]
    if volume.ndim == 4:
        zoom_factors = zoom_factors + [1.0]
    order = 0 if is_label else 1
    resized = zoom(volume, zoom_factors, order=order, prefilter=(order > 1))

    # Guarantee exact output shape (zoom may be off by ±1)
    if volume.ndim == 4:
        out = np.zeros((*target_shape, volume.shape[-1]), dtype=resized.dtype)
    else:
        out = np.zeros(target_shape, dtype=resized.dtype)

    src_slices = tuple(slice(0, min(resized.shape[i], target_shape[i]))
                       for i in range(3))
    dst_slices = src_slices
    if volume.ndim == 4:
        out[dst_slices + (slice(None),)] = resized[src_slices + (slice(None),)]
    else:
        out[dst_slices] = resized[src_slices]
    return out


def preprocess_subject(image_path, label_path=None, target_shape=(128, 128, 128)):
    """Full preprocessing pipeline for one subject.

    Steps:
        1. Load 4D NIfTI → (H, W, D, 4)
        2. Z-score normalize per modality (brain mask)
        3. Resample to 1 mm isotropic
        4. Crop to brain bounding box
        5. Resize/pad to target_shape

    Returns:
        image : float32 ndarray (4, H, W, D)  — channels-first for PyTorch
        label : int64  ndarray (H, W, D)  or  None
    """
    img_data, affine, spacing = load_volume(image_path)  # (H, W, D, 4)

    img_data = zscore_normalize(img_data)
    img_data = resample_volume(img_data, spacing)

    label_data = None
    if label_path:
        lbl_img = nib.load(label_path)
        label_data = lbl_img.get_fdata(dtype=np.float32).astype(np.int64)
        lbl_spacing = tuple(np.abs(np.diag(lbl_img.affine)[:3]).tolist())
        label_data = resample_volume(label_data, lbl_spacing, is_label=True)

    if label_data is not None:
        img_data, label_data = crop_to_brain(img_data, label_data)
    else:
        img_data = crop_to_brain(img_data)

    img_data = resize_or_pad(img_data, target_shape)
    if label_data is not None:
        label_data = resize_or_pad(label_data, target_shape, is_label=True)

    # (H, W, D, 4) → (4, H, W, D)
    img_data = img_data.transpose(3, 0, 1, 2).copy()

    return img_data.astype(np.float32), label_data
