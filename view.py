"""Lightweight utility for inspecting NPZ and NIfTI pipeline output files."""

import argparse
import sys
import numpy as np
import torch

from ct_data_management.processing.utils import InteractiveViewer


def load(path: str) -> torch.Tensor:
    """Load a .npz, .nii, or .nii.gz file and return a (C, H, W, D) float32 tensor."""
    if path.endswith('.npz'):
        return _load_npz(path)
    elif path.endswith('.nii') or path.endswith('.nii.gz'):
        return _load_nifti(path)
    else:
        print(f"Error: unsupported file format '{path}'. Expected .npz, .nii, or .nii.gz.", file=sys.stderr)
        sys.exit(1)


def _load_npz(path: str) -> torch.Tensor:
    npz = np.load(path)
    data = npz['data'].astype(np.float32)
    if data.ndim == 2:
        data = data[None, :, :, None]   # (H, W) → (1, H, W, 1)
    elif data.ndim == 3:
        data = data[None]               # (H, W, D) → (1, H, W, D)
    # data is already (C, H, W, D) if ndim == 4
    return torch.from_numpy(data)


def _load_nifti(path: str) -> torch.Tensor:
    import nibabel as nib
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None, None]       # (H, W) → (H, W, D=1, C=1)
        data = data.transpose(3, 0, 1, 2)   # → (1, H, W, 1)
    elif data.ndim == 3:
        data = data[:, :, :, None]          # (H, W, D) → (H, W, D, 1)
        data = data.transpose(3, 0, 1, 2)   # → (1, H, W, D)
    elif data.ndim == 4:
        data = data.transpose(3, 0, 1, 2)   # (H, W, D, C) → (C, H, W, D)
    else:
        print(f"Error: unexpected NIfTI data shape {data.shape}.", file=sys.stderr)
        sys.exit(1)
    return torch.from_numpy(data)


def instances_to_channels(seg: torch.Tensor) -> torch.Tensor:
    """Convert instance-ID encoding to channel encoding.

    If any channel contains values > 1, that channel is interpreted as an
    instance segmentation mask where each positive integer is a unique instance.
    Each instance is split into its own binary channel.

    Channels that are already binary (max <= 1) are left as-is.
    """
    out_channels = []
    for c in range(seg.shape[0]):
        channel = seg[c]
        if channel.max() > 1:
            instance_ids = channel.unique()
            instance_ids = instance_ids[instance_ids > 0]
            for iid in instance_ids:
                out_channels.append((channel == iid).float()[None])
        else:
            out_channels.append(channel[None])
    return torch.cat(out_channels, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description='Inspect NPZ or NIfTI pipeline output files using the InteractiveViewer.'
    )
    parser.add_argument('ct', help='CT volume file (.npz, .nii, .nii.gz)')
    parser.add_argument('seg', nargs='*', default=[],
                        help='Segmentation file(s) (.npz, .nii, .nii.gz) — optional, multiple allowed')
    args = parser.parse_args()

    ct_tensor = load(args.ct)
    data = {'ct': ct_tensor}

    if args.seg:
        seg_tensors = [instances_to_channels(load(p)) for p in args.seg]
        data['nodule_seg'] = torch.cat(seg_tensors, dim=0)

    InteractiveViewer()(data, {})


if __name__ == '__main__':
    main()
