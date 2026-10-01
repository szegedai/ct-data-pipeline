from pathlib import Path
import csv
import fcntl
import os
import numpy as np
import nibabel as nib
from .pipeline import PipelinePart


def _decompose_affine(affine):
    """Decompose a 4×4 affine into spacing, origin, and direction cosines.

    Args:
        affine: (4, 4) numpy array — the spatial affine matrix.

    Returns:
        spacing:   (3,) float64 — voxel size in mm along each axis.
        origin:    (3,) float64 — world-space coordinates of voxel (0, 0, 0).
        direction: (3, 3) float64 — unit direction cosine matrix (column-major,
                   i.e. each column is the unit vector for that axis).
    """
    col_vecs = affine[:3, :3]  # (3, 3): columns are spacing-scaled axis vectors
    spacing   = np.linalg.norm(col_vecs, axis=0).astype(np.float64)  # (3,)
    origin    = affine[:3, 3].astype(np.float64)                      # (3,)
    direction = (col_vecs / spacing).astype(np.float64)               # (3, 3)
    return spacing, origin, direction


class NPZWriter(PipelinePart):
    """Save CT and segmentation data as NPZ files.

    Each file contains:
        data:      voxel array (C×H×W×D for 3-D mode, C×H×W for 2-D mode).
        affine:    4×4 spatial affine matrix.
        spacing:   voxel size in mm (x, y, z).
        origin:    world-space origin of voxel (0, 0, 0).
        direction: 3×3 direction cosine matrix.

    Args:
        ct_save_path:  Directory to write CT files into.  ``None`` skips CT writing
                       (used when CT has already been written by an earlier writer).
        seg_save_path: Directory to write segmentation files into.  ``None`` skips
                       segmentation writing.
        save_mode:     '3d' — one NPZ per volume (default):
                              <save_path>/<series_id>.npz
                       '2d' — one NPZ per axial slice (last axis, D in C×H×W×D):
                              <save_path>/<series_id>_<slice_idx:04d>.npz
                       The original 3D affine is stored in every 2D slice file
                       as a spatial reference.
        compress:      If True (default), use np.savez_compressed.
                       If False, use np.savez (faster writes, larger files).
        seg_key:       Key in ``data`` dict to read the segmentation tensor from.
                       Default: ``'nodule_seg'``.
    """

    def __init__(self, ct_save_path, seg_save_path, save_mode='3d',
                 compress=True, seg_key='nodule_seg'):
        if save_mode not in ('3d', '2d'):
            raise ValueError(f"save_mode must be '3d' or '2d', got '{save_mode}'")
        self._ct_save_path  = ct_save_path
        self._seg_save_path = seg_save_path
        self._save_mode     = save_mode
        self._savez         = np.savez_compressed if compress else np.savez
        self._seg_key       = seg_key

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_data  = data.get('ct')
        seg_data = data.get(self._seg_key)

        save_id = params.get('id', None)
        if save_id is None:
            raise ValueError('No valid id was provided. Ensure IDGenerator runs before NPZWriter.')

        if self._save_mode == '3d':
            self._save_3d(ct_data, seg_data, save_id)
        else:
            self._save_2d(ct_data, seg_data, save_id)

        return data, params

    # --- Save modes ---

    def _save_3d(self, ct_data, seg_data, save_id):
        if ct_data is not None and self._ct_save_path is not None:
            self._write(self._ct_save_path, save_id, ct_data.cpu().numpy(), ct_data.affine.cpu().numpy())
        if seg_data is not None and self._seg_save_path is not None:
            self._write(self._seg_save_path, save_id, seg_data.cpu().numpy(), seg_data.affine.cpu().numpy())

    def _save_2d(self, ct_data, seg_data, save_id):
        # Tensor layout after pipeline: (C, H, W, D); slices along last axis.
        n_slices = (ct_data if ct_data is not None else seg_data).shape[-1]

        # Preserve the 3D affine in every slice file as a spatial reference.
        affine = (ct_data if ct_data is not None else seg_data).affine.cpu().numpy()

        for i in range(n_slices):
            slice_id = f'{save_id}_{i:04d}'
            if ct_data is not None and self._ct_save_path is not None:
                self._write(self._ct_save_path, slice_id, ct_data[..., i].cpu().numpy(), affine)
            if seg_data is not None and self._seg_save_path is not None:
                self._write(self._seg_save_path, slice_id, seg_data[..., i].cpu().numpy(), affine)

    def _write(self, save_dir, file_id, data_arr, affine_arr):
        spacing, origin, direction = _decompose_affine(affine_arr)
        path = os.path.join(save_dir, file_id)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._savez(
            path,
            data=data_arr,
            affine=affine_arr,
            spacing=spacing,
            origin=origin,
            direction=direction,
            allow_pickle=False,
        )


class NIfTIWriter(PipelinePart):
    """Save CT and segmentation data as NIfTI files (.nii or .nii.gz).

    The full spatial metadata (spacing, origin, orientation) is encoded in the
    NIfTI affine header so that the files are immediately usable by medical
    imaging software (ITK-SNAP, 3D Slicer, FSL, etc.).

    Data layout:
        The pipeline produces tensors in (C, H, W, D) layout.  NIfTI expects
        (X, Y, Z[, T]) with no channel axis, so this writer squeezes single-
        channel volumes to (H, W, D) and stacks multi-channel volumes to
        (H, W, D, C) as a 4-D NIfTI.

    Args:
        ct_save_path:  Directory to write CT files into.  ``None`` skips CT writing.
        seg_save_path: Directory to write segmentation files into.  ``None`` skips
                       segmentation writing.
        save_mode:     '3d' — one NIfTI per volume (default).
                       '2d' — one NIfTI per axial slice; the affine origin is
                              shifted per-slice so each file is spatially accurate.
        compress:      If True (default), write .nii.gz.
                       If False, write .nii (faster writes, larger files).
        seg_key:       Key in ``data`` dict to read the segmentation tensor from.
                       Default: ``'nodule_seg'``.
    """

    def __init__(self, ct_save_path, seg_save_path, save_mode='3d',
                 compress=True, seg_key='nodule_seg'):
        if save_mode not in ('3d', '2d'):
            raise ValueError(f"save_mode must be '3d' or '2d', got '{save_mode}'")
        self._ct_save_path  = ct_save_path
        self._seg_save_path = seg_save_path
        self._save_mode     = save_mode
        self._ext           = '.nii.gz' if compress else '.nii'
        self._seg_key       = seg_key

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_data  = data.get('ct')
        seg_data = data.get(self._seg_key)

        save_id = params.get('id', None)
        if save_id is None:
            raise ValueError('No valid id was provided. Ensure IDGenerator runs before NIfTIWriter.')

        if self._save_mode == '3d':
            self._save_3d(ct_data, seg_data, save_id)
        else:
            self._save_2d(ct_data, seg_data, save_id)

        return data, params

    # --- Save modes ---

    def _save_3d(self, ct_data, seg_data, save_id):
        if ct_data is not None and self._ct_save_path is not None:
            arr    = ct_data.cpu().numpy()
            affine = ct_data.affine.cpu().numpy()
            self._write(self._ct_save_path, save_id, arr, affine, is_seg=False)
        if seg_data is not None and self._seg_save_path is not None:
            arr    = seg_data.cpu().numpy()
            affine = seg_data.affine.cpu().numpy()
            self._write(self._seg_save_path, save_id, arr, affine, is_seg=True)

    def _save_2d(self, ct_data, seg_data, save_id):
        # Tensor layout: (C, H, W, D); iterate over the depth (last) axis.
        ref = ct_data if ct_data is not None else seg_data
        n_slices   = ref.shape[-1]
        affine_3d  = ref.affine.cpu().numpy()  # (4, 4)

        for i in range(n_slices):
            # Shift the origin along the third spatial axis by i voxels so
            # each slice file records its actual world-space position.
            slice_affine        = affine_3d.copy()
            slice_affine[:3, 3] = affine_3d[:3, 3] + i * affine_3d[:3, 2]

            slice_id = f'{save_id}_{i:04d}'
            if ct_data is not None and self._ct_save_path is not None:
                self._write(self._ct_save_path, slice_id,
                            ct_data[..., i].cpu().numpy(), slice_affine, is_seg=False)
            if seg_data is not None and self._seg_save_path is not None:
                self._write(self._seg_save_path, slice_id,
                            seg_data[..., i].cpu().numpy(), slice_affine, is_seg=True)

    def _write(self, save_dir, file_id, data_arr, affine_arr, is_seg):
        """Write a single NIfTI file.

        Args:
            save_dir:  Destination directory.
            file_id:   Base file name (no extension).
            data_arr:  numpy array in (C, H, W[, D]) layout.
            affine_arr: (4, 4) affine matrix.
            is_seg:    If True, cast to int16 (label map); otherwise float32 (CT).
        """
        # (C, H, W, D) → (H, W, D) for single-channel, (H, W, D, C) for multi.
        if data_arr.shape[0] == 1:
            volume = data_arr[0]           # squeeze channel axis
        else:
            volume = np.moveaxis(data_arr, 0, -1)  # (C, H, W, D) → (H, W, D, C)

        dtype  = np.int16 if is_seg else np.float32
        volume = volume.astype(dtype)

        img  = nib.Nifti1Image(volume, affine=affine_arr)
        path = os.path.join(save_dir, file_id + self._ext)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, path)


_CATALOG_COLUMNS = ['patient_id', 'series_uid', 'dataset', 'volume_mm3', 'entropy']


class NoduleCatalogWriter(PipelinePart):
    """Appends per-nodule statistics to a shared CSV catalog after each processed scan.

    Reads pre-computed per-component statistics from ``params['nodule_components']``
    as populated by ``NoduleStatsTransform``.  Must be placed after that transform
    in the pipeline.

    The CSV is written with an exclusive POSIX file lock so concurrent worker
    processes can safely append to the same file.

    Args:
        catalog_path: Path to the output CSV file.
        dataset:      Dataset label written into every row ('lidc_idri',
                      'nsclc_radiomics', 'nlst_radiologist', or 'nlst_ai').
    """

    def __init__(self, catalog_path: str, dataset: str):
        self._catalog_path = catalog_path
        self._dataset      = dataset

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        components = params.get('nodule_components')
        if not components or not components['stats']:
            return data, params

        series_uid = params.get('id', '')
        patient_id = ''
        ct_header  = params.get('ct_header')
        if ct_header is not None:
            patient_id = str(getattr(ct_header, 'PatientID', ''))

        rows = [
            {
                'patient_id': patient_id,
                'series_uid': series_uid,
                'dataset':    self._dataset,
                'volume_mm3': s['volume_mm3'],
                'entropy':    s['entropy'],
            }
            for s in components['stats']
        ]

        Path(self._catalog_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self._catalog_path, 'a', newline='') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Seek to true end after acquiring the lock — f.tell() at open
                # time reflects the file size when this process opened it, not
                # the current size, so two workers opening an empty file would
                # both see 0 and both write the header without this seek.
                f.seek(0, 2)
                writer = csv.DictWriter(f, fieldnames=_CATALOG_COLUMNS)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(rows)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        return data, params
