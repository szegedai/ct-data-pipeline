#!/usr/bin/env python
"""
seg2nii — convert a DICOM SEG series to NIfTI.
Drop-in replacement for dcm2niix for DICOM SEG series.

Usage:
    python seg2nii.py -o <output_dir> -f <stem> <input_dir>
    python seg2nii.py -z n -a y -o <output_dir> -f <stem> <input_dir>

Output:
    Single segment   → <output_dir>/<stem>.nii
    Multiple segments → <output_dir>/<stem>_1.nii, <stem>_2.nii, ...

Exit codes: 0 on success, 1 on error (matches dcm2niix behaviour).
"""

import argparse
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
import pydicom


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Convert a DICOM SEG series to NIfTI (drop-in for dcm2niix).'
    )
    p.add_argument('input_dir')
    p.add_argument('-o', dest='output_dir', required=True,  help='Output directory')
    p.add_argument('-f', dest='stem',       required=True,  help='Output filename stem')
    p.add_argument('-z', dest='compress',   default='n',    help='Accepted, always writes .nii')
    p.add_argument('-a', dest='anon',       default='y',    help='Accepted, silently ignored')
    # Absorb any other dcm2niix flags without erroring
    args, _ = p.parse_known_args(argv)
    return args


# ---------------------------------------------------------------------------
# Step 1: load the SEG DICOM dataset
# ---------------------------------------------------------------------------

def load_seg_dataset(input_dir: str) -> pydicom.Dataset:
    """
    Load the SEG DICOM from input_dir.
    SEG data is typically stored in a single .dcm file; if multiple exist,
    use the largest (most likely to contain all frames).
    """
    dcm_files = glob(os.path.join(input_dir, '*.dcm'))
    if not dcm_files:
        raise FileNotFoundError(f'No DICOM files found in: {input_dir}')
    dcm_files.sort(key=os.path.getsize, reverse=True)
    return pydicom.dcmread(dcm_files[0])


# ---------------------------------------------------------------------------
# Step 2: unpack pixel data
# ---------------------------------------------------------------------------

def unpack_frames(ds: pydicom.Dataset) -> np.ndarray:
    """
    Decode pixel data into an (n_frames, rows, cols) array.

    BINARY segmentations are bit-packed (1 bit per pixel, LSB-first). Using
    np.unpackbits with explicit bitorder='little' is more reliable than
    pydicom's pixel_array across pydicom versions.

    FRACTIONAL segmentations are standard uint8/uint16 and decoded via
    pydicom's pixel_array.
    """
    n_frames = int(ds.NumberOfFrames)
    rows     = int(ds.Rows)
    cols     = int(ds.Columns)
    seg_type = ds.SegmentationType.strip().upper()

    if seg_type == 'BINARY':
        raw   = np.frombuffer(ds.PixelData, dtype=np.uint8)
        bits  = np.unpackbits(raw, bitorder='little')
        total = n_frames * rows * cols
        return bits[:total].reshape(n_frames, rows, cols).astype(np.float32)
    else:
        frames = ds.pixel_array
        if frames.ndim == 2:
            frames = frames[np.newaxis]
        return frames.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: geometry helpers
# ---------------------------------------------------------------------------

def compute_slice_spacing(ipps: list, normal: np.ndarray) -> float:
    """
    Compute slice spacing (mm) as the median gap between unique slice positions
    projected onto the slice normal. Median is robust against occasional
    duplicate or outlier IPP values in malformed files.
    """
    projs = sorted({round(float(np.dot(ipp, normal)), 4) for ipp in ipps})
    if len(projs) < 2:
        return 1.0  # degenerate single-frame case; 1 mm fallback
    gaps = [projs[i + 1] - projs[i] for i in range(len(projs) - 1)]
    return float(np.median(gaps))


def build_affine(iop: np.ndarray, ipp_first: np.ndarray,
                 pixel_spacing: np.ndarray, slice_spacing: float) -> np.ndarray:
    """
    Build a 4×4 NIfTI affine matrix from DICOM geometry tags.

    Our volume has shape (rows, cols, n_slices), so:
      axis 0 = row index j  → moves in the column direction (col_cos) by row_spacing (dr)
      axis 1 = col index i  → moves in the row direction (row_cos) by col_spacing (dc)
      axis 2 = slice index k → moves along the slice normal by slice_spacing

    iop          : (6,)  ImageOrientationPatient — row cosines [0:3], col cosines [3:6]
    ipp_first    : (3,)  ImagePositionPatient of the k=0 slice
    pixel_spacing: (2,)  [row_spacing_mm, col_spacing_mm]  (DICOM PixelSpacing order)
    slice_spacing: float mm between consecutive k-planes
    """
    row_cos = iop[:3]
    col_cos = iop[3:]
    normal  = np.cross(row_cos, col_cos)
    dr, dc  = pixel_spacing
    affine  = np.eye(4, dtype=np.float64)
    affine[:3, 0] = col_cos * dr   # axis 0 (row j)   → col direction × row spacing
    affine[:3, 1] = row_cos * dc   # axis 1 (col i)   → row direction × col spacing
    affine[:3, 2] = normal  * slice_spacing
    affine[:3, 3] = ipp_first
    return affine


# ---------------------------------------------------------------------------
# Step 4: reconstruct volumes
# ---------------------------------------------------------------------------

def reconstruct(ds: pydicom.Dataset) -> tuple:
    """
    Reconstruct one 3-D volume per logical segment.

    The grid extent is derived from the unique slice positions present in the
    SEG frames. If the SEG only annotates a subset of the CT slices (sparse
    annotation), the output NIfTI covers only that extent — the affine
    encodes the correct spatial offset so downstream resampling aligns it to
    the CT correctly.

    Returns:
        volumes : dict { segment_number (int) -> np.ndarray (rows, cols, n_slices) float32 }
        affine  : np.ndarray (4, 4) float64
    """
    if not hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
        raise ValueError(
            'PerFrameFunctionalGroupsSequence is missing. '
            'Only enhanced / multi-frame DICOM SEG is supported.'
        )

    shared = ds.SharedFunctionalGroupsSequence[0]
    iop    = np.array([float(v) for v in
                       shared.PlaneOrientationSequence[0].ImageOrientationPatient])
    ps     = np.array([float(v) for v in
                       shared.PixelMeasuresSequence[0].PixelSpacing])
    normal = np.cross(iop[:3], iop[3:])

    per_frame = ds.PerFrameFunctionalGroupsSequence

    # Read all per-frame IPPs up front (reused for both grid construction and placement)
    all_ipps = [
        np.array([float(v) for v in
                  frame.PlanePositionSequence[0].ImagePositionPatient])
        for frame in per_frame
    ]

    spacing     = compute_slice_spacing(all_ipps, normal)
    projs       = [float(np.dot(ipp, normal)) for ipp in all_ipps]
    ipp_first   = all_ipps[int(np.argmin(projs))]
    n_slices    = len({round(p, 4) for p in projs})
    proj_origin = float(np.dot(ipp_first, normal))

    # DICOM stores IPP and IOP in LPS coordinates (x+=Left, y+=Posterior, z+=Superior).
    # NIfTI/nibabel/MONAI expect RAS (x+=Right, y+=Anterior, z+=Superior).
    # dcm2niix applies this conversion for the CT — we must do the same for the SEG.
    # Conversion: negate x and y, keep z.
    lps2ras  = np.array([-1., -1., 1.])
    iop_ras  = np.concatenate([iop[:3] * lps2ras, iop[3:] * lps2ras])
    ipp_ras  = ipp_first * lps2ras
    affine   = build_affine(iop_ras, ipp_ras, ps, spacing)

    rows, cols = int(ds.Rows), int(ds.Columns)
    volumes = {
        int(seg.SegmentNumber): np.zeros((rows, cols, n_slices), dtype=np.float32)
        for seg in ds.SegmentSequence
    }

    frames  = unpack_frames(ds)  # (n_frames, rows, cols)
    skipped = 0

    for i, frame_meta in enumerate(per_frame):
        seg_no = int(frame_meta.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        k      = int(round((projs[i] - proj_origin) / spacing))

        if not (0 <= k < n_slices):
            skipped += 1
            continue

        if seg_no in volumes:
            volumes[seg_no][:, :, k] = frames[i]

    if skipped:
        print(f'WARNING: {skipped} frame(s) outside grid extent — skipped.',
              file=sys.stderr)

    for seg_no, vol in volumes.items():
        n_nonzero = int(np.count_nonzero(vol))
        if n_nonzero == 0:
            print(f'WARNING: segment {seg_no} has 0 annotated voxels after reconstruction.',
                  file=sys.stderr)
        else:
            print(f'INFO: segment {seg_no}: {n_nonzero} annotated voxels '
                  f'in volume {vol.shape}.',
                  file=sys.stderr)

    return volumes, affine


# ---------------------------------------------------------------------------
# Step 5: write NIfTI output
# ---------------------------------------------------------------------------

def write_nifti(volumes: dict, affine: np.ndarray,
                output_dir: str, stem: str) -> list:
    """
    Write one uncompressed NIfTI file per segment.

    Naming convention (matches dcm2niix):
        single segment   → <stem>.nii
        multiple segments → <stem>_1.nii, <stem>_2.nii, ...  (1-indexed, by SegmentNumber)

    Returns list of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    seg_numbers = sorted(volumes.keys())
    written     = []

    for seg_no in seg_numbers:
        fname = f'{stem}.nii' if len(seg_numbers) == 1 else f'{stem}_{seg_no}.nii'
        path  = os.path.join(output_dir, fname)
        nib.save(nib.Nifti1Image(volumes[seg_no], affine), path)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    try:
        ds              = load_seg_dataset(args.input_dir)
        volumes, affine = reconstruct(ds)
        written         = write_nifti(volumes, affine, args.output_dir, args.stem)
        for p in written:
            print(f'Converted: {p}')
        sys.exit(0)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
