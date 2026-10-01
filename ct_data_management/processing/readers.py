import sys
import logging
import torch
import monai.transforms as mt
import subprocess
import os
import pydicom
from glob import glob
from .pipeline import PipelinePart
from ..utils import SmartTemporaryDirectory

_logger = logging.getLogger('pipeline')

_SEG2NII = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'seg2nii.py')


class DICOMFileSystemReader(PipelinePart):
    def __init__(self, backend='NibabelReader', dtype=None, return_headers=True):
        self._backend = mt.LoadImage(backend, image_only=True, ensure_channel_first=True, dtype=dtype)
        self._return_headers = return_headers

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_path      = data.get('ct')
        seg_path_list = data.get('seg_list')

        if ct_path is not None:
            if self._return_headers:
                params['ct_header'] = pydicom.dcmread(self._find_dicom(ct_path), stop_before_pixels=True)
                #params['ct_id'] = ct_path.split('/')[-1]
            data['ct'] = self._read_ct_dicom(ct_path)

        if seg_path_list is not None:
            if self._return_headers:
                params['seg_header_list'] = [
                    pydicom.dcmread(self._find_dicom(p), stop_before_pixels=True)
                    for p in seg_path_list
                ]
                #params['seg_id'] = seg_path.split('/')[-1]
            data['seg_list'] = [self._read_seg_dicom(p) for p in seg_path_list]

        return data, params

    @staticmethod
    def _find_dicom(path):
        """Return the path to one DICOM file in a directory, or raise a clear error."""
        dcm_files = glob(os.path.join(path, '*.dcm'))
        if not dcm_files:
            raise FileNotFoundError(f'No DICOM files found in: {path}')
        return dcm_files[0]

    def _read_ct_dicom(self, path):
        # Use the actual size of the DICOM files as the space estimate for the
        # temporary NIfTI conversion, not the filesystem-wide usage figure.
        required_space = sum(
            os.path.getsize(f) for f in glob(os.path.join(path, '*.dcm'))
        )
        with SmartTemporaryDirectory(required_space) as temp_dir:
            try:
                subprocess.run(
                    ['dcm2niix', '-z', 'n', '-o', temp_dir, '-f', '_temp_dcm2niix_file', str(path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                stderr_text = e.stderr.decode(errors='replace').strip() if e.stderr else '(no output)'
                raise RuntimeError(
                    f'dcm2niix failed for {os.path.basename(path)}: {stderr_text}'
                ) from None
            # Use a glob rather than an exact name: newer dcm2niix versions append
            # timestamps or echo suffixes (e.g. _temp_dcm2niix_file_20240101.nii).
            nii_files = sorted(glob(os.path.join(temp_dir, '_temp_dcm2niix_file*.nii')))
            if not nii_files:
                raise FileNotFoundError(
                    f'dcm2niix exited successfully but produced no output for: {path}'
                )
            # If dcm2niix splits a series into multiple volumes, take the largest
            # (most voxels), which is always the primary CT volume.
            nii_path = max(nii_files, key=os.path.getsize)
            data = self._backend(nii_path)
        return data

    def _read_seg_dicom(self, path):
        # Estimate output NIfTI size from the DICOM header — not from the DICOM file
        # size — because BINARY SEG stores 1 bit/pixel but we write float32 (4 bytes/pixel),
        # giving up to a 32× expansion. Using the file size would cause SmartTemporaryDirectory
        # to pick a location with far too little free space.
        dcm_files = sorted(glob(os.path.join(path, '*.dcm')), key=os.path.getsize, reverse=True)
        if not dcm_files:
            raise FileNotFoundError(f'No DICOM files found in: {path}')
        hdr        = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        n_segs     = len(hdr.SegmentSequence) if hasattr(hdr, 'SegmentSequence') else 1
        n_frames   = int(getattr(hdr, 'NumberOfFrames', n_segs))
        rows, cols = int(hdr.Rows), int(hdr.Columns)
        # Upper bound: n_segs NIfTI files, each up to n_frames × rows × cols float32 voxels.
        required_space = n_segs * n_frames * rows * cols * 4

        with SmartTemporaryDirectory(required_space) as temp_dir:
            try:
                result = subprocess.run(
                    [sys.executable, _SEG2NII, '-z', 'n', '-o', temp_dir, '-f', '_temp_seg2nii_file', str(path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                stderr_text = e.stderr.decode(errors='replace').strip() if e.stderr else '(no output)'
                raise RuntimeError(
                    f'seg2nii failed for {os.path.basename(path)}: {stderr_text}'
                ) from None
            # Route seg2nii stderr to the appropriate Python log level.
            # INFO: lines → DEBUG (file only, not console); WARNING: lines → WARNING.
            if result.stderr:
                series = os.path.basename(path)
                for line in result.stderr.decode(errors='replace').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('WARNING:'):
                        _logger.warning('seg2nii [%s]: %s', series, line[8:].strip())
                    else:
                        # INFO: lines and anything else → debug (written to log file only)
                        _logger.debug('seg2nii [%s]: %s', series, line)
            nii_files = sorted(glob(os.path.join(temp_dir, '_temp_seg2nii_file*.nii')))
            if not nii_files:
                raise FileNotFoundError(
                    f'seg2nii exited successfully but produced no output for: {path}'
                )
            if len(nii_files) == 1:
                data = self._backend(nii_files[0])
            else:
                # Multi-segment SEG: stack all segments along the channel axis → (N, H, W, D)
                data = torch.cat([self._backend(f) for f in nii_files], dim=0)
        return data


class PACSReader(PipelinePart):
    pass
