import numpy as np
import torch
import torch.nn.functional as F
import monai.transforms as mt
import scipy.ndimage as ndi
from monai.data import MetaTensor
from monai.transforms import SpatialCrop
from .pipeline import PipelinePart


class DataAnomalyError(Exception):
    pass


class IDGenerator(PipelinePart):
    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_header = params.get('ct_header', None)
        if ct_header is not None:
            params['id'] = ct_header.SeriesInstanceUID
        return data, params


class FilterSegmentsTransform(PipelinePart):
    """Filter DICOM segmentation files into separate nodule and lung mask lists.

    For each SEG file the transform runs two independent label-matching passes and
    accumulates results into ``data['nodule_seg_list']`` and ``data['lung_seg_list']``
    respectively.  A label set of ``None`` means "do not load this type".

    Raises ``DataAnomalyError`` after processing all files if a non-None label set
    produced no matching segments across any file.
    """

    def __init__(self, nodule_labels=None, lung_labels=None):
        self._nodule_labels = list(nodule_labels) if nodule_labels else None
        self._lung_labels   = list(lung_labels)   if lung_labels   else None

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg_header_list = params.get('seg_header_list', [])
        seg_data_list   = data.pop('seg_list', [])   # raw reader field; consumed here

        nodule_out, lung_out = [], []

        for seg_header, seg_data in zip(seg_header_list, seg_data_list):
            seg_map = self._build_segment_map(seg_header)

            if self._nodule_labels is not None:
                mask = self._extract(seg_data, seg_map, self._nodule_labels)
                if mask is not None:
                    nodule_out.append(mask)

            if self._lung_labels is not None:
                mask = self._extract(seg_data, seg_map, self._lung_labels)
                if mask is not None:
                    lung_out.append(mask)

        if self._nodule_labels is not None and not nodule_out:
            raise DataAnomalyError('No nodule segmentation found.')
        if self._lung_labels is not None and not lung_out:
            raise DataAnomalyError('No lung segmentation found.')

        if nodule_out:
            data['nodule_seg_list'] = nodule_out
        if lung_out:
            data['lung_seg_list'] = lung_out

        return data, params

    @staticmethod
    def _build_segment_map(seg_header):
        """Return {segment_index: lowercase_label} from DICOM SegmentSequence."""
        m = {}
        if 'SegmentSequence' in seg_header:
            for item in seg_header.SegmentSequence:
                idx   = item.SegmentNumber - 1
                label = (getattr(item, 'SegmentLabel', None)
                         or getattr(item, 'SegmentDescription', '') or '').lower()
                m[idx] = label
        return m

    @staticmethod
    def _extract(seg_data, seg_map, label_patterns):
        """Sum channels whose label matches any pattern; return None if none match."""
        indices = [i for i, lbl in seg_map.items()
                   if any(p in lbl for p in label_patterns)
                   and i < seg_data.size(0)]
        if not indices:
            return None
        return seg_data[indices].sum(0, keepdim=True).clamp_(0, 1)


class HUClipAndNormTransform(PipelinePart):
    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_data = data.get('ct')

        if ct_data is not None:
            ct_data.clip_(self.clip_min, self.clip_max)
            data['ct'] = (ct_data - self.clip_min) / (self.clip_max - self.clip_min)

        return data, params


class OrientTransform(PipelinePart):
    def __init__(self, orientation='RAS'):
        self._backend = mt.Orientation(axcodes=orientation, labels=None)

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is not None:
            data['ct'] = self._backend(data['ct'])
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [self._backend(s) for s in data[key]]
        return data, params


class ResampleTransform(PipelinePart):
    def __init__(self, spacing=(1., 1., 1.), ct_mode='bilinear', seg_mode='nearest'):
        self._ct_backend  = mt.Spacing(pixdim=spacing, mode=ct_mode)
        self._seg_backend = mt.ResampleToMatch(mode=seg_mode, padding_mode='zeros')

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is not None:
            data['ct'] = self._ct_backend(data['ct'])
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [self._seg_backend(s, data['ct']) for s in data[key]]
        return data, params


class LungDilationTransform(PipelinePart):
    """Expand the lung mask outward by iterating binary dilation.

    Operates on ``data['lung_seg_list']`` only.  The nodule masks are untouched.
    When this runs before ``ROICropTransform`` the crop bounding box (computed from
    dilated_lung ∪ nodule) will include a margin around the pleural boundary,
    covering nodules that protrude past the lung annotation.

    Uses 26-connectivity so the expansion is isotropic in all directions.
    """

    _STRUCTURE = ndi.generate_binary_structure(3, 3)   # 26-connectivity

    def __init__(self, n_dilations: int):
        self._n = n_dilations

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        lung_segs = data.get('lung_seg_list')
        if not lung_segs:
            return data, params
        data['lung_seg_list'] = [
            MetaTensor(
                torch.from_numpy(
                    ndi.binary_dilation(
                        seg[0].cpu().numpy().astype(bool),
                        structure=self._STRUCTURE,
                        iterations=self._n,
                    )
                ).unsqueeze(0).to(seg.dtype),
                meta=seg.meta,
            )
            for seg in lung_segs
        ]
        return data, params


class ROICropTransform(PipelinePart):
    """Crop CT and all segmentation masks to a bounding box around the chest.

    The bounding box is derived from the **temporary union** of all lung masks
    (possibly dilated by a preceding ``LungDilationTransform``) and all nodule
    masks.  This ensures that:

    - Nodules excluded from the lung annotation are still inside the crop.
    - Nodules that protrude past the pleural surface are fully retained.

    The union is not stored; ``lung_seg_list`` and ``nodule_seg_list`` remain
    separate after cropping.  MONAI's ``SpatialCrop`` updates each tensor's
    affine automatically, so downstream writers produce geometrically correct
    output.
    """

    def __init__(self, padding: int = 1):
        self._padding = padding

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is None:
            return data, params

        # Gather all lung + nodule masks as the bounding-box reference.
        ref_segs = []
        for key in ('lung_seg_list', 'nodule_seg_list'):
            if data.get(key):
                ref_segs.extend(data[key])

        if not ref_segs:
            return data, params

        shape = ref_segs[0][0].shape   # (H, W, D)
        union = np.zeros(shape, dtype=bool)
        for seg in ref_segs:
            union |= seg[0].cpu().numpy().astype(bool)

        if not union.any():
            raise DataAnomalyError('Combined lung + nodule mask is empty; cannot compute crop.')

        roi_start, roi_end = [], []
        for ax in range(3):
            nax = tuple(j for j in range(3) if j != ax)
            idx = np.where(union.any(axis=nax))[0]
            roi_start.append(max(0,         int(idx.min()) - self._padding))
            roi_end.append(  min(shape[ax], int(idx.max()) + 1 + self._padding))

        cropper = SpatialCrop(roi_start=roi_start, roi_end=roi_end)
        data['ct'] = cropper(data['ct'])
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [cropper(s) for s in data[key]]

        return data, params


class NoduleStatsTransform(PipelinePart):
    """Compute per-nodule volume and entropy statistics after merging and normalisation.

    Performs 26-connectivity connected-component analysis on ``data['nodule_seg']``
    and stores per-component statistics in ``params['nodule_components']``::

        {
            'labeled': np.ndarray,    # (H, W, D) integer component labels
            'stats': [
                {
                    'label':        int,    # component label in 'labeled'
                    'volume_mm3':   float,
                    'entropy':      float,  # Shannon entropy in bits
                    'mean_hu_norm': float,  # mean normalised CT value in [0, 1]
                },
                ...
            ]
        }

    This transform is the single source of per-nodule statistics.  Downstream
    filter transforms (``NoduleVolumeFilterTransform``, ``NoduleHUFilterTransform``, ``NoduleEntropyFilterTransform``)
    and ``NoduleCatalogWriter`` all read from ``params['nodule_components']`` rather
    than deriving statistics independently.

    Must be placed after both ``MergeSegmentsTransform`` and ``HUClipAndNormTransform``.

    Args:
        hu_clip_min:  Lower HU bound used during normalisation (default −1000).
        hu_clip_max:  Upper HU bound used during normalisation (default  400).
        bin_width_hu: Histogram bin width in HU for entropy (default 25).
    """

    _STRUCTURE = ndi.generate_binary_structure(3, 3)   # 26-connectivity

    def __init__(
        self,
        hu_clip_min: float = -1000.0,
        hu_clip_max: float = 400.0,
        bin_width_hu: float = 25.0,
    ):
        self._n_bins = round((hu_clip_max - hu_clip_min) / bin_width_hu)

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg    = data.get('nodule_seg')
        ct_data = data.get('ct')
        if seg is None or ct_data is None:
            return data, params

        ct_np   = ct_data[0].cpu().numpy().astype(np.float32)
        mask_np = seg[0].cpu().numpy()

        affine       = seg.affine.cpu().numpy()
        spacing      = np.linalg.norm(affine[:3, :3], axis=0)
        voxel_volume = float(np.prod(spacing))

        labeled, n = ndi.label(mask_np > 0, structure=self._STRUCTURE)

        stats = []
        for lbl in range(1, n + 1):
            comp        = labeled == lbl
            voxel_count = int(comp.sum())
            if voxel_count == 0:
                continue
            ct_voxels = ct_np[comp]
            counts, _ = np.histogram(ct_voxels, bins=self._n_bins, range=(0.0, 1.0))
            p         = counts / counts.sum()
            p         = p[p > 0]
            entropy   = abs(float(-np.sum(p * np.log2(p))))
            stats.append({
                'label':        lbl,
                'volume_mm3':   round(voxel_count * voxel_volume, 4),
                'entropy':      round(entropy, 6),
                'mean_hu_norm': float(ct_voxels.mean()),
            })

        params['nodule_components'] = {'labeled': labeled, 'stats': stats}
        return data, params


class NoduleVolumeFilterTransform(PipelinePart):
    """Remove nodule components outside a plausible volume range.

    Reads and updates ``params['nodule_components']`` populated by
    ``NoduleStatsTransform``, so it must be placed after that transform.

    Thresholds are in mm³.  After resampling to 1 × 1 × 1 mm spacing,
    volume_mm3 ≈ voxel count.

    Raises ``DataAnomalyError`` if every component is removed by the filter.

    Suggested thresholds (adjust after medical expert review):
        min_volume =    14.0   ≈ 3 mm Ø sphere (Fleischner lower bound)
        max_volume = 14137.0   ≈ 30 mm Ø sphere (Fleischner nodule–mass boundary)
    """

    def __init__(
        self,
        min_volume: float = float('-inf'),
        max_volume: float = float('inf'),
    ):
        self._min = min_volume
        self._max = max_volume

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        components = params.get('nodule_components')
        if not components or not components['stats']:
            return data, params

        stats   = components['stats']
        labeled = components['labeled']
        seg     = data['nodule_seg']

        passing = [s for s in stats if self._min <= s['volume_mm3'] <= self._max]

        if not passing:
            raise DataAnomalyError(
                f'All {len(stats)} nodule component(s) removed by volume filter '
                f'(volume outside [{self._min}, {self._max}] mm³). Series dropped.'
            )

        if len(passing) < len(stats):
            keeping  = {s['label'] for s in passing}
            new_mask = np.isin(labeled, list(keeping))
            data['nodule_seg'] = MetaTensor(
                torch.from_numpy(new_mask).unsqueeze(0).to(seg.dtype),
                meta=seg.meta,
            )
            params['nodule_components'] = {'labeled': labeled, 'stats': passing}

        return data, params


class NoduleHUFilterTransform(PipelinePart):
    """Remove nodule components outside a plausible mean HU range.

    Reads and updates ``params['nodule_components']`` populated by
    ``NoduleStatsTransform``, so it must be placed after that transform.

    ``min_hu`` / ``max_hu`` are in raw HU units and are converted to the
    normalised [0, 1] scale internally using ``hu_clip_min`` / ``hu_clip_max``,
    so they are directly comparable to the ``mean_hu_norm`` values in the stats.

    Raises ``DataAnomalyError`` if every component is removed by the filter.

    Suggested thresholds (adjust after medical expert review):
        min_hu = -800.0   below ground-glass range; excludes air artefacts
        max_hu =  700.0   above calcified nodules; excludes dense cortical bone
    """

    def __init__(
        self,
        min_hu: float = float('-inf'),
        max_hu: float = float('inf'),
        hu_clip_min: float = -1000.0,
        hu_clip_max: float = 400.0,
    ):
        self._min_hu      = min_hu
        self._max_hu      = max_hu
        hu_range          = hu_clip_max - hu_clip_min
        self._min_hu_norm = (min_hu - hu_clip_min) / hu_range
        self._max_hu_norm = (max_hu - hu_clip_min) / hu_range

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        components = params.get('nodule_components')
        if not components or not components['stats']:
            return data, params

        stats   = components['stats']
        labeled = components['labeled']
        seg     = data['nodule_seg']

        passing = [
            s for s in stats
            if self._min_hu_norm <= s['mean_hu_norm'] <= self._max_hu_norm
        ]

        if not passing:
            raise DataAnomalyError(
                f'All {len(stats)} nodule component(s) removed by HU filter '
                f'(mean HU outside [{self._min_hu}, {self._max_hu}] HU). Series dropped.'
            )

        if len(passing) < len(stats):
            keeping  = {s['label'] for s in passing}
            new_mask = np.isin(labeled, list(keeping))
            data['nodule_seg'] = MetaTensor(
                torch.from_numpy(new_mask).unsqueeze(0).to(seg.dtype),
                meta=seg.meta,
            )
            params['nodule_components'] = {'labeled': labeled, 'stats': passing}

        return data, params


class NoduleEntropyFilterTransform(PipelinePart):
    """Remove nodule components whose Shannon entropy falls outside a given range.

    Reads and updates ``params['nodule_components']`` populated by
    ``NoduleStatsTransform``, so it must be placed after that transform.

    Entropy values match those written to the nodule catalog by
    ``NoduleCatalogWriter``, so thresholds can be derived directly from
    catalog analysis.

    Raises ``DataAnomalyError`` if every component is removed by the filter.

    Args:
        min_entropy: Lower entropy bound in bits (inclusive).  Default −∞.
        max_entropy: Upper entropy bound in bits (inclusive).  Default +∞.
    """

    def __init__(
        self,
        min_entropy: float = float('-inf'),
        max_entropy: float = float('inf'),
    ):
        self._min = min_entropy
        self._max = max_entropy

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        components = params.get('nodule_components')
        if not components or not components['stats']:
            return data, params

        stats   = components['stats']
        labeled = components['labeled']
        seg     = data['nodule_seg']

        passing = [s for s in stats if self._min <= s['entropy'] <= self._max]

        if not passing:
            raise DataAnomalyError(
                f'All {len(stats)} nodule component(s) removed by entropy filter '
                f'(entropy outside [{self._min}, {self._max}] bits). Series dropped.'
            )

        if len(passing) < len(stats):
            keeping  = {s['label'] for s in passing}
            new_mask = np.isin(labeled, list(keeping))
            data['nodule_seg'] = MetaTensor(
                torch.from_numpy(new_mask).unsqueeze(0).to(seg.dtype),
                meta=seg.meta,
            )
            params['nodule_components'] = {'labeled': labeled, 'stats': passing}

        return data, params


class MergeSegmentsTransform(PipelinePart):
    """Merge per-file segmentation lists into single tensors.

    Consumes ``data['nodule_seg_list']`` and ``data['lung_seg_list']`` and
    produces ``data['nodule_seg']`` and ``data['lung_seg']`` respectively.
    Lists with a single entry are used directly; multiple entries are summed
    (binary OR) and clamped to [0, 1].
    """

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        for list_key, seg_key in (('nodule_seg_list', 'nodule_seg'),
                                   ('lung_seg_list',   'lung_seg')):
            lst = data.pop(list_key, None)
            if not lst:
                data[seg_key] = None
            elif len(lst) == 1:
                data[seg_key] = lst[0]
            else:
                data[seg_key] = torch.stack(lst, dim=0).sum(dim=0).clamp_(0, 1)
        return data, params


class ComputeROITransform(PipelinePart):
    """Compute the ROI mask as the union of the lung and nodule segmentations.

    This is the only point in the pipeline where a merged lung+nodule mask
    exists.  The result is stored in ``data['roi_seg']`` and is only added to
    the pipeline when ``roi`` is included in ``--output``.

    Must run before ``NoduleInstanceSegTransform`` so that ``roi_seg`` is a
    clean binary mask, not a mix of instance labels and binary lung values.
    """

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        lung   = data.get('lung_seg')
        nodule = data.get('nodule_seg')

        if lung is None and nodule is None:
            raise DataAnomalyError('Cannot compute ROI: both lung_seg and nodule_seg are absent.')

        if lung is None:
            data['roi_seg'] = nodule
        elif nodule is None:
            data['roi_seg'] = lung
        else:
            roi = (lung + nodule).clamp_(0, 1)
            data['roi_seg'] = MetaTensor(roi, meta=lung.meta)

        return data, params


class ToDeviceTransform(PipelinePart):
    def __init__(self, device='cpu'):
        self._device = torch.device(device)

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        for key in ('ct', 'nodule_seg', 'lung_seg', 'roi_seg'):
            if data.get(key) is not None:
                data[key] = data[key].to(self._device)
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [s.to(self._device) for s in data[key]]
        return data, params


class NoduleInstanceSegTransform(PipelinePart):
    """Convert a binary nodule segmentation mask into an instance segmentation mask.

    Runs 3D connected-component analysis (26-connectivity) on ``data['nodule_seg']``
    so that each spatially distinct nodule receives a unique positive integer
    label. Background remains 0.  Two voxels that touch — even diagonally —
    are treated as belonging to the same nodule, matching the assumption that
    touching nodules in 3D CT are a single lesion.

    Must be placed after ``MergeSegmentsTransform`` (which produces
    ``data['nodule_seg']``) and after ``ComputeROITransform`` (so that
    ``roi_seg`` is a binary union, not mixed with instance labels).
    """

    _STRUCTURE = ndi.generate_binary_structure(3, 3)   # 26-connectivity

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg = data.get('nodule_seg')
        if seg is None:
            return data, params

        mask_np = seg[0].cpu().numpy().astype(bool)
        labeled_np, _ = ndi.label(mask_np, structure=self._STRUCTURE)

        # int16 supports up to 32 767 instances per volume, which is ample.
        labeled_tensor = torch.from_numpy(labeled_np).unsqueeze(0).to(torch.int16)
        data['nodule_seg'] = MetaTensor(labeled_tensor, meta=seg.meta)

        return data, params
