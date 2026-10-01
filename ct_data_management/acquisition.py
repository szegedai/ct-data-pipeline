from dataclasses import dataclass
from idc_index import index
import logging
import pandas as pd
import os
import warnings
import pydicom
import shutil
from tqdm import tqdm
from glob import glob

_logger = logging.getLogger('pipeline')


@dataclass(frozen=True)
class SeriesFilter:
    """
    Matches a SEG series by SeriesDescription.
    Append '%' to the pattern to use a SQL LIKE / prefix match.
    """
    pattern: str

    def to_sql_condition(self) -> str:
        if '%' in self.pattern:
            return f"SeriesDescription LIKE '{self.pattern}'"
        return f"SeriesDescription = '{self.pattern}'"

    def matches(self, description: str) -> bool:
        if self.pattern.endswith('%'):
            return description.startswith(self.pattern[:-1])
        return description == self.pattern


@dataclass(frozen=True)
class DatasetConfig:
    collection_id: str
    # Ordered highest-priority first. For each study, only the SEGs matched
    # by the first filter that has at least one hit are downloaded.
    seg_priority_filters: tuple[SeriesFilter, ...]


NSCLC_RADIOMICS_INFO = DatasetConfig(
    collection_id='nsclc_radiomics',
    seg_priority_filters=(SeriesFilter('Segmentation'),),
)

LIDC_IDRI_INFO = DatasetConfig(
    collection_id='lidc_idri',
    seg_priority_filters=(SeriesFilter('Segmentation of Nodule %'),),
)

NLST_RADIOLOGIST_INFO = DatasetConfig(
    collection_id='nlst',
    seg_priority_filters=(
        SeriesFilter('AIMI lung and nodule radiologist 8 corrected segmentation'),
        SeriesFilter('AIMI lung and nodule radiologist 5 corrected segmentation'),
        SeriesFilter('AIMI lung and nodule radiologist 4 corrected segmentation'),
    ),
)

NLST_AI_INFO = DatasetConfig(
    collection_id='nlst',
    seg_priority_filters=(
        SeriesFilter('AIMI lung and nodule AI segmentation'),
    ),
)


def _select_by_priority(
    group: pd.DataFrame,
    filters: tuple[SeriesFilter, ...],
) -> list[str]:
    for f in filters:
        matched = group[group['SeriesDescription'].apply(f.matches)]
        if not matched.empty:
            return matched['SeriesInstanceUID'].tolist()
    return []


class DataIntegrityError(Exception):
    pass


class IDCFileSystemDataManager:
    _MANIFEST_FILE = 'manifest.csv'

    def __init__(self, data_path, data_info: DatasetConfig):
        self._data_path = data_path
        self._data_info = data_info

    # --- Public API ---

    def sync_data(self):
        self._download_segmentations()
        self._build_manifest()
        self._download_ct()
        return self

    def get_paths(self):
        manifest = self._load_manifest()

        grouped = (
            manifest
            .groupby('CTSeriesInstanceUID')['SEGSeriesInstanceUID']
            .apply(list)
            .reset_index()
        )

        ct_dir  = os.path.join(self._data_path, 'ct')
        seg_dir = os.path.join(self._data_path, 'seg')

        for row in grouped.itertuples(index=False):
            ct_id     = row.CTSeriesInstanceUID
            seg_ids   = row.SEGSeriesInstanceUID
            ct_path   = os.path.join(ct_dir, ct_id)
            seg_paths = [os.path.join(seg_dir, seg_id) for seg_id in seg_ids]

            if not os.path.isdir(ct_path):
                _logger.warning('CT directory missing, skipping: %s', ct_path)
                continue

            valid_seg_paths = [p for p in seg_paths if os.path.isdir(p)]
            n_missing = len(seg_paths) - len(valid_seg_paths)
            if n_missing:
                _logger.warning('%d SEG director(ies) missing for CT %s.', n_missing, ct_id)
            if not valid_seg_paths:
                _logger.warning('No valid SEG directories remain for CT %s, skipping.', ct_id)
                continue

            yield ct_path, valid_seg_paths

    # --- Sync stages (each idempotent; can be called independently) ---

    def _download_segmentations(self):
        client = self._make_client()
        filters = self._data_info.seg_priority_filters
        conditions = ' OR '.join(f.to_sql_condition() for f in filters)

        _logger.info('Querying IDC for SEG series...')
        candidates = client.sql_query(f'''
            SELECT SeriesInstanceUID, StudyInstanceUID, SeriesDescription
            FROM   index
            WHERE  collection_id = '{self._data_info.collection_id}'
              AND  Modality = 'SEG'
              AND  ({conditions})
        ''')

        seg_ids = []
        for _, study_group in candidates.groupby('StudyInstanceUID'):
            seg_ids.extend(_select_by_priority(study_group, filters))

        _logger.info('Downloading %d SEG series...', len(seg_ids))
        seg_dir = os.path.join(self._data_path, 'seg')
        os.makedirs(seg_dir, exist_ok=True)
        client.download_dicom_series(
            seriesInstanceUID=seg_ids,
            downloadDir=seg_dir,
            dirTemplate='%SeriesInstanceUID',
        )

    def _build_manifest(self):
        manifest_path = os.path.join(self._data_path, self._MANIFEST_FILE)

        if os.path.exists(manifest_path):
            existing    = pd.read_csv(manifest_path, header=0)
            known_segs  = set(existing['SEGSeriesInstanceUID'].tolist())
        else:
            existing   = pd.DataFrame(columns=['CTSeriesInstanceUID', 'SEGSeriesInstanceUID'])
            known_segs = set()

        seg_dir = os.path.join(self._data_path, 'seg')
        seg_series_dirs = sorted([
            os.path.join(seg_dir, d)
            for d in os.listdir(seg_dir)
            if os.path.isdir(os.path.join(seg_dir, d))
        ])

        _logger.info('Building manifest from SEG DICOM headers...')
        new_pairs = []
        for series_path in tqdm(seg_series_dirs):
            seg_id = os.path.basename(series_path)
            if seg_id in known_segs:
                continue

            dcm_files = glob(os.path.join(series_path, '*.dcm'))
            if not dcm_files:
                _logger.warning('No DICOM files in %s, skipping.', series_path)
                continue

            seg_metadata = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            try:
                ct_id = str(seg_metadata.ReferencedSeriesSequence[0].SeriesInstanceUID)
            except (AttributeError, IndexError) as e:
                _logger.error('Metadata extraction failed for %s: %s', series_path, type(e).__name__)
                _logger.error('Removing defective SEG series: %s', series_path)
                shutil.rmtree(series_path)
                continue

            new_pairs.append({'CTSeriesInstanceUID': ct_id, 'SEGSeriesInstanceUID': seg_id})

        if new_pairs:
            updated = pd.concat([existing, pd.DataFrame(new_pairs)], ignore_index=True)
            updated.to_csv(manifest_path, index=False)
            _logger.info('Manifest updated with %d new entries → %s', len(new_pairs), manifest_path)
        else:
            _logger.info('Manifest is already up to date.')

    def _download_ct(self):
        manifest = self._load_manifest()
        all_ct_ids = manifest['CTSeriesInstanceUID'].unique().tolist()

        ct_dir = os.path.join(self._data_path, 'ct')
        os.makedirs(ct_dir, exist_ok=True)

        missing_ct_ids = [ct for ct in all_ct_ids if not os.path.isdir(os.path.join(ct_dir, ct))]
        if not missing_ct_ids:
            _logger.info('All CT series already present on disk.')
            return

        _logger.info('Downloading %d/%d CT series...', len(missing_ct_ids), len(all_ct_ids))
        client = self._make_client()
        client.download_dicom_series(
            seriesInstanceUID=missing_ct_ids,
            downloadDir=ct_dir,
            dirTemplate='%SeriesInstanceUID',
        )

    # --- Helpers ---

    def _load_manifest(self):
        manifest_path = os.path.join(self._data_path, self._MANIFEST_FILE)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f'Manifest not found at {manifest_path}. Run sync_data() first.'
            )
        return pd.read_csv(manifest_path, header=0)

    @staticmethod
    def _make_client():
        pd.set_option('mode.chained_assignment', None)
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', pd.errors.ChainedAssignmentError)
        return index.IDCClient()
