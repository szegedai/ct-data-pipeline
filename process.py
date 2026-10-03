'''
Unified processing entry point for all CT datasets.

Driven by a YAML or TOML configuration file that specifies one or more runs.
Each run is merged with optional shared defaults defined in the same file,
which in turn fall back to built-in defaults for any unspecified field.

Usage:
    python process.py configs/default.yaml
    python process.py configs/default.toml
    python process.py my_run.yaml

Minimal config (YAML):
    runs:
      - dataset: lidc_idri
        output:  nodule

Full config with shared defaults and multiple runs (YAML):
    defaults:
      raw_data_path:  ../data/raw
      save_format:    numpy
      compress:       false
      label_type:     instance
      volume_filter: true

    runs:
      - dataset: lidc_idri
        output:  nodule
        nodule_save_mode: 3d

      - dataset: nsclc_radiomics
        output:        nodule,roi
        nodule_save_mode: 3d
        roi_save_mode: 2d

See configs/ for full YAML and TOML examples with all available options documented.
'''

import argparse
import csv
import logging
import logging.handlers
import multiprocessing as mp
import os
import traceback
from collections import Counter
from datetime import datetime
from multiprocessing import current_process

import torch
from tqdm import tqdm

from ct_data_management.acquisition import (
    IDCFileSystemDataManager,
    LIDC_IDRI_INFO,
    NSCLC_RADIOMICS_INFO,
    NLST_RADIOLOGIST_INFO,
    NLST_AI_INFO,
)
from ct_data_management.datasets import dataset_priority
from ct_data_management.processing.pipeline import PipelineStack
from ct_data_management.processing.readers import DICOMFileSystemReader
from ct_data_management.processing.transforms import (
    IDGenerator,
    FilterSegmentsTransform,
    OrientTransform,
    ResampleTransform,
    LungDilationTransform,
    ROICropTransform,
    MergeSegmentsTransform,
    HUClipAndNormTransform,
    NoduleStatsTransform,
    RequireNodulesTransform,
    NoduleVolumeFilterTransform,
    NoduleHUFilterTransform,
    NoduleEntropyFilterTransform,
    ComputeROITransform,
    ToDeviceTransform,
    NoduleInstanceSegTransform,
)
from ct_data_management.processing.writers import (
    NPZWriter,
    NIfTIWriter,
    NoduleCatalogWriter,
    assemble_catalog,
    assemble_dropped,
    catalog_part_path,
    dropped_report_path,
    remove_catalog_part,
    remove_dropped_part,
    write_dropped_part,
)
from ct_data_management.processing.utils import InteractiveViewer, TimePipelinePart



# Per-dataset configuration: which DICOM label patterns map to each segmentation type.
# lung_labels=None means the dataset has no lung segmentation.
DATASET_CONFIGS = {
    'lidc_idri':        LIDC_IDRI_INFO,
    'nsclc_radiomics':  NSCLC_RADIOMICS_INFO,
    'nlst_radiologist': NLST_RADIOLOGIST_INFO,
    'nlst_ai':          NLST_AI_INFO,
}

DATASET_SEG_CONFIGS = {
    'lidc_idri':        {'nodule_labels': ['nodule'],   'lung_labels': None},
    'nsclc_radiomics':  {'nodule_labels': ['neoplasm'], 'lung_labels': ['lung']},
    'nlst_radiologist': {'nodule_labels': ['nodule'],   'lung_labels': ['lung']},
    'nlst_ai':          {'nodule_labels': ['nodule'],   'lung_labels': ['lung']},
}

# --- Worker process state (process-local globals) ---

_worker_pipeline  = None
_worker_log_queue = None


def _worker_init(n_gpus, log_queue, pipeline_kwargs):
    global _worker_pipeline, _worker_log_queue

    worker_idx = current_process()._identity[0] - 1  # 0-based
    device = f'cuda:{worker_idx % n_gpus}' if n_gpus > 0 else 'cpu'

    _worker_log_queue = log_queue
    _worker_pipeline  = build_pipeline(device=device, **pipeline_kwargs)

    # Route all log records in this worker to the shared queue in the main process.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(log_queue))


def _process_series(args):
    ct_path, seg_path_list = args
    series_id = os.path.basename(ct_path)
    logger = logging.getLogger('pipeline')
    try:
        _worker_pipeline({'ct': ct_path, 'seg_list': seg_path_list}, {})
        logger.debug(f'OK: {series_id}')
        return {'series_id': series_id, 'ok': True}
    except Exception as e:
        logger.error(f'FAILED: {series_id}')
        logger.error(f'  CT:   {ct_path}')
        logger.error(f'  SEG:  {seg_path_list}')
        logger.error(f'  {type(e).__name__}: {e}')
        logger.debug(traceback.format_exc())
        return {'series_id': series_id, 'ok': False,
                'error_type': type(e).__name__, 'error': str(e)}


# --- Pipeline builder ---

def build_pipeline(
    *,
    save_path: str,
    dataset: str,
    nodule_labels: list,
    lung_labels: list,
    outputs: list,
    nodule_save_modes: list,
    lung_save_modes: list,
    roi_save_modes: list,
    save_formats: list,
    compress: bool,
    device: str,
    viewer: bool,
    label_types: list,
    lung_dilations: int,
    roi_crop: bool,
    roi_padding: int,
    volume_filter: bool,
    volume_min: float,
    volume_max: float,
    hu_filter: bool,
    hu_min: float,
    hu_max: float,
    entropy_filter: bool,
    entropy_min: float,
    entropy_max: float,
    hu_clip_min: float,
    hu_clip_max: float,
    catalog_path: str,
):
    parts = [DICOMFileSystemReader(return_headers=True, dtype=torch.float16)]

    if device != 'cpu':
        parts.append(ToDeviceTransform(device=device))

    parts += [
        IDGenerator(),
        FilterSegmentsTransform(nodule_labels=nodule_labels, lung_labels=lung_labels),
        OrientTransform(),
        ResampleTransform(),
    ]

    if lung_dilations > 0:
        parts.append(LungDilationTransform(n_dilations=lung_dilations))

    if roi_crop:
        parts.append(ROICropTransform(padding=roi_padding))

    parts += [
        MergeSegmentsTransform(),
        HUClipAndNormTransform(clip_min=hu_clip_min, clip_max=hu_clip_max),
    ]

    if device != 'cpu':
        parts.append(ToDeviceTransform(device='cpu'))

    has_nodule = 'nodule' in outputs

    # NoduleStatsTransform always runs when nodule output is requested so that
    # filters and the catalog writer share a single connected-component analysis.
    if has_nodule:
        parts.append(NoduleStatsTransform(hu_clip_min=hu_clip_min, hu_clip_max=hu_clip_max))
        parts.append(RequireNodulesTransform())

        if volume_filter:
            parts.append(NoduleVolumeFilterTransform(min_volume=volume_min, max_volume=volume_max))

        if hu_filter:
            parts.append(NoduleHUFilterTransform(
                min_hu=hu_min,
                max_hu=hu_max,
                hu_clip_min=hu_clip_min,
                hu_clip_max=hu_clip_max,
            ))

        if entropy_filter:
            parts.append(NoduleEntropyFilterTransform(min_entropy=entropy_min, max_entropy=entropy_max))

    if viewer:
        parts.append(InteractiveViewer())
        return PipelineStack(parts)

    has_lung     = 'lung'   in outputs
    has_roi      = 'roi'    in outputs
    has_semantic = 'semantic' in label_types
    has_instance = 'instance' in label_types

    # ct_written tracks which (save_mode, save_format) pairs have already had
    # their CT directory assigned, so CT is written exactly once per pair.
    ct_written = set()

    def make_writer(seg_dir, save_mode, save_format, seg_key):
        pair   = (save_mode, save_format)
        ct_dir = None if pair in ct_written else os.path.join(save_path, f'ct_{save_mode}')
        ct_written.add(pair)
        if save_format == 'nifti':
            return NIfTIWriter(ct_dir, seg_dir, save_mode=save_mode,
                               compress=compress, seg_key=seg_key)
        return NPZWriter(ct_dir, seg_dir, save_mode=save_mode,
                         compress=compress, seg_key=seg_key)

    # --- Semantic writers (nodule + lung use binary masks before instance transform) ---
    if has_semantic:
        if has_nodule:
            for sm in nodule_save_modes:
                seg_dir = os.path.join(save_path, f'nodule_sem_seg_{sm}')
                for sf in save_formats:
                    parts.append(make_writer(seg_dir, sm, sf, 'nodule_seg'))

        if has_lung:
            for sm in lung_save_modes:
                seg_dir = os.path.join(save_path, f'lung_sem_seg_{sm}')
                for sf in save_formats:
                    parts.append(make_writer(seg_dir, sm, sf, 'lung_seg'))

    # --- ROI writers (union of lung + nodule; must run before instance transform) ---
    if has_roi:
        parts.append(ComputeROITransform())
        for sm in roi_save_modes:
            seg_dir = os.path.join(save_path, f'roi_sem_seg_{sm}')
            for sf in save_formats:
                parts.append(make_writer(seg_dir, sm, sf, 'roi_seg'))

    # --- Instance writers (nodule_seg is overwritten with instance labels here) ---
    if has_instance and has_nodule:
        parts.append(NoduleInstanceSegTransform())
        for sm in nodule_save_modes:
            seg_dir = os.path.join(save_path, f'nodule_inst_seg_{sm}')
            for sf in save_formats:
                parts.append(make_writer(seg_dir, sm, sf, 'nodule_seg'))

    # --- Nodule catalog (reads from params['nodule_components'], not nodule_seg) ---
    if catalog_path is not None and has_nodule:
        parts.append(NoduleCatalogWriter(
            catalog_path=catalog_path,
            dataset=dataset,
        ))

    return PipelineStack(parts)


# --- Logging ---

class _TqdmLoggingHandler(logging.StreamHandler):
    """Routes console log output through tqdm.write() to avoid disrupting the progress bar."""
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: str, dataset: str, outputs: list, log_queue=None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_tag = '_'.join(outputs)
    log_file = os.path.join(log_dir, f'{dataset}_{output_tag}_{timestamp}.log')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s'))

    ch = _TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))

    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in logger.handlers[:]:
        h.close()
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'Logging to {log_file}')

    if log_queue is not None:
        listener = logging.handlers.QueueListener(log_queue, fh, ch, respect_handler_level=True)
        return logger, listener

    return logger, None


# --- Helpers ---

def already_processed(save_path: str, series_id: str, outputs: list,
                      nodule_save_modes: list, lung_save_modes: list, roi_save_modes: list,
                      save_formats: list, label_types: list, compress: bool) -> bool:
    save_modes_by_output = {'nodule': nodule_save_modes,
                            'lung':   lung_save_modes,
                            'roi':    roi_save_modes}

    # Check CT dirs — CT is written once per (save_mode, save_format) pair across all outputs.
    ct_pairs_needed = {(sm, sf)
                       for output in outputs
                       for sm in save_modes_by_output[output]
                       for sf in save_formats}
    for save_mode, save_format in ct_pairs_needed:
        ext  = '.nii.gz' if (save_format == 'nifti' and compress) else \
               '.nii'    if  save_format == 'nifti'               else '.npz'
        stem = series_id if save_mode == '3d' else f'{series_id}_0000'
        ct_dir = os.path.join(save_path, f'ct_{save_mode}')
        if not os.path.exists(os.path.join(ct_dir, stem + ext)):
            return False

    # Check per-output segmentation dirs using each output's own save modes.
    for output in outputs:
        for save_mode in save_modes_by_output[output]:
            for save_format in save_formats:
                ext  = '.nii.gz' if (save_format == 'nifti' and compress) else \
                       '.nii'    if  save_format == 'nifti'               else '.npz'
                stem = series_id if save_mode == '3d' else f'{series_id}_0000'

                if output == 'nodule':
                    if 'semantic' in label_types:
                        seg_dir = os.path.join(save_path, f'nodule_sem_seg_{save_mode}')
                        if not os.path.exists(os.path.join(seg_dir, stem + ext)):
                            return False
                    if 'instance' in label_types:
                        seg_dir = os.path.join(save_path, f'nodule_inst_seg_{save_mode}')
                        if not os.path.exists(os.path.join(seg_dir, stem + ext)):
                            return False
                elif output == 'lung':
                    seg_dir = os.path.join(save_path, f'lung_sem_seg_{save_mode}')
                    if not os.path.exists(os.path.join(seg_dir, stem + ext)):
                        return False
                elif output == 'roi':
                    seg_dir = os.path.join(save_path, f'roi_sem_seg_{save_mode}')
                    if not os.path.exists(os.path.join(seg_dir, stem + ext)):
                        return False
    return True


def order_runs(cfgs: list) -> list:
    """Sort run configs by dataset priority (stable for runs of equal priority).

    Runs must execute in priority order for ``claim_series`` to give each CT
    series to the highest-priority dataset that annotates it.
    """
    return sorted(cfgs, key=lambda cfg: dataset_priority(cfg.get('dataset', '')))


def claim_series(owners: dict, dataset: str, all_paths: list) -> tuple[list, Counter]:
    """Keep the CT series of ``all_paths`` that no earlier run has claimed.

    ``owners`` maps series UID → dataset and is shared by all runs writing to the
    same output directory.  Because runs execute in priority order, a series that
    appears in several manifests — an NLST scan with both a radiologist-corrected
    and an AI-generated annotation — is processed exactly once, by the
    highest-priority dataset, instead of being overwritten by a later run.

    Ownership follows the manifests, not the processing outcome: if the owning run
    fails on a series, the series is not handed to a lower-priority dataset.

    Returns the owned (ct_path, seg_path_list) pairs and a Counter of the series
    left to other datasets, keyed by the owning dataset.
    """
    owned, deferred = [], Counter()
    for ct_path, seg_path_list in all_paths:
        owner = owners.setdefault(os.path.basename(ct_path), dataset)
        if owner == dataset:
            owned.append((ct_path, seg_path_list))
        else:
            deferred[owner] += 1
    return owned, deferred


# --- Config ---

DEFAULTS = {
    # Paths
    'raw_data_path':       '../data/raw',
    'processed_data_path': '../data/processed',
    'save_path':           None,        # overrides processed_data_path/<dataset> when set
    # Output
    'output':              'nodule',    # comma-separated or list: nodule, lung, roi
    'save_format':         'numpy',     # comma-separated or list: numpy, nifti
    'compress':            True,
    'label_type':          'semantic',  # comma-separated or list: semantic, instance
    'save_mode':           '3d',        # global fallback; comma-separated or list: 3d, 2d
    'nodule_save_mode':    None,        # overrides save_mode for nodule output
    'lung_save_mode':      None,        # overrides save_mode for lung output
    'roi_save_mode':       None,        # overrides save_mode for roi output
    # Execution
    'sync':                False,
    'download_only':       False,
    'reprocess':           False,
    'cpu':                 False,
    'workers':             None,        # defaults to number of available GPUs (min 1)
    'viewer':              False,
    # Hyperparameters
    'lung_dilations':      0,           # dilation iterations on lung mask (0 = disabled)
    'roi_crop':            False,
    'roi_padding':         1,           # voxels of padding around ROI crop bounding box
    'volume_filter':       True,
    'volume_min':          5.0,
    'volume_max':          float('inf'),
    'hu_filter':           False,
    'hu_min':              float('-inf'),
    'hu_max':              float('inf'),
    'entropy_filter':      False,
    'entropy_min':         float('-inf'),
    'entropy_max':         float('inf'),
    # HU windowing applied before normalisation
    'hu_clip_min':         -1000.0,
    'hu_clip_max':         400.0,
    # Nodule catalog output path. Defaults to <save_path>/nodule_catalog.csv.
    # Each processed series writes its rows to <catalog>_parts/<series_uid>.csv;
    # the catalog itself is rebuilt from these fragments at the end of every
    # invocation and covers all series owned by its runs, including series
    # skipped by the already_processed() check (reprocess=false).
    'catalog_path':        None,
}


def load_config(path: str) -> dict:
    """Load a YAML or TOML config file and return the raw dict."""
    if path.endswith(('.yaml', '.yml')):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    if path.endswith('.toml'):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib     # pip install tomli on Python < 3.11
        with open(path, 'rb') as f:
            return tomllib.load(f)
    raise SystemExit(f"Unsupported config format '{path}'. Use .yaml, .yml, or .toml.")


def parse_overrides(items: list) -> dict:
    """Parse ``--set KEY=VALUE`` items; values use YAML syntax (``workers=6``, ``sync=false``)."""
    import yaml
    overrides = {}
    for item in items:
        key, sep, value = item.partition('=')
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f'--set expects KEY=VALUE, got "{item}".')
        if key not in DEFAULTS:
            raise SystemExit(f'--set: unknown option "{key}". Choose from {sorted(DEFAULTS)}.')
        overrides[key] = yaml.safe_load(value) if value.strip() else None
    return overrides


def _to_list(val) -> list:
    """Normalise a string or list config value to a deduplicated list."""
    if isinstance(val, list):
        return list(dict.fromkeys(str(v).strip() for v in val))
    return list(dict.fromkeys(v.strip() for v in str(val).split(',')))


# --- Run ---

def run_one(cfg: dict, run_index: int, total_runs: int, owners_by_path: dict) -> dict | None:
    """Process one run.

    ``owners_by_path`` maps an absolute save path to the series → dataset
    ownership shared by all runs writing there (see ``claim_series``).

    Returns ``{'save_path': ..., 'catalog_path': ... or None, 'series': [...]}``
    with the series owned by this run (``catalog_path`` is None when the run keeps
    no nodule catalog), or ``None`` when nothing was processed (download only,
    viewer).
    """
    VALID_OUTPUTS      = {'nodule', 'lung', 'roi'}
    VALID_SAVE_MODES   = {'3d', '2d'}
    VALID_SAVE_FORMATS = {'numpy', 'nifti'}
    VALID_LABEL_TYPES  = {'semantic', 'instance'}

    dataset = cfg.get('dataset')
    if not dataset:
        raise SystemExit(f'Run {run_index + 1}: "dataset" is required.')
    if dataset not in DATASET_CONFIGS:
        raise SystemExit(f'Run {run_index + 1}: unknown dataset "{dataset}". '
                         f'Choose from {sorted(DATASET_CONFIGS)}.')

    outputs      = _to_list(cfg['output'])
    save_formats = _to_list(cfg['save_format'])
    label_types  = _to_list(cfg['label_type'])
    save_modes   = _to_list(cfg['save_mode'])

    def check(vals, valid, field):
        bad = [v for v in vals if v not in valid]
        if bad:
            raise SystemExit(f'Run {run_index + 1} ({dataset}): '
                             f'invalid {field!r} value(s) {bad}. '
                             f'Choose from {sorted(valid)}.')

    check(outputs,      VALID_OUTPUTS,       'output')
    check(save_formats, VALID_SAVE_FORMATS,  'save_format')
    check(label_types,  VALID_LABEL_TYPES,   'label_type')
    check(save_modes,   VALID_SAVE_MODES,    'save_mode')

    def resolve_mode(val):
        if val is None:
            return list(dict.fromkeys(save_modes))
        modes = _to_list(val)
        check(modes, VALID_SAVE_MODES, 'save_mode override')
        return modes

    nodule_save_modes = resolve_mode(cfg.get('nodule_save_mode'))
    lung_save_modes   = resolve_mode(cfg.get('lung_save_mode'))
    roi_save_modes    = resolve_mode(cfg.get('roi_save_mode'))

    outputs     = sorted(set(outputs),     key=lambda o: {'nodule': 0, 'lung': 1, 'roi': 2}[o])
    label_types = sorted(set(label_types), key=lambda t: 0 if t == 'semantic' else 1)

    seg_config    = DATASET_SEG_CONFIGS[dataset]
    nodule_labels = seg_config['nodule_labels']
    lung_labels   = seg_config['lung_labels']

    def err(msg):
        raise SystemExit(f'Run {run_index + 1} ({dataset}): {msg}')

    if 'lung' in outputs and lung_labels is None:
        err(f'output "lung" requires lung annotations, which "{dataset}" does not have.')
    if 'roi' in outputs and lung_labels is None:
        err(f'output "roi" (lung ∪ nodule) requires lung annotations, which "{dataset}" does not have.')
    if 'instance' in label_types and 'nodule' not in outputs:
        err('"instance" label_type requires output to include "nodule".')
    if (cfg['roi_crop'] or cfg['lung_dilations'] > 0) and lung_labels is None:
        err('"roi_crop" and "lung_dilations" require lung annotations.')
    if any(cfg[f] for f in ('volume_filter', 'hu_filter', 'entropy_filter')) and 'nodule' not in outputs:
        err('"volume_filter", "hu_filter", and "entropy_filter" require output to include "nodule".')

    raw_path  = os.path.join(cfg['raw_data_path'], dataset)
    save_path = cfg['save_path'] or os.path.join(cfg['processed_data_path'], dataset)
    log_dir   = os.path.join(save_path, 'logs')

    cpu       = cfg['cpu'] or cfg['viewer']
    n_gpus    = 0 if cpu else torch.cuda.device_count()
    n_workers = 1 if cpu else (cfg['workers'] if cfg['workers'] is not None else max(n_gpus, 1))

    data_manager = IDCFileSystemDataManager(raw_path, DATASET_CONFIGS[dataset])

    catalog_path   = cfg['catalog_path'] or os.path.join(save_path, 'nodule_catalog.csv')
    catalog_active = 'nodule' in outputs and not cfg['viewer']
    dropped_path   = dropped_report_path(save_path)

    pipeline_kwargs = dict(
        save_path=save_path,
        dataset=dataset,
        nodule_labels=nodule_labels,
        lung_labels=lung_labels,
        outputs=outputs,
        nodule_save_modes=nodule_save_modes,
        lung_save_modes=lung_save_modes,
        roi_save_modes=roi_save_modes,
        save_formats=save_formats,
        compress=cfg['compress'],
        viewer=cfg['viewer'],
        label_types=label_types,
        lung_dilations=cfg['lung_dilations'],
        roi_crop=cfg['roi_crop'],
        roi_padding=cfg['roi_padding'],
        volume_filter=cfg['volume_filter'],
        volume_min=cfg['volume_min'],
        volume_max=cfg['volume_max'],
        hu_filter=cfg['hu_filter'],
        hu_min=cfg['hu_min'],
        hu_max=cfg['hu_max'],
        entropy_filter=cfg['entropy_filter'],
        entropy_min=cfg['entropy_min'],
        entropy_max=cfg['entropy_max'],
        hu_clip_min=cfg['hu_clip_min'],
        hu_clip_max=cfg['hu_clip_max'],
        catalog_path=catalog_path,
    )

    if n_workers > 1:
        log_queue = mp.Manager().Queue()
        logger, listener = setup_logging(log_dir, dataset, outputs, log_queue=log_queue)
        listener.start()
    else:
        logger, _ = setup_logging(log_dir, dataset, outputs)

    prefix = f'[{run_index + 1}/{total_runs}] ' if total_runs > 1 else ''
    logger.info(f'{prefix}Dataset: {dataset}  Output: {outputs}  '
                f'Save formats: {save_formats}  Label types: {label_types}')
    logger.info(f'Save modes — nodule: {nodule_save_modes}  '
                f'lung: {lung_save_modes}  roi: {roi_save_modes}')
    logger.info(f'Raw path:  {raw_path}')
    logger.info(f'Save path: {save_path}')
    logger.info(f'Workers:   {n_workers}  GPUs: {n_gpus}')

    if cfg['sync'] or cfg['download_only']:
        logger.info('Syncing raw data...')
        data_manager.sync_data()

    if cfg['download_only']:
        logger.info('Download complete. Skipping processing (download_only: true).')
        if n_workers > 1:
            listener.stop()
        return None

    owners = owners_by_path.setdefault(os.path.abspath(save_path), {})
    all_paths, deferred = claim_series(owners, dataset, list(data_manager.get_paths()))
    for owner, n in sorted(deferred.items()):
        logger.info(f'{n} series are also annotated by higher-priority dataset "{owner}" '
                    f'and are processed there, not as "{dataset}".')
    n_total = len(all_paths)

    if not cfg['reprocess']:
        tasks = [
            (ct_path, seg_path_list)
            for ct_path, seg_path_list in all_paths
            if not already_processed(
                save_path, os.path.basename(ct_path),
                outputs,
                nodule_save_modes, lung_save_modes, roi_save_modes,
                save_formats, label_types, cfg['compress'],
            )
        ]
        n_skipped = n_total - len(tasks)
    else:
        tasks     = list(all_paths)
        n_skipped = 0

    n_tasks = len(tasks)
    n_processed = n_failed = 0

    logger.info(f'Found {n_total} CT series. {n_skipped} already processed, {n_tasks} to run.')

    if catalog_active:
        # A series processed now gets a fresh catalog fragment; removing the old
        # one first means a series that fails or is dropped leaves no stale rows.
        for ct_path, _ in tasks:
            remove_catalog_part(catalog_path, os.path.basename(ct_path))

        task_ids = {os.path.basename(p) for p, _ in tasks}
        no_part  = [uid for uid in (os.path.basename(p) for p, _ in all_paths)
                    if uid not in task_ids
                    and not os.path.exists(catalog_part_path(catalog_path, uid))]
        if no_part:
            logger.warning(
                f'{len(no_part)} already-processed series have no nodule catalog fragment '
                f'(written by an older pipeline version?) and will be missing from the catalog. '
                f'Re-run with "reprocess: true" or into an empty save_path.'
            )

    def record_failure(series_id, error_type, error):
        if not cfg['viewer']:
            write_dropped_part(dropped_path, series_id, dataset, error_type, error)

    if not cfg['viewer']:
        # Series processed now get a fresh entry in the dropped-series report if
        # they fail again, and none if they succeed.
        for ct_path, _ in tasks:
            remove_dropped_part(dropped_path, os.path.basename(ct_path))

    if n_workers > 1:
        pool_init_args = (n_gpus, log_queue, pipeline_kwargs)
        try:
            with mp.Pool(n_workers, initializer=_worker_init, initargs=pool_init_args) as pool:
                for result in tqdm(pool.imap_unordered(_process_series, tasks), total=n_tasks):
                    if result['ok']:
                        n_processed += 1
                    else:
                        n_failed += 1
                        record_failure(result['series_id'], result['error_type'], result['error'])
        except mp.ProcessError as e:
            logger.error(f'Worker process lost: {e}')
        finally:
            listener.stop()
    else:
        device   = 'cuda:0' if n_gpus > 0 else 'cpu'
        pipeline = build_pipeline(device=device, **pipeline_kwargs)
        for ct_path, seg_path_list in tqdm(tasks, total=n_tasks):
            series_id = os.path.basename(ct_path)
            try:
                pipeline({'ct': ct_path, 'seg_list': seg_path_list}, {})
                n_processed += 1
                logger.debug(f'OK: {series_id}')
            except Exception as e:
                n_failed += 1
                logger.error(f'FAILED: {series_id}')
                logger.error(f'  CT:   {ct_path}')
                logger.error(f'  SEG:  {seg_path_list}')
                logger.error(f'  {type(e).__name__}: {e}')
                logger.debug(traceback.format_exc())
                record_failure(series_id, type(e).__name__, str(e))

    logger.info(
        f'Done. Total: {n_total}  Processed: {n_processed}  '
        f'Skipped: {n_skipped}  Failed: {n_failed}'
    )

    if cfg['viewer']:
        return None
    return {'save_path': save_path,
            'catalog_path': catalog_path if catalog_active else None,
            'series': [os.path.basename(p) for p, _ in all_paths]}


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(
        description='CT nodule dataset processing pipeline.',
        epilog='See configs/ for example YAML and TOML configuration files.',
    )
    parser.add_argument('config', help='Path to a YAML (.yaml/.yml) or TOML (.toml) config file.')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a config option for every run, e.g. --set workers=6 '
                             '--set save_path=/data/release. Can be repeated.')
    args = parser.parse_args()

    raw            = load_config(args.config)
    global_defaults = raw.get('defaults', {})
    runs           = raw.get('runs', [])
    overrides      = parse_overrides(args.overrides)

    if not runs:
        raise SystemExit('Config file must define at least one entry under "runs".')

    mp.set_start_method('spawn', force=True)
    torch.set_grad_enabled(False)

    cfgs    = [{**DEFAULTS, **global_defaults, **run_override, **overrides} for run_override in runs]
    ordered = order_runs(cfgs)
    if ordered != cfgs:
        print('Runs reordered by dataset priority: '
              + ', '.join(str(cfg.get('dataset')) for cfg in ordered))

    owners_by_path = {}   # save path → {series_uid: owning dataset}
    catalogs       = {}   # catalog path → series owned by the runs that write it
    save_paths     = {}   # save path → series owned by the runs that write there
    for i, cfg in enumerate(ordered):
        result = run_one(cfg, run_index=i, total_runs=len(ordered), owners_by_path=owners_by_path)
        if result is not None:
            save_paths.setdefault(os.path.abspath(result['save_path']), []).extend(result['series'])
            if result['catalog_path'] is not None:
                catalogs.setdefault(result['catalog_path'], []).extend(result['series'])

    logger = logging.getLogger('pipeline')
    for catalog_path, series in catalogs.items():
        summary = assemble_catalog(catalog_path, series)
        logger.info(f'Nodule catalog written to {catalog_path}: {summary["rows"]} nodules '
                    f'in {summary["series"]} series.')
        if summary['missing']:
            logger.info(f'{len(summary["missing"])} series have no catalog rows '
                        f'(dropped by the nodule checks or failed; see the run logs).')

    for save_path, series in save_paths.items():
        report_path = dropped_report_path(save_path)
        summary     = assemble_dropped(report_path, series)
        logger.info(f'Dropped-series report written to {report_path}: '
                    f'{summary["rows"]} of {len(series)} series dropped.')
        with open(report_path, newline='') as f:
            counts = Counter((row['dataset'], row['reason']) for row in csv.DictReader(f))
        for (dataset, reason), n in sorted(counts.items()):
            logger.info(f'  {dataset:<18} {reason:<20} {n}')
        if any(reason != 'DataAnomalyError' for _, reason in counts):
            logger.warning('Some series failed for reasons other than the data checks '
                           '(reason != DataAnomalyError). Re-running the same command '
                           'retries them; failures caused by the source data are listed '
                           'in configs/known_failures.csv.')


if __name__ == '__main__':
    main()
