'''
Lean catalog-only pipeline — no CT or segmentation files are written.

Driven by the same YAML or TOML config file used by process.py.  For each
run entry, only the fields relevant to cataloging are used:

    dataset, raw_data_path, processed_data_path, save_path,
    hu_clip_min, hu_clip_max, catalog_path

All output, save_format, label_type, and writer settings are ignored.
No filters (volume, HU, entropy) are applied so every annotated nodule
component is recorded, including small artefacts — the point is to analyse
the full distribution before choosing filter thresholds.

Usage:
    python catalog_nodules.py configs/default.toml
    python catalog_nodules.py configs/default.yaml
'''

import argparse
import logging
import os
import sys
import traceback

import torch
from tqdm import tqdm

from ct_data_management.acquisition import IDCFileSystemDataManager
from ct_data_management.processing.pipeline import PipelineStack
from ct_data_management.processing.readers import DICOMFileSystemReader
from ct_data_management.processing.transforms import (
    IDGenerator,
    FilterSegmentsTransform,
    OrientTransform,
    ResampleTransform,
    MergeSegmentsTransform,
    HUClipAndNormTransform,
    NoduleStatsTransform,
)
from ct_data_management.processing.writers import NoduleCatalogWriter
from process import load_config, DEFAULTS, DATASET_CONFIGS, DATASET_SEG_CONFIGS


def build_pipeline(
    dataset: str,
    catalog_path: str,
    hu_clip_min: float,
    hu_clip_max: float,
) -> PipelineStack:
    seg_config = DATASET_SEG_CONFIGS[dataset]
    return PipelineStack([
        DICOMFileSystemReader(return_headers=True, dtype=torch.float16),
        IDGenerator(),
        FilterSegmentsTransform(
            nodule_labels=seg_config['nodule_labels'],
            lung_labels=None,   # lung mask not needed without ROI/dilation transforms
        ),
        OrientTransform(),
        ResampleTransform(),
        MergeSegmentsTransform(),
        HUClipAndNormTransform(clip_min=hu_clip_min, clip_max=hu_clip_max),
        NoduleStatsTransform(hu_clip_min=hu_clip_min, hu_clip_max=hu_clip_max),
        NoduleCatalogWriter(
            catalog_path=catalog_path,
            dataset=dataset,
        ),
    ])


def run_one(cfg: dict, run_index: int, total_runs: int) -> None:
    logger = logging.getLogger('catalog_nodules')

    dataset = cfg.get('dataset')
    if not dataset:
        raise SystemExit(f'Run {run_index + 1}: "dataset" is required.')
    if dataset not in DATASET_CONFIGS:
        raise SystemExit(
            f'Run {run_index + 1}: unknown dataset "{dataset}". '
            f'Choose from {sorted(DATASET_CONFIGS)}.'
        )

    raw_path  = os.path.join(cfg['raw_data_path'], dataset)
    save_path = cfg['save_path'] or os.path.join(cfg['processed_data_path'], dataset)
    catalog_path = cfg['catalog_path'] or os.path.join(save_path, 'nodule_catalog.csv')
    hu_clip_min  = cfg['hu_clip_min']
    hu_clip_max  = cfg['hu_clip_max']

    prefix = f'[{run_index + 1}/{total_runs}] ' if total_runs > 1 else ''
    logger.info('%sDataset: %s', prefix, dataset)
    logger.info('Raw path:    %s', raw_path)
    logger.info('Catalog out: %s', catalog_path)

    data_manager = IDCFileSystemDataManager(raw_path, DATASET_CONFIGS[dataset])

    if cfg.get('sync') or cfg.get('download_only'):
        logger.info('Syncing raw data...')
        data_manager.sync_data()

    if cfg.get('download_only'):
        logger.info('Download complete. Skipping processing (download_only: true).')
        return

    all_paths    = list(data_manager.get_paths())
    logger.info('Found %d CT series.', len(all_paths))

    pipeline = build_pipeline(
        dataset=dataset,
        catalog_path=catalog_path,
        hu_clip_min=hu_clip_min,
        hu_clip_max=hu_clip_max,
    )

    n_ok = n_fail = 0
    for ct_path, seg_path_list in tqdm(all_paths, total=len(all_paths)):
        series_id = os.path.basename(ct_path)
        try:
            pipeline({'ct': ct_path, 'seg_list': seg_path_list}, {})
            n_ok += 1
        except Exception as e:
            n_fail += 1
            logger.error('FAILED: %s — %s: %s', series_id, type(e).__name__, e)
            logger.debug(traceback.format_exc())

    logger.info('Done. OK: %d  Failed: %d', n_ok, n_fail)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s %(message)s',
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description='Build a nodule catalog CSV without writing any processed files.',
        epilog='Uses the same config format as process.py. No nodule filters are applied.',
    )
    parser.add_argument('config', help='Path to a YAML (.yaml/.yml) or TOML (.toml) config file.')
    args = parser.parse_args()

    raw            = load_config(args.config)
    global_defaults = raw.get('defaults', {})
    runs           = raw.get('runs', [])

    if not runs:
        raise SystemExit('Config file must define at least one entry under "runs".')

    torch.set_grad_enabled(False)

    for i, run_override in enumerate(runs):
        cfg = {**DEFAULTS, **global_defaults, **run_override}
        run_one(cfg, run_index=i, total_runs=len(runs))


if __name__ == '__main__':
    main()
