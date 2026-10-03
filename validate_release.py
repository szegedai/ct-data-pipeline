'''
Consistency checks for a processed release before it is published.

Checks that the processed files, the nodule catalog and the train/test split
describe the same set of scans, that the split is patient-isolated and keeps
non-eligible (AI-only) scans out of the test set, and that every scan missing
from the release is listed in the dropped-series report written by process.py,
either dropped by the data checks or listed in configs/known_failures.csv.
Optionally, it also checks that the nodule masks on disk match the catalog and,
against the raw manifests, that every scan was processed with the annotation of
its highest-priority dataset and is either released or reported as dropped.

Usage:
    python validate_release.py ../data/processed/release
    python validate_release.py ../data/processed/release --raw ../data/raw --check-masks all --workers 16

Arguments:
    data_dir         Directory with ct_3d/, nodule_sem_seg_3d/, ... (the save_path of process.py)
    --split          Split CSV (default: <data_dir>/split.csv)
    --catalog        Nodule catalog CSV (default: <data_dir>/nodule_catalog.csv)
    --dropped        Dropped-series report (default: <data_dir>/dropped_series.csv)
    --known-failures Series that fail because of their source data (default:
                     configs/known_failures.csv next to this script)
    --raw            raw_data_path of process.py; enables the ownership and
                     completeness checks against the per-dataset manifests
    --check-masks    'all', a number of randomly chosen scans (always including every
                     nlst_radiologist scan), or 0 to skip (default: 200)
    --min-volume     Smallest nodule volume allowed in the catalog in mm³ (default: 5)

Exits with status 1 if any check fails.  The summary at the end lists the
per-dataset and per-split counts that are reported in the paper.
'''

import argparse
import os
import random
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ct_data_management.datasets import TEST_ELIGIBLE_DATASETS, dataset_priority

SPLIT_VALUES = {'train', 'test', 'excluded'}
DATA_CHECK_REASON = 'DataAnomalyError'     # series dropped by the pipeline's data checks
DEFAULT_KNOWN_FAILURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'configs', 'known_failures.csv')
_SLICE_RE    = re.compile(r'^(?P<uid>.+)_(?P<idx>\d{4})\.npz$')


class Report:
    def __init__(self):
        self.errors = 0

    def ok(self, msg):
        print(f'  OK     {msg}')

    def fail(self, msg, examples=None):
        self.errors += 1
        print(f'  ERROR  {msg}')
        for e in ([] if examples is None else list(examples))[:5]:
            print(f'           e.g. {e}')

    def check(self, condition, ok_msg, fail_msg, examples=None):
        if condition:
            self.ok(ok_msg)
        else:
            self.fail(fail_msg, examples)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _npz_ids(directory):
    if not os.path.isdir(directory):
        return None
    return {f[:-4] for f in os.listdir(directory) if f.endswith('.npz')}


def _slice_counts(directory):
    """Map series UID → sorted slice indices for a 2-D output directory."""
    if not os.path.isdir(directory):
        return None
    slices = defaultdict(list)
    for f in os.listdir(directory):
        m = _SLICE_RE.match(f)
        if m:
            slices[m['uid']].append(int(m['idx']))
    return {uid: sorted(idx) for uid, idx in slices.items()}


def _npz_shape(path, key='data'):
    """Shape of one array in an .npz file, read from its header only."""
    with zipfile.ZipFile(path) as z, z.open(f'{key}.npy') as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(fh)
    return shape


def _mask_components(path):
    """26-connected components of a nodule mask: (sorted voxel counts, values ok)."""
    from scipy import ndimage
    with np.load(path) as f:
        mask = f['data']
    values_ok = bool(np.isin(np.unique(mask), (0, 1)).all())
    labeled, n = ndimage.label(mask[0] > 0, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labeled.ravel())[1:] if n else np.array([], dtype=int)
    return sorted(int(s) for s in sizes), values_ok


def _mask_job(args):
    uid, path = args
    try:
        sizes, values_ok = _mask_components(path)
        return uid, sizes, values_ok, None
    except Exception as e:      # reported as a failed check, not a crash
        return uid, None, None, f'{type(e).__name__}: {e}'


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_catalog(rep, catalog, min_volume):
    print('\nNodule catalog')
    required = {'patient_id', 'series_uid', 'dataset', 'volume_mm3', 'entropy'}
    missing  = required - set(catalog.columns)
    if missing:
        rep.fail(f'catalog is missing columns {sorted(missing)}')
        return
    rep.check(not catalog[list(required)].isna().any().any(),
              'no empty values', 'rows with empty values',
              catalog[catalog[list(required)].isna().any(axis=1)].series_uid)

    labels = catalog.groupby('series_uid')['dataset'].nunique()
    rep.check((labels == 1).all(), 'every scan has a single dataset label',
              f'{int((labels > 1).sum())} scans carry several dataset labels '
              f'(scan processed by more than one dataset)', labels[labels > 1].index)

    pats = catalog.groupby('series_uid')['patient_id'].nunique()
    rep.check((pats == 1).all(), 'every scan belongs to a single patient',
              f'{int((pats > 1).sum())} scans have several patient IDs', pats[pats > 1].index)

    small = catalog[catalog['volume_mm3'] < min_volume]
    rep.check(small.empty, f'all nodule components ≥ {min_volume} mm³',
              f'{len(small)} components below {min_volume} mm³', small.series_uid)


def check_split(rep, split, catalog):
    print('\nTrain/test split')
    rep.check(split['series_uid'].is_unique, 'one row per scan', 'duplicate scans in split',
              split[split['series_uid'].duplicated()].series_uid)
    bad = split[~split['split'].isin(SPLIT_VALUES)]
    rep.check(bad.empty, f'split values within {sorted(SPLIT_VALUES)}',
              f'{len(bad)} rows with unknown split values', bad.split.unique())

    cat_ds = catalog.groupby('series_uid')['dataset'].first()
    s_set, c_set = set(split.series_uid), set(cat_ds.index)
    rep.check(s_set == c_set, f'split and catalog list the same {len(s_set)} scans',
              f'split and catalog differ: {len(s_set - c_set)} only in split, '
              f'{len(c_set - s_set)} only in catalog', sorted(s_set ^ c_set))
    merged = split.join(cat_ds.rename('cat_dataset'), on='series_uid')
    diff = merged[merged['cat_dataset'].notna() & (merged['dataset'] != merged['cat_dataset'])]
    rep.check(diff.empty, 'dataset labels agree with the catalog',
              f'{len(diff)} scans have different dataset labels in split and catalog', diff.series_uid)

    per_patient = split.groupby('patient_id')['split'].agg(set)
    leak = per_patient[per_patient.apply(lambda s: {'train', 'test'} <= s)]
    rep.check(leak.empty, 'no patient in both train and test',
              f'{len(leak)} patients appear in both train and test', leak.index)

    test_bad = split[(split['split'] == 'test') & ~split['dataset'].isin(TEST_ELIGIBLE_DATASETS)]
    rep.check(test_bad.empty, f'test scans only from {sorted(TEST_ELIGIBLE_DATASETS)}',
              f'{len(test_bad)} test scans from non-eligible datasets', test_bad.series_uid)

    excl = split[split['split'] == 'excluded']
    excl_bad = excl[excl['dataset'].isin(TEST_ELIGIBLE_DATASETS)]
    rep.check(excl_bad.empty, 'excluded scans are all non-eligible scans',
              f'{len(excl_bad)} eligible scans marked excluded', excl_bad.series_uid)
    test_patients = set(split.loc[split['split'] == 'test', 'patient_id'])
    orphan = excl[~excl['patient_id'].isin(test_patients)]
    rep.check(orphan.empty, 'excluded scans all belong to test patients',
              f'{len(orphan)} excluded scans of patients without test scans', orphan.series_uid)


def check_files(rep, data_dir, split):
    print('\nProcessed files')
    expected = set(split.series_uid)
    shapes = {}

    ct3d = _npz_ids(os.path.join(data_dir, 'ct_3d'))
    if ct3d is None:
        rep.fail('ct_3d/ not found')
        return None
    rep.check(ct3d == expected, f'ct_3d/ holds exactly the {len(expected)} scans of the split',
              f'ct_3d/ differs from the split: {len(ct3d - expected)} extra, '
              f'{len(expected - ct3d)} missing', sorted(ct3d ^ expected))

    for sub in ('nodule_sem_seg_3d', 'nodule_inst_seg_3d', 'lung_sem_seg_3d'):
        ids = _npz_ids(os.path.join(data_dir, sub))
        if ids is None:
            continue
        rep.check(ids == ct3d, f'{sub}/ matches ct_3d/ ({len(ids)} scans)',
                  f'{sub}/ differs from ct_3d/: {len(ids - ct3d)} extra, {len(ct3d - ids)} missing',
                  sorted(ids ^ ct3d))

    mismatch = []
    for uid in sorted(ct3d & expected):
        ct_shape = _npz_shape(os.path.join(data_dir, 'ct_3d', f'{uid}.npz'))
        shapes[uid] = ct_shape
        seg = os.path.join(data_dir, 'nodule_sem_seg_3d', f'{uid}.npz')
        if os.path.exists(seg) and _npz_shape(seg) != ct_shape:
            mismatch.append(uid)
    rep.check(not mismatch, 'nodule masks have the shape of their CT volume',
              f'{len(mismatch)} nodule masks differ in shape from their CT', mismatch)

    ct2d  = _slice_counts(os.path.join(data_dir, 'ct_2d'))
    roi2d = _slice_counts(os.path.join(data_dir, 'roi_sem_seg_2d'))
    slices = {}
    if ct2d is not None or roi2d is not None:
        ct2d, roi2d = ct2d or {}, roi2d or {}
        rep.check(set(ct2d) <= expected, 'every 2-D scan is in the split',
                  f'{len(set(ct2d) - expected)} scans in ct_2d/ are not in the split',
                  sorted(set(ct2d) - expected))
        rep.check(ct2d == roi2d, f'ct_2d/ and roi_sem_seg_2d/ hold the same slices ({len(ct2d)} scans)',
                  'ct_2d/ and roi_sem_seg_2d/ hold different slices',
                  sorted(u for u in set(ct2d) | set(roi2d) if ct2d.get(u) != roi2d.get(u)))
        gaps = [u for u, idx in ct2d.items() if idx != list(range(len(idx)))]
        rep.check(not gaps, 'slice indices are contiguous from 0', f'{len(gaps)} scans with missing slices', gaps)
        depth = [u for u, idx in ct2d.items() if u in shapes and len(idx) != shapes[u][-1]]
        rep.check(not depth, 'every 2-D scan has one slice per axial position of its volume',
                  f'{len(depth)} scans whose slice count differs from the volume depth', depth)

        ds_of = split.set_index('series_uid')['dataset']
        coverage = split.assign(has_2d=split.series_uid.isin(ct2d)).groupby('dataset')['has_2d'].agg(['sum', 'count'])
        partial = coverage[(coverage['sum'] > 0) & (coverage['sum'] < coverage['count'])]
        rep.check(partial.empty, '2-D slices cover each dataset completely or not at all: '
                  + ', '.join(f'{ds} {int(r["sum"])}/{int(r["count"])}' for ds, r in coverage.iterrows()),
                  'some datasets have 2-D slices for only part of their scans',
                  [f'{ds}: {int(r["sum"])}/{int(r["count"])}' for ds, r in partial.iterrows()])
        slices = {u: len(idx) for u, idx in ct2d.items() if u in ds_of.index}
    return slices


def check_masks(rep, data_dir, split, catalog, how, workers, seed=0):
    if how in ('0', 'none'):
        return
    uids = sorted(split.series_uid)
    if how != 'all':
        n = int(how)
        nlst_rad = set(split.loc[split['dataset'] == 'nlst_radiologist', 'series_uid'])
        rest = [u for u in uids if u not in nlst_rad]
        random.Random(seed).shuffle(rest)
        uids = sorted(nlst_rad | set(rest[:max(0, n - len(nlst_rad))]))
    print(f'\nNodule masks vs catalog ({len(uids)} scans; volumes in voxels = mm³ at 1 mm spacing)')

    expected = catalog.groupby('series_uid')['volume_mm3'].apply(lambda v: sorted(round(x) for x in v))
    jobs = [(u, os.path.join(data_dir, 'nodule_sem_seg_3d', f'{u}.npz')) for u in uids]
    wrong_count, wrong_volume, not_binary, errors = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for uid, sizes, values_ok, err in pool.map(_mask_job, jobs, chunksize=4):
            if err:
                errors.append(f'{uid}: {err}')
                continue
            if not values_ok:
                not_binary.append(uid)
            exp = expected.get(uid, [])
            if len(sizes) != len(exp):
                wrong_count.append(f'{uid}: {len(sizes)} components on disk, {len(exp)} in catalog')
            elif sizes != exp:
                wrong_volume.append(f'{uid}: volumes {sizes[:6]} on disk vs {exp[:6]} in catalog')
    rep.check(not errors, 'all masks readable', f'{len(errors)} masks could not be read', errors)
    rep.check(not not_binary, 'masks are binary (0/1)', f'{len(not_binary)} masks with other values', not_binary)
    rep.check(not wrong_count, 'component count of every mask matches the catalog',
              f'{len(wrong_count)} masks have a different number of components than the catalog', wrong_count)
    rep.check(not wrong_volume, 'component volumes of every mask match the catalog',
              f'{len(wrong_volume)} masks have different component volumes than the catalog', wrong_volume)


def check_dropped(rep, dropped, known, split):
    print('\nDropped series (not in the release)')
    if dropped is None:
        rep.fail('dropped_series.csv not found (written by process.py; re-run process.py '
                 'with this pipeline version on the same save_path to create it)')
        return
    rep.check(dropped['series_uid'].is_unique, f'one row per dropped scan ({len(dropped)} scans)',
              'duplicate scans in the dropped-series report',
              dropped[dropped['series_uid'].duplicated()].series_uid)
    both = sorted(set(dropped.series_uid) & set(split.series_uid))
    rep.check(not both, 'no dropped scan is in the release',
              f'{len(both)} scans are both released and reported as dropped', both)

    other  = dropped[dropped['reason'] != DATA_CHECK_REASON]
    known_ids = set(known.series_uid)
    unknown = other[~other['series_uid'].isin(known_ids)]
    rep.check(unknown.empty,
              f'every scan dropped for a reason other than the data checks is a known '
              f'source-data failure ({len(other)} scans)',
              f'{len(unknown)} scans failed for other reasons (out of memory, I/O, ...): '
              f're-run process.py with the same config to retry them; a scan that fails '
              f'because of its source data belongs in configs/known_failures.csv',
              [f'{r.series_uid} ({r.dataset}): {r.reason}: {r.detail}' for r in unknown.itertuples()])

    counts = dropped.groupby(['dataset', 'reason', 'detail']).size()
    for (ds, reason, detail), n in sorted(counts.items(), key=lambda kv: (dataset_priority(kv[0][0]), kv[0][1:])):
        print(f'  info   {ds:17s} {n:4d}  {reason}: {detail[:70]}')


def check_ownership(rep, raw_dir, split, dropped):
    print('\nDataset ownership and completeness (against the raw manifests)')
    owner = {}
    in_manifest = defaultdict(set)
    for ds in sorted({'lidc_idri', 'nsclc_radiomics', 'nlst_radiologist', 'nlst_ai'}, key=dataset_priority):
        path = os.path.join(raw_dir, ds, 'manifest.csv')
        if not os.path.exists(path):
            print(f'  -      no manifest for {ds} ({path})')
            continue
        for uid in pd.read_csv(path, dtype=str)['CTSeriesInstanceUID'].unique():
            in_manifest[ds].add(uid)
            owner.setdefault(uid, ds)
    if not owner:
        rep.fail(f'no manifests found under {raw_dir}')
        return
    labelled = split.set_index('series_uid')['dataset']
    unknown = [u for u in labelled.index if u not in owner]
    wrong = [f'{u}: released as {labelled[u]}, highest-priority dataset is {owner[u]}'
             for u in labelled.index if u in owner and owner[u] != labelled[u]]
    rep.check(not unknown, 'every released scan appears in a raw manifest',
              f'{len(unknown)} released scans are in no manifest', unknown)
    rep.check(not wrong, 'every scan is released under its highest-priority dataset',
              f'{len(wrong)} scans are released under a lower-priority dataset', wrong)
    if dropped is not None:
        dropped_ds = dropped.set_index('series_uid')['dataset']
        d_unknown = [u for u in dropped_ds.index if u not in owner]
        d_wrong = [f'{u}: reported as {dropped_ds[u]}, highest-priority dataset is {owner[u]}'
                   for u in dropped_ds.index if u in owner and owner[u] != dropped_ds[u]]
        rep.check(not d_unknown and not d_wrong,
                  'every dropped scan was processed under its highest-priority dataset',
                  f'{len(d_unknown) + len(d_wrong)} dropped scans are in no manifest or were '
                  f'processed under a lower-priority dataset', d_unknown + d_wrong)
        accounted = set(labelled.index) | set(dropped_ds.index)
        missing = sorted(u for u in owner if u not in accounted)
        rep.check(not missing, f'every manifest scan ({len(owner)}) is either released or reported as dropped',
                  f'{len(missing)} manifest scans are neither released nor reported as dropped '
                  f'(not processed, or processed by an older pipeline version)', missing)
    else:
        dropped_ds = pd.Series(dtype=str)
    for ds in sorted(in_manifest, key=dataset_priority):
        owned = {u for u in in_manifest[ds] if owner[u] == ds}
        released = owned & set(labelled.index)
        n_dropped = len(owned & set(dropped_ds.index))
        print(f'  info   {ds:17s} manifest scans: {len(in_manifest[ds]):5d}  owned: {len(owned):5d}  '
              f'released: {len(released):5d}  dropped: {n_dropped:3d}')


def summary(split, catalog, slices):
    print('\nSummary (for Data Records / Data Overview / Technical Validation)')
    nodules = catalog.groupby('series_uid').size()
    s = split.assign(nodules=split.series_uid.map(nodules).fillna(0).astype(int),
                     slices=split.series_uid.map(slices).fillna(0).astype(int))
    order = sorted(s['dataset'].unique(), key=dataset_priority)
    print(f'  {"dataset":17s} {"split":9s} {"scans":>6s} {"patients":>9s} {"nodules":>8s} {"2-D slices":>11s}')
    for ds in order:
        for sp in ('train', 'test', 'excluded'):
            g = s[(s.dataset == ds) & (s.split == sp)]
            if len(g):
                print(f'  {ds:17s} {sp:9s} {len(g):6d} {g.patient_id.nunique():9d} '
                      f'{g.nodules.sum():8d} {g.slices.sum():11d}')
    for sp in ('train', 'test', 'excluded'):
        g = s[s.split == sp]
        print(f'  {"all":17s} {sp:9s} {len(g):6d} {g.patient_id.nunique():9d} '
              f'{g.nodules.sum():8d} {g.slices.sum():11d}')
    print(f'  {"all":17s} {"total":9s} {len(s):6d} {s.patient_id.nunique():9d} '
          f'{s.nodules.sum():8d} {s.slices.sum():11d}')


def main():
    parser = argparse.ArgumentParser(description='Validate a processed release before publishing it.')
    parser.add_argument('data_dir')
    parser.add_argument('--split', default=None)
    parser.add_argument('--catalog', default=None)
    parser.add_argument('--dropped', default=None)
    parser.add_argument('--known-failures', default=DEFAULT_KNOWN_FAILURES)
    parser.add_argument('--raw', default=None)
    parser.add_argument('--check-masks', default='200')
    parser.add_argument('--min-volume', type=float, default=5.0)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    split_path   = args.split or os.path.join(args.data_dir, 'split.csv')
    catalog_path = args.catalog or os.path.join(args.data_dir, 'nodule_catalog.csv')
    dtypes = {'patient_id': str, 'series_uid': str, 'dataset': str}
    split   = pd.read_csv(split_path, dtype={**dtypes, 'split': str})
    catalog = pd.read_csv(catalog_path, dtype=dtypes)
    dropped_path = args.dropped or os.path.join(args.data_dir, 'dropped_series.csv')
    dropped = (pd.read_csv(dropped_path, dtype=str, keep_default_na=False)
               if os.path.exists(dropped_path) else None)
    known   = pd.read_csv(args.known_failures, dtype=str, keep_default_na=False)
    print(f'Data:    {args.data_dir}\nSplit:   {split_path}\nCatalog: {catalog_path}\n'
          f'Dropped: {dropped_path}\nKnown failures: {args.known_failures}')

    rep = Report()
    check_catalog(rep, catalog, args.min_volume)
    check_split(rep, split, catalog)
    slices = check_files(rep, args.data_dir, split) or {}
    check_dropped(rep, dropped, known, split)
    check_masks(rep, args.data_dir, split, catalog, args.check_masks, args.workers)
    if args.raw:
        check_ownership(rep, args.raw, split, dropped)
    summary(split, catalog, slices)

    print(f'\n{"FAILED" if rep.errors else "PASSED"}: {rep.errors} check(s) failed.')
    sys.exit(1 if rep.errors else 0)


if __name__ == '__main__':
    main()
