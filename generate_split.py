'''
Generate a stratified train/test split from a nodule catalog CSV.

The catalog is produced automatically by the processing pipeline (via
NoduleCatalogWriter) and contains one row per nodule with columns:
  patient_id, series_uid, dataset, volume_mm3, entropy

Test set eligibility (scan level):
  - lidc_idri        all scans
  - nsclc_radiomics  all scans
  - nlst_radiologist all scans (hard cap: 102 patients)
  - nlst_ai          never in the test set

A patient is test-eligible if at least one of their scans is.  When an eligible
patient is drawn for the test set, their eligible scans go to the test set and
their remaining scans (AI-only NLST scans of a patient who also has a
radiologist-corrected scan) are marked 'excluded': they belong to neither split,
which keeps the test labels radiologist-derived and the split patient-isolated.
The volume/entropy statistics used for stratification are computed from the
eligible scans only.

Every scan must carry a single dataset label in the catalog; the processing
pipeline guarantees this by processing each CT series in one dataset only.

Output columns: series_uid, patient_id, dataset, split (train | test | excluded).
Given the same catalog, the split is identical across runs: the catalog is
sorted before sampling, so it does not depend on the order of its rows.

Stratification uses a 4 × 4 × 3 cell grid:
  axis 0 — median nodule volume quartile (4 bins)
  axis 1 — mean nodule entropy quartile  (4 bins)
  axis 2 — dataset                       (3: lidc_idri, nsclc_radiomics,
                                              nlst_radiologist)

All eligible patients are pooled together and each cell's quota is
proportional to its share of the full eligible pool.  This balances the
combined volume/entropy distribution first while keeping each dataset's
contribution proportional to the data it actually has in each bin.

If the proportional NLST quota exceeds 102, NLST cell quotas are scaled
down to sum to exactly 102 and the freed slots are redistributed
proportionally to the non-NLST cells.

Sampling is done at the patient level so no patient appears in both
the training and the test split.

Usage:
    python generate_split.py nodule_catalog.csv
    python generate_split.py nodule_catalog.csv --test-size 306 --seed 42 --output split.csv
'''

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from ct_data_management.datasets import TEST_ELIGIBLE_DATASETS, dataset_priority

_logger = logging.getLogger('generate_split')

NLST_RADIOLOGIST_CAP   = 102
BALANCE_WARNING_SIZE   = 3 * NLST_RADIOLOGIST_CAP   # 306

N_QUANTILE_BINS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_catalog(catalog_csv: str) -> pd.DataFrame:
    """Read and sanity-check the nodule catalog; return it sorted by scan."""
    catalog = pd.read_csv(catalog_csv, dtype={'patient_id': str, 'series_uid': str, 'dataset': str})
    required = {'patient_id', 'series_uid', 'dataset', 'volume_mm3', 'entropy'}
    missing  = required - set(catalog.columns)
    if missing:
        raise ValueError(f'Catalog is missing columns: {missing}')

    if catalog[['patient_id', 'series_uid', 'dataset']].isna().any().any() \
            or (catalog['patient_id'].str.strip() == '').any():
        raise ValueError('Catalog has rows without patient_id, series_uid or dataset.')

    labels = catalog.groupby('series_uid')['dataset'].nunique()
    if (labels > 1).any():
        bad = labels[labels > 1].index.tolist()
        raise ValueError(
            f'{len(bad)} series carry more than one dataset label (e.g. {bad[:3]}). '
            f'Each CT series must be catalogued by exactly one dataset; '
            f're-run process.py, which processes every series in a single dataset.'
        )

    patients = catalog.groupby('series_uid')['patient_id'].nunique()
    if (patients > 1).any():
        raise ValueError('Some series are assigned to more than one patient_id.')

    # Sorting makes the sampling below independent of the catalog's row order,
    # which depends on the order in which parallel workers finished.
    return catalog.sort_values(['series_uid', 'volume_mm3', 'entropy'], kind='mergesort')


def _aggregate_to_scan(catalog: pd.DataFrame) -> pd.DataFrame:
    """Collapse one-row-per-nodule catalog to one row per scan."""
    return (
        catalog
        .groupby('series_uid', sort=True)
        .agg(
            patient_id    = ('patient_id',  'first'),
            dataset       = ('dataset',     'first'),
            median_volume = ('volume_mm3',  'median'),
            mean_entropy  = ('entropy',     'mean'),
        )
        .reset_index()
    )


def _aggregate_to_patient(per_scan: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per patient (handles multi-scan patients)."""
    def _priority_dataset(datasets: pd.Series) -> str:
        return min(datasets, key=dataset_priority)

    return (
        per_scan
        .groupby('patient_id', sort=True)
        .agg(
            dataset       = ('dataset',       _priority_dataset),
            median_volume = ('median_volume', 'median'),
            mean_entropy  = ('mean_entropy',  'mean'),
        )
        .reset_index()
    )


def _assign_bins(series: pd.Series, n: int) -> pd.Series:
    """Quantile-bin a series; fall back to rank-based cut on duplicate edges."""
    try:
        return pd.qcut(series, q=n, labels=False, duplicates='drop')
    except ValueError:
        return pd.qcut(series.rank(method='first'), q=n, labels=False)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def generate_split(
    catalog_csv: str,
    test_size: int = BALANCE_WARNING_SIZE,
    seed: int = 42,
    output: str = 'split.csv',
) -> pd.DataFrame:

    catalog = _read_catalog(catalog_csv)

    if test_size > BALANCE_WARNING_SIZE:
        _logger.warning(
            'Requested test size %d exceeds 3 × %d = %d. '
            'NLST radiologist pool is limited to %d scans; '
            'the test set will be underrepresented for that sub-dataset.',
            test_size, NLST_RADIOLOGIST_CAP, BALANCE_WARNING_SIZE, NLST_RADIOLOGIST_CAP,
        )

    rng = np.random.default_rng(seed)

    # --- Aggregate catalog rows to one row per scan ---
    per_scan = _aggregate_to_scan(catalog)
    per_scan.loc[:, 'eligible'] = per_scan['dataset'].isin(TEST_ELIGIBLE_DATASETS)

    # --- Test-eligible patients, described by their eligible scans only ---
    eligible = _aggregate_to_patient(per_scan[per_scan['eligible']])
    if eligible.empty:
        raise ValueError('No test-eligible patients found in the catalog.')

    # --- Bin volume and entropy across the COMBINED eligible pool ---
    # Bins are global so the same cell means the same thing across datasets.
    eligible.loc[:, 'volume_bin']  = _assign_bins(eligible['median_volume'], N_QUANTILE_BINS)
    eligible.loc[:, 'entropy_bin'] = _assign_bins(eligible['mean_entropy'],  N_QUANTILE_BINS)

    # --- Compute proportional quota for each (volume_bin, entropy_bin, dataset) cell ---
    # Each cell gets a quota proportional to its fraction of the eligible pool.
    total_eligible = len(eligible)
    cells = eligible.groupby(['volume_bin', 'entropy_bin', 'dataset'], observed=True)

    cell_quotas = {}
    for key, group in cells:
        cell_quotas[key] = round(test_size * len(group) / total_eligible)

    # --- Enforce the NLST radiologist cap ---
    # If the sum of NLST cell quotas exceeds 102, scale them down proportionally
    # and redistribute the freed slots to non-NLST cells.
    nlst_keys     = [k for k in cell_quotas if k[2] == 'nlst_radiologist']
    non_nlst_keys = [k for k in cell_quotas if k[2] != 'nlst_radiologist']

    total_nlst_quota = sum(cell_quotas[k] for k in nlst_keys)
    if total_nlst_quota > NLST_RADIOLOGIST_CAP:
        freed = total_nlst_quota - NLST_RADIOLOGIST_CAP

        # Scale NLST quotas down proportionally.
        for k in nlst_keys:
            cell_quotas[k] = round(cell_quotas[k] * NLST_RADIOLOGIST_CAP / total_nlst_quota)

        # Redistribute freed slots to non-NLST cells proportionally.
        non_nlst_total = sum(cell_quotas[k] for k in non_nlst_keys)
        if non_nlst_total > 0:
            for k in non_nlst_keys:
                cell_quotas[k] += round(freed * cell_quotas[k] / non_nlst_total)

    # --- Sample from each cell ---
    test_selected = []
    for key, group in cells:
        quota = min(cell_quotas[key], len(group))
        if quota <= 0:
            continue
        chosen = rng.choice(group['patient_id'].values, size=quota, replace=False)
        test_selected.extend(chosen.tolist())

    # --- Rounding adjustment ---
    # Cell quotas are independently rounded so their sum may differ from
    # test_size by a small amount.  Adjust randomly while respecting the cap.
    test_selected = list(dict.fromkeys(test_selected))   # deduplicate

    if len(test_selected) > test_size:
        test_selected = rng.choice(test_selected, size=test_size, replace=False).tolist()

    elif len(test_selected) < test_size:
        nlst_used = sum(
            1 for pid in test_selected
            if eligible.loc[eligible['patient_id'] == pid, 'dataset'].iloc[0] == 'nlst_radiologist'
        )
        remaining = eligible[~eligible['patient_id'].isin(test_selected)]
        # Respect the NLST cap when filling remaining slots.
        if nlst_used >= NLST_RADIOLOGIST_CAP:
            remaining = remaining[remaining['dataset'] != 'nlst_radiologist']
        gap   = min(test_size - len(test_selected), len(remaining))
        extra = rng.choice(remaining['patient_id'].values, size=gap, replace=False)
        test_selected.extend(extra.tolist())

    test_patient_set = set(test_selected)

    # --- Log breakdown by dataset ---
    selected_patients = eligible[eligible['patient_id'].isin(test_patient_set)]
    for ds, grp in selected_patients.groupby('dataset', observed=True):
        _logger.info('Test set — %s: %d patients', ds, len(grp))

    # --- Assign a split to every scan ---
    # Scans of test patients go to the test set only if they are test-eligible;
    # the others (AI-only scans) are excluded so that they appear in neither split.
    in_test = per_scan['patient_id'].isin(test_patient_set)
    per_scan.loc[:, 'split'] = np.where(
        ~in_test, 'train', np.where(per_scan['eligible'], 'test', 'excluded')
    )

    result = per_scan[['series_uid', 'patient_id', 'dataset', 'split']]
    result.to_csv(output, index=False)

    counts = result['split'].value_counts()
    _logger.info(
        'Split saved to %s  (test: %d  train: %d  excluded: %d scans)',
        output, counts.get('test', 0), counts.get('train', 0), counts.get('excluded', 0),
    )
    excluded = result[result['split'] == 'excluded']
    if not excluded.empty:
        _logger.info(
            'Excluded %d non-eligible scan(s) of %d test patient(s) (%s).',
            len(excluded), excluded['patient_id'].nunique(),
            ', '.join(f'{ds}: {n}' for ds, n in excluded['dataset'].value_counts().items()),
        )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s %(message)s',
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description='Generate a stratified train/test split from a nodule catalog.',
        epilog='The catalog CSV is produced by running the processing pipeline.',
    )
    parser.add_argument('catalog',     help='Path to nodule_catalog.csv')
    parser.add_argument('--test-size', type=int, default=BALANCE_WARNING_SIZE,
                        help=f'Total number of patients in the test set (default: {BALANCE_WARNING_SIZE})')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output',    default='split.csv',
                        help='Output CSV path (default: split.csv)')
    args = parser.parse_args()

    generate_split(
        catalog_csv=args.catalog,
        test_size=args.test_size,
        seed=args.seed,
        output=args.output,
    )


if __name__ == '__main__':
    main()
