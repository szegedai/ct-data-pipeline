# ct-data-pipeline

Harmonises and merges public lung CT datasets from the
[NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) (IDC)
into one machine-learning-ready dataset: LIDC-IDRI, NSCLC-Radiomics and the NLST
scans with radiologist-corrected or AI-generated nodule segmentations.
Every CT series is reoriented to RAS, resampled to 1 × 1 × 1 mm, HU-windowed and
normalised, and saved with its binary nodule mask; NSCLC-Radiomics and NLST
scans also get 2-D axial slices with a lung + nodule region-of-interest mask.
A nodule catalogue and a patient-level train/test split are produced alongside,
and a validation script checks the result before release.

This repository contains the code used to produce the published release. The
release itself and its description are in the accompanying Data Descriptor.

## Installation

Python 3.13 (the release was produced with 3.13.16) on Linux; a CUDA GPU is
optional but makes processing much faster.

```bash
git clone https://github.com/szegedai/ct-data-pipeline.git
cd ct-data-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact package versions of the release, including
`idc-index-data==23.3.1`, which fixes the input to IDC data release v23.
`dcm2niix` is installed as a Python package and must be on the `PATH`
(it is, inside the activated environment).

Alternatively, `container/ct-data-pipeline.def` builds an
[Apptainer](https://apptainer.org/) image with the same environment:

```bash
apptainer build --fakeroot ct-data-pipeline.sif container/ct-data-pipeline.def
apptainer exec --nv ct-data-pipeline.sif python process.py configs/release.yaml
```

## Reproducing the release

`configs/release.yaml` holds the settings of the release. Paths and execution
settings can be changed on the command line with `--set KEY=VALUE` (repeatable)
without editing the file; all other values determine the content of the release.

```bash
# 1. Download the raw data from IDC and process it.
python process.py configs/release.yaml
#    e.g. with other paths and 6 workers sharing one GPU:
#    python process.py configs/release.yaml --set raw_data_path=/data/raw \
#        --set save_path=/data/release --set workers=6

# 2. Patient-level train/test split (306 test patients, seed 42).
python generate_split.py ../data/processed/release/nodule_catalog.csv \
    --output ../data/processed/release/split.csv

# 3. Validate the release; it must end with PASSED.
python validate_release.py ../data/processed/release --raw ../data/raw --check-masks all
```

`process.py` skips series that are already processed, so an interrupted run is
resumed by running the same command again. Failed series are retried as well:
series that fail for a transient reason (for example CUDA out of memory when
too many workers share a GPU) are written on the next run. With `sync: false`
(`--set sync=false`) the raw data already on disk is used without contacting IDC.

**Workers and GPUs.** By default one worker runs per GPU. Several workers can
share a GPU (`--set workers=N`); the release was processed with 6 workers on one
NVIDIA H100 (93 GB), whereas 16 workers ran out of GPU memory. `cpu: true` forces
a single CPU worker.

### Expected result

| Dataset | Scans in IDC v23 manifest | Released scans | Dropped | Nodules |
|---|---:|---:|---:|---:|
| NLST, radiologist-corrected | 102 | 98 | 4 | 825 |
| LIDC-IDRI | 883 | 870 | 13 | 2,627 |
| NSCLC-Radiomics | 421 | 407 | 14 | 455 |
| NLST, AI-generated | 940¹ | 927 | 13 | 5,632 |
| **Total** | **2,346** | **2,302** | **44** | **9,539** |

¹ 1,042 scans, of which 102 are processed with their radiologist-corrected annotation.

Every scan of the manifests is either released or listed in
`dropped_series.csv`. Of the 44 dropped scans, 43 are removed by the data checks
(`DataAnomalyError`: no nodule or lung segmentation, or no nodule voxels left
after resampling) and one NSCLC-Radiomics series cannot be converted from DICOM
(listed in `configs/known_failures.csv`). `validate_release.py` fails if a scan
is missing for any other reason.

## Output

All datasets are written side by side into `save_path`; files are named by CT
series instance UID.

| Path | Content |
|---|---|
| `ct_3d/<uid>.npz` | CT volume, 1 mm isotropic, HU window [−1000, 400] scaled to [0, 1] |
| `nodule_sem_seg_3d/<uid>.npz` | Binary nodule mask of the same shape |
| `ct_2d/<uid>_<slice>.npz` | Axial CT slices (NSCLC-Radiomics, NLST) |
| `roi_sem_seg_2d/<uid>_<slice>.npz` | Lung + nodule mask of each slice |
| `nodule_catalog.csv` | One row per nodule: `patient_id, series_uid, dataset, volume_mm3, entropy` |
| `split.csv` | `series_uid, patient_id, dataset, split` with split `train`, `test` or `excluded`² |
| `dropped_series.csv` | Scans not in the release: `series_uid, dataset, reason, detail` |
| `logs/` | Processing logs |
| `*_parts/` | Per-series fragments the CSV files are assembled from (not part of the release) |

² `excluded`: scans with only an AI-generated annotation that belong to a test
patient; they are kept out of both training and testing.

Each `.npz` file holds the array `data` and its geometry (`affine`, `spacing`,
`origin`, `direction`); 2-D slices carry the geometry of their volume.

A scan annotated by several datasets is processed once, by the
highest-priority one (NLST radiologist-corrected > LIDC-IDRI > NSCLC-Radiomics >
NLST AI-generated; `ct_data_management/datasets.py`).

## Other tools

* `catalog_nodules.py` — nodule catalogue of the raw annotations without the
  volume filter and without writing any files (used for the nodule volume
  distribution):
  `python catalog_nodules.py configs/release.yaml --set sync=false`
  writes `<save_path>/nodule_catalog_unfiltered.csv`.
* `view.py` — interactive viewer for `.npz` and NIfTI outputs.
* `seg2nii.py` — converts a DICOM SEG series to NIfTI.
* `configs/default.yaml`, `configs/default.toml` — all configuration options
  with comments.

## License

Apache License 2.0; see `LICENSE`. The datasets themselves are distributed by
IDC under their own licences.
