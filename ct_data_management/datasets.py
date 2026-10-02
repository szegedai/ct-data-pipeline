'''
Dataset (annotation stream) constants shared by the processing pipeline and the
split generator.  Kept free of heavy imports so that lightweight scripts such as
generate_split.py can use it without pulling in torch, MONAI or idc-index.
'''

# Lower value = higher priority.
#
# Processing: a CT series that appears in the manifests of several runs that
# write to the same output directory is processed only by the highest-priority
# run.  This matters for NLST, where the AIMI release provides an AI-generated
# annotation for most scans and a radiologist-corrected annotation for a subset
# of the same scans: those scans must be written (and catalogued) once, with the
# radiologist-corrected annotation.
#
# Splitting: the same order determines the dataset label of a patient whose
# scans come from several streams.
DATASET_PRIORITY: dict[str, int] = {
    'nlst_radiologist': 0,
    'lidc_idri':        1,
    'nsclc_radiomics':  2,
    'nlst_ai':          3,
}

# Streams whose annotations were produced or reviewed by radiologists.  Only
# scans from these streams may appear in the test split.
TEST_ELIGIBLE_DATASETS = frozenset({'lidc_idri', 'nsclc_radiomics', 'nlst_radiologist'})


def dataset_priority(dataset: str) -> int:
    return DATASET_PRIORITY.get(dataset, len(DATASET_PRIORITY))
