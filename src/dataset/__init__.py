from src.dataset.base import DatasetBase
from src.dataset.beir import Nfcorpus, Arguana

# DATASETS = {"nfcorpus": Nfcorpus, "arguana": Arguana, "scifact": SciFact, "scidocs": Scidocs, "trec-covid": TrecCovid}
DATASETS = {"nfcorpus": Nfcorpus, "arguana": Arguana}


def get_dataset(key: str) -> DatasetBase:
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {key}")

    return DATASETS[key]()
