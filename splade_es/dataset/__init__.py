import logging

from splade_es.dataset.base import DatasetBase
from splade_es.dataset.beir import Nfcorpus, Arguana, SciFact, Scidocs, TrecCovid

logger = logging.getLogger(__name__)

DATASETS = {"nfcorpus": Nfcorpus, "arguana": Arguana, "scifact": SciFact, "scidocs": Scidocs, "trec-covid": TrecCovid}


def get_dataset(key: str) -> DatasetBase:
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {key}")
    
    logger.info("Loading dataset: %s", key)

    return DATASETS[key]()
