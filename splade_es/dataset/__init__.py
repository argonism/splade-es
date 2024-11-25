import logging

from splade_es.dataset.base import DatasetBase, Doc, Query
from splade_es.dataset.beir import (
    NQ,
    Arguana,
    ClimateFever,
    FiQA,
    Nfcorpus,
    Quora,
    Scidocs,
    SciFact,
    Touchev2,
    TrecCovid,
)

logger = logging.getLogger(__name__)

DATASETS = {
    "nfcorpus": Nfcorpus,
    "arguana": Arguana,
    "scifact": SciFact,
    "scidocs": Scidocs,
    "trec-covid": TrecCovid,
    "quora": Quora,
    "fiqa": FiQA,
    "touche": Touchev2,
    "nq": NQ,
    "climate-fever": ClimateFever
}


def get_dataset(key: str) -> DatasetBase:
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {key}")

    logger.info("Loading dataset: %s", key)

    return DATASETS[key]()
