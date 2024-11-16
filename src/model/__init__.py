from src.model.base import SearchModelBase
from src.model.bm25 import ESBM25
from src.model.splade import ESSplade

MODELS = {
    "bm25": ESBM25, "splade": ESSplade
}


def get_search_model(key: str) -> type[SearchModelBase]:
    if key not in MODELS:
        raise ValueError(f"Unknown model {key}")
    return MODELS[key]
