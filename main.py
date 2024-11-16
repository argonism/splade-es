from os import environ

from elasticsearch import Elasticsearch
import ranx

from src.dataset import get_dataset
from src.model import get_search_model



def main():
    es_client = Elasticsearch(environ["ELASTICSEARCH_URL"])

    dataset_name = "nfcorpus"
    dataset = get_dataset(dataset_name)

    # search_model = get_search_model("bm25")(es_client, dataset, reset_index=True)
    search_model = get_search_model("splade")(es_client, dataset, reset_index=True)

    # search_model.index(dataset.corpus_iter())
    search_result = search_model.search(dataset.queries)

    # Print 10 search result examples:
    for result in list(search_result.values())[:5]:
        print(f"{result}")

    run = ranx.Run(search_result)
    metrics = [
        "ndcg",
        "ndcg@10",
        "ndcg@20",
        "recall",
        "recall@10",
        "recall@20",
    ]
    eval_result = ranx.evaluate(dataset.qrels, run, metrics=metrics)
    print(eval_result)

if __name__ == "__main__":
    from src.model.bm25 import logger
    logger.setLevel("DEBUG")
    main()
