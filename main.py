from os import environ
import logging

from elasticsearch import Elasticsearch
import ranx

from splade_es.dataset import get_dataset
from splade_es.model import get_search_model


logger = logging.getLogger("splade_es")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-[%(name)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def main():
    es_client = Elasticsearch(environ["ELASTICSEARCH_URL"])

    dataset_name = "scifact"
    dataset = get_dataset(dataset_name)

    # search_model = get_search_model("bm25")(es_client, dataset, reset_index=True)
    search_model = get_search_model("splade")(es_client, dataset, reset_index=True)

    DEBUG = False
    corpus = dataset.corpus_iter()
    if DEBUG:
        corpus = [doc for i, doc in enumerate(dataset.corpus_iter()) if i < 3]

    search_model.index(corpus)
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
    logger.setLevel(logging.DEBUG)
    main()
