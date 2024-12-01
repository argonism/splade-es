import json
import logging
import os
import re
import time
from itertools import batched
from pathlib import Path
from typing import Generator, Iterable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import gokart
import gokart.target
import luigi
import torch
from elasticsearch.helpers import BulkIndexError
from gokart.task import TaskOnKart
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from splade_es.dataset import Doc, get_dataset
from splade_es.tasks.base import BaseTask
from splade_es.utils import ElasticsearchClient, PartialFilesManager
from splade_es.tasks.splade import SpladeIndexTask, SpladeEncoder, INDEX_SCHEMA, SpladeESAccessTask

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VECTOR_FIELD = os.getenv("SPLADE_VECTOR_FIELD", "splade_term_weights")


class SpladeDocSearchTask(SpladeESAccessTask):
    dataset = luigi.Parameter()
    top_k = luigi.IntParameter(default=100)
    encoder_path = luigi.Parameter()

    batch_size = luigi.IntParameter(default=250)

    def requires(self) -> TaskOnKart:
        return SpladeIndexTask(encoder_path=self.encoder_path)

    def output(self) -> gokart.target.TargetOnKart:
        return self.cache_path(f"splade/{self.encoder_path}/{self.dataset}/search/{self.index_name}_{self.top_k}.pkl")

    def run(self):
        def yield_query(index_name: str, query_vectors: Iterable[dict[str, float]], size: int = 100):
            for query in query_vectors:
                yield {"index": index_name}
                yield {
                    "query": {
                        "sparse_vector": {
                            "query_vector": query,
                            "field": VECTOR_FIELD,
                        }
                    },
                    "size": size,
                }
        splade_index = self.load()
        dataset = get_dataset(self.dataset)
        splade_encoder = SpladeEncoder(
            encoder_path=self.encoder_path, device="cpu"
        )

        es_client = ElasticsearchClient(splade_index, dataset, INDEX_SCHEMA)

        queries = dataset.queries
        queries = list(queries)
        search_result: dict[str, dict[str, float]] = {}
        search_start_time = time.perf_counter()
        for batch_queries in tqdm(
            batched(queries, self.batch_size),
            total=len(queries) // self.batch_size,
            desc="Searching",
        ):
            tokenized_texts = splade_encoder.tokenize([query.text for query in batch_queries])

            # Set term weights to 1 for all tokens in the query
            term_weights: list[dict[str, float]] = []
            for tokenized_text in tokenized_texts:
                term_weight = {}
                for token in tokenized_text:
                    term_weight[token] = 1
                term_weights.append(term_weight)

            es_res = es_client.msearch(
                searches=yield_query(splade_index, term_weights, size=self.top_k)
            )
            for response, query, queries_vector in zip(
                es_res["responses"], batch_queries, term_weights
            ):
                if "error" in response:
                    logger.error("error: %s", response)
                    logger.error("queries_vector: %s", queries_vector)
                    continue

                search_result[query.id] = {
                    hit["_id"]: hit["_score"] for hit in response["hits"]["hits"]
                }

        search_end_time = time.perf_counter()
        logger.info("Searching took %.2f seconds", search_end_time - search_start_time)
        logger.info("Retrieved %d search results", len(search_result))

        for i, qid in enumerate(search_result):
            if i >= 5:
                break
            filtered_results = {k: v for k, v in list(search_result[qid].items())[:5]}

        self.dump(search_result)
