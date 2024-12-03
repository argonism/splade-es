import logging
import os
import time
from itertools import batched
from typing import Iterable

import gokart
import gokart.target
import luigi
from gokart.task import TaskOnKart
from tqdm import tqdm

from splade_es.dataset import Doc, get_dataset, Query
from splade_es.utils import ElasticsearchClient
from splade_es.tasks.splade import SpladeIndexTask, SpladeEncoder, INDEX_SCHEMA, SpladeESAccessTask

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VECTOR_FIELD = os.getenv("SPLADE_VECTOR_FIELD", "splade_term_weights")

class SpladePhraseSearchTask(SpladeESAccessTask):
    dataset = luigi.Parameter()
    encoder_path = luigi.Parameter()
    top_k = luigi.IntParameter(default=100)

    batch_size = luigi.IntParameter(default=250)

    def requires(self) -> TaskOnKart:
        return SpladeIndexTask()

    def output(self) -> gokart.target.TargetOnKart:
        return self.cache_path(f"splade-phrase/{self.encoder_path}/{self.dataset}/search/{self.index_name}_{self.top_k}.pkl")

    def run(self):
        def yield_query(
                index_name: str,
                queries: Iterable[Query],
                query_vectors: Iterable[dict[str, float]],
                doc_fields: list[str],
                query_field: str = "text",
                slop: int = 1,
                phrase_weight: int = 2,
                size: int = 100
            ):
            for query, query_vector in zip(queries, query_vectors):
                yield {"index": index_name}
                yield {
                    "query": {
                        "function_score": {
                            "query": {
                                "sparse_vector": {
                                    "field": VECTOR_FIELD,
                                    "query_vector": query_vector
                                }
                            },
                            "functions": [
                                {
                                    "filter": {
                                        "multi_match": {
                                        "query": getattr(query, query_field),
                                        "fields": doc_fields,
                                        "slop": slop
                                        }
                                    },
                                    "weight": phrase_weight
                                }
                            ]
                        }
                    },
                    "size": size
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
            term_weights = splade_encoder.encode_to_dict([query.text for query in batch_queries])

            es_res = es_client.msearch(
                searches=yield_query(
                    splade_index,
                    batch_queries,
                    term_weights,
                    dataset.doc_text_fields,
                    size=self.top_k
                )
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