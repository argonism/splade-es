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

INDEX_SCHEMA = {
    "settings": {},
    "mappings": {"properties": {}},
}


class BM25ESAccessTask(BaseTask):
    dataset = luigi.Parameter()
    suffix = luigi.Parameter(default="")
    debug = luigi.BoolParameter(default=False)

    index_batch_size = luigi.IntParameter(default=5_000)

    @property
    def index_name(self) -> str:
        index_name = f"bm25.{self.dataset}"
        suffix = "debug" if self.suffix == "" and self.debug else ""
        if suffix:
            index_name += f".{suffix}"
        return index_name


class BM25SearchTask(BM25ESAccessTask):
    top_k = luigi.IntParameter(default=100)

    batch_size = luigi.IntParameter(default=250)

    def requires(self) -> TaskOnKart:
        return BM25IndexTask()

    def output(self) -> gokart.target.TargetOnKart:
        return self.cache_path(f"bm25/{self.dataset}/search/{self.index_name}_{self.top_k}.pkl")

    def run(self):
        def yield_query(index_name: str, queries: list[str], search_fields: list[str], size: int = 100):
            for query in queries:
                yield {"index": index_name}
                yield {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": search_fields,
                        }
                    },
                    "size": size,
                }

        dataset = get_dataset(self.dataset)

        client = ElasticsearchClient(
            index_name=self.index_name, dataset=dataset, index_schema=INDEX_SCHEMA
        )

        logger.info(f"Searching from %s", self.index_name)

        queries = dataset.queries
        queries = list(queries)
        search_result = {}
        search_start_time = time.perf_counter()
        for batch_queries in tqdm(
            batched(queries, self.batch_size),
            total=len(queries) // self.batch_size,
            desc="Searching batch queries"
        ):
            res = client.msearch(
                searches=yield_query(
                    self.index_name, [query.text for query in batch_queries], dataset.doc_text_fields, self.top_k
                ),
            )
            for response, query in zip(res["responses"], batch_queries):
                if "error" in response:
                    logger.error("error: %s", response)
                    logger.error("query: %s", query)
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
            print(f"search result {qid}: {filtered_results}")

        self.dump(search_result)


class BM25IndexTask(BM25ESAccessTask):
    corpus_load_batch_size = luigi.IntParameter(default=250)

    def output(self) -> gokart.target.TargetOnKart:
        return self.cache_path(f"bm25/{self.dataset}/index/{self.index_name}.pkl")

    def _make_body(self, content) -> str:
        if isinstance(content, str):
            return content
        elif content is None:
            return ""
        elif isinstance(content, (int, float)):
            return str(int)
        elif isinstance(content, list):
            return " ".join([str(item) for item in content])
        else:
            raise ValueError(f"Unexpected content type: {content}")

    def run(self):
        def make_bulk_insert_body(docs: Iterable[Doc], doc_text_fields: list[str]) -> Generator[dict, None, None]:
            for doc in docs:
                if not hasattr(doc, "id"):
                    raise ValueError(f"{doc} does not have id field")

                yield {"_index": self.index_name, "_id": doc.doc_id, "_source": {
                    field: self._make_body(getattr(doc, field))
                    for field in doc_text_fields
                }}

        dataset = get_dataset(self.dataset)

        client = ElasticsearchClient(
            index_name=self.index_name, dataset=dataset, index_schema=INDEX_SCHEMA
        )

        logger.info(f"Indexing to %s", self.index_name)

        write_count = 0
        for batch_docs in tqdm(
            batched(dataset.corpus_iter(self.debug), self.corpus_load_batch_size),
            total=dataset.docs_count // self.corpus_load_batch_size,
            desc="SPLADE Encoding"
        ):
            success, errors = client.bulk(
                make_bulk_insert_body(batch_docs, dataset.doc_text_fields),
                self.index_name,
                reset_index=True
            )
            write_count += success

        logger.info(f"Indexed {write_count} documents to {self.index_name}")
        self.dump(self.index_name)
