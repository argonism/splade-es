import logging
from itertools import batched
from typing import Generator, Generic, Iterable, TypeVar

import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError, bulk
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from splade_es.dataset.base import DatasetBase, Doc, Query
from splade_es.model.base import SearchModelBase, SearchResultType
from splade_es.utils import MiddleFileHandler

logger = logging.getLogger(__name__)


INDEX_SCHEMA = {
    "settings": {},
    "mappings": {
        "properties": {
            "docid": {"type": "keyword"},
            "splade_for_search": {
                "type": "sparse_vector",
            },
        },
    },
}


class SpladeEncoder(object):
    def __init__(
        self, encoder_path: str, device: str, verbose: bool = True
    ) -> None:
        self.splade = AutoModelForMaskedLM.from_pretrained(encoder_path)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        self.vocab_dict = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.verbose = verbose

        logger.debug("loading to device: %s", device)
        self.device = device
        self.splade.to(device)

    def tokenize(self, texts: list[str]) -> list[list[str]]:
        return [self.tokenizer.tokenize(text) for text in texts]

    def _encode_texts(self, texts: list[str], batch_size: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        iterator = (
            tqdm(
                batched(texts, batch_size),
                total=len(texts) // batch_size,
                desc="Encoding texts",
            )
            if self.verbose
            else batched(texts, batch_size)
        )
        for batch in iterator:
            tokenized = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            # MLM output (input length x vocab size)
            output = self.splade(**tokenized, return_dict=True).logits
            # max-pooling against each token's vocab size vectors
            # We handle this output as a expanded document
            output, _ = torch.max(
                torch.log(1 + torch.relu(output))
                * tokenized["attention_mask"].unsqueeze(-1),
                dim=1,
            )
            outputs.append(output)
        return torch.cat(outputs)

    def get_model_output(
        self, texts: list[str], batch_size: int = 32
    ) -> list[dict[str, float]]:
        with torch.no_grad():
            model_outputs = self._encode_texts(texts, batch_size=batch_size)

        return model_outputs

    def output_to_term_weights(self, pooled_output: torch.Tensor) -> dict[str, float]:
        return {
            self.vocab_dict[int(idx)]: pooled_output[idx].item()
            for idx in torch.nonzero(pooled_output).squeeze()
            if not "." == self.vocab_dict[int(idx)]
        }

    def weight_and_expand(
        self, texts: list[str], batch_size: int = 32
    ) -> list[dict[str, float]]:
        model_outputs = self.get_model_output(texts, batch_size=batch_size)

        expand_terms_list: list[dict] = []
        for model_output in model_outputs:
            expand_terms_list.append(
                self.output_to_term_weights(model_output)
            )

        return expand_terms_list


DocType = TypeVar("DocType")

class SparseVectorMiddleEntry(BaseModel, Generic[DocType]):
    doc: DocType
    sparse_vector: dict[str, float]


class ESSplade(SearchModelBase, model_name="splade", index_schema=INDEX_SCHEMA):
    VECTOR_FIELD = "splade_for_search"

    def __init__(
        self,
        es_client: Elasticsearch,
        dataset: DatasetBase,
        reset_index: bool = False,
        encoder_path: str = "naver/splade-v3",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(es_client, dataset, reset_index=reset_index, **kwargs)
        self.splade = SpladeEncoder(encoder_path=encoder_path, device=device)

    def _make_properties(self) -> dict:
        return {}

    def _get_text_for_encode(self, docs: Iterable[Doc]) -> list[str]:
        texts = []
        for doc in docs:
            text = " ".join(
                [getattr(doc, field, "") for field in ["title", "url", "text"]]
            )
            texts.append(text)
        return texts

    def index(self, docs: Iterable[Doc]) -> None:
        def _make_bulk_insert_body(
            entries: Iterable[SparseVectorMiddleEntry[self.dataset.doc_type]]
        ) -> Generator[dict[str, dict], None, None]:
            for entry in entries:
                doc_fields = {
                    field: self._make_body(getattr(entry.doc, field))
                    for field in self.dataset.doc_text_fields
                }
                doc_fields[self.VECTOR_FIELD] = sparse_vector
                yield {
                    "index": {"_index": self.index_name, "_id": entry.doc.doc_id},
                    "_source": doc_fields
                }

        batch_size = 5
        insert_batch_size = 2
        docs_indexed = 0
        with MiddleFileHandler(self.index_name) as handler:
            with handler.step_with_incremental_files("encode") as step:
                for batch_docs in batched(docs, batch_size):
                    field_texts = self._get_text_for_encode(batch_docs)

                    model_output = self.splade.get_model_output(field_texts)

                    torch.save(
                        (batch_docs, model_output),
                        step.get_new_path_to_file_incremental()
                    )

            with handler.step_with_jsonl("sparse_vectors", SparseVectorMiddleEntry[self.dataset.doc_type]) as step:
                for path in tqdm(handler.previous_step.read_each_files()):
                    doc_output = torch.load(path)
                    for doc, output in zip(*doc_output):
                        sparse_vector = self.splade.output_to_term_weights(output)
                        entry = SparseVectorMiddleEntry[self.dataset.doc_type](doc=doc, sparse_vector=sparse_vector)
                        step.writeone(entry)

            for encoded_docs in batched(handler.get_step("sparse_vectors").read(), insert_batch_size):
                try:
                    success, error = bulk(
                        self.client,
                        _make_bulk_insert_body(encoded_docs),
                        index=self.index_name,
                    )
                    docs_indexed += success
                    if error:
                        logger.error("bulk error: %s", error)
                except BulkIndexError as e:
                    logger.error("BulkIndexError")
                    logger.error("errors: %s", e.errors)
                    raise e

        logger.info("Indexed %d documents", docs_indexed)

    def search(self, queries: Iterable[Query]) -> SearchResultType:
        queries = list(queries)
        logger.info("Searching with %s", len(queries))

        def yield_query(query_vectors: Iterable[dict[str, float]]):
            for query in query_vectors:
                yield {"index": self.index_name}
                yield {
                    "query": {
                        "sparse_vector": {
                            "query_vector": query,
                            "field": self.VECTOR_FIELD,
                        }
                    },
                    "size": 100,
                }

        n = 250
        search_result = {}
        for batch_queries in tqdm(
            batched(queries, n), "Batch query", total=len(queries) // n
        ):
            queries_vectors = self.splade.weight_and_expand(
                [query.text for query in batch_queries]
            )


            res = self.client.msearch(
                searches=yield_query(queries_vectors), index=self.index_name
            )
            for response, query, queries_vector in zip(
                res["responses"], batch_queries, queries_vectors
            ):
                if "error" in response:
                    logger.error("error: %s", response)
                    logger.error("queries_vector: %s", queries_vector)
                    continue

                search_result[query.id] = {
                    hit["_id"]: hit["_score"] for hit in response["hits"]["hits"]
                }

        logger.info("Retrieved %s results", len(search_result))

        return search_result
