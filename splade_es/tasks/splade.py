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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VECTOR_FIELD = os.getenv("SPLADE_VECTOR_FIELD", "splade_term_weights")

INDEX_SCHEMA = {
    "settings": {},
    "mappings": {
        "properties": {
            "docid": {"type": "keyword"},
            VECTOR_FIELD: {
                "type": "sparse_vector",
            },
        },
    },
}



def dictionalize_model_outputs(model_outputs: torch.Tensor, vocab_dict: dict[int, str]) -> list[dict[str, float]]:
    expand_terms_list: list[dict[str, float]] = []
    for model_output in model_outputs:
        indexes = torch.nonzero(model_output, as_tuple=False).squeeze()
        expand_terms = {}
        for idx in indexes:
            weight = model_output[idx].item()
            if weight > 0:
                token = vocab_dict[int(idx)]
                # Ignore dot because elasticsearch does not support it
                if token == ".":
                    continue
                expand_terms[token] = weight

        expand_terms_list.append(expand_terms)

    return expand_terms_list


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

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        with torch.no_grad():
            model_outputs = self._encode_texts(texts, batch_size=batch_size)
            return model_outputs

    def model_outputs_to_dict(self, model_outputs: torch.Tensor) -> list[dict[str, float]]:
        return dictionalize_model_outputs(model_outputs, self.vocab_dict)

    def encode_to_dict(self, texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
        model_outputs = self.encode_texts(texts, batch_size=batch_size)
        return self.model_outputs_to_dict(model_outputs)

class SpladeESAccessTask(BaseTask):
    dataset = luigi.Parameter()
    encoder_path = luigi.Parameter()
    suffix = luigi.Parameter(default="")
    debug = luigi.BoolParameter(default=False)

    index_batch_size = luigi.IntParameter(default=5_000)

    @property
    def index_name(self) -> str:
        encoder_path_clean = re.sub(r"[/\.]", "_", self.encoder_path)
        index_name = f"splade.{encoder_path_clean}.{self.dataset}"
        suffix = "debug" if self.suffix == "" and self.debug else ""
        if suffix:
            index_name += f".{suffix}"
        return index_name


class SpladeSearchTask(SpladeESAccessTask):
    dataset = luigi.Parameter()
    encoder_path = luigi.Parameter()
    top_k = luigi.IntParameter(default=100)

    batch_size = luigi.IntParameter(default=250)

    def requires(self) -> TaskOnKart:
        return SpladeIndexTask(rerun=self.rerun)

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
        for batch_queries in tqdm(
            batched(queries, self.batch_size),
            total=len(queries) // self.batch_size,
            desc="Searching",
        ):
            term_weights = splade_encoder.encode_to_dict([query.text for query in batch_queries])

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

        logger.info("Retrieved %d search results", len(search_result))

        print(f"search results len: {len(search_result)}")
        for i, qid in enumerate(search_result):
            if i >= 5:
                break
            filtered_results = {k: v for k, v in list(search_result[qid].items())[:5]}
            print(f"search result {qid}: {filtered_results}")

        self.dump(search_result)


class SpladeIndexTask(SpladeESAccessTask):
    dataset = luigi.Parameter()
    encoder_path = luigi.Parameter()
    suffix = luigi.Parameter(default="")

    index_batch_size = luigi.IntParameter(default=5_000)

    def requires(self) -> TaskOnKart:
        return SpladeDictionalizeTask(rerun=self.rerun)

    def _load_jsonl(self, jsonl_path: Path) -> Generator[dict, None, None]:
        with jsonl_path.open("r") as f:
            for line in f:
                yield json.loads(line)

    def output(self) -> gokart.target.TargetOnKart:
        return self.cache_path(f"splade/{self.encoder_path}/{self.dataset}/index/{self.index_name}.txt")

    def run(self):
        def _make_bulk_insert_body(
            docs: Iterable[dict],
            doc_text_fields: list[str] | None = None,
        ) -> Generator:
            for doc in docs:
                if "doc_id" not in doc:
                    raise ValueError(f"{doc} does not have doc_id field")

                doc_fields = {}
                if doc_text_fields is None:
                    doc_fields = doc
                else:
                    for field in doc_text_fields:
                        if field not in doc:
                            logger.warning("%s does not have %s field", doc, field)
                            raise ValueError(f"{doc} does not have {field} field")
                        doc_fields[field] = doc[field]
                yield {"_index": self.index_name, "_id": doc["doc_id"], "_source": doc_fields}

        dataset = get_dataset(self.dataset)
        jsonl_path, write_count = self.load()
        es_client = ElasticsearchClient(self.index_name, dataset, INDEX_SCHEMA)

        logger.info("Indexing to index %s", self.index_name)
        num_indexed = 0
        try:
            for batched_entries in tqdm(
                batched(self._load_jsonl(jsonl_path), self.index_batch_size),
                total=write_count // self.index_batch_size,
                desc="Indexing to Elasticsearch"
            ):
                logger.info("Indexing %d entries", len(batched_entries))
                success, _ = es_client.bulk(
                    operations=_make_bulk_insert_body(batched_entries),
                    index=self.index_name,
                    reset_index=True
                )
                num_indexed += success
        except BulkIndexError as e:
            logger.error("BulkIndexError")
            logger.error("errors: %s", e.errors[0])
            logger.error("errors: %s", e.errors[0]['index']['error'])
            raise e

        self.dump(self.index_name)


def dictionalize_worker(path: Path, vocab_dict: dict[int, str]) -> list[dict[str, float]]:
    docs, model_outputs = torch.load(path)
    expand_terms_list = dictionalize_model_outputs(model_outputs, vocab_dict)

    jsonl_lines: list[dict] = []
    for doc, term_weight in zip(docs, expand_terms_list):
        jsonl_line = doc.model_dump()
        jsonl_line[VECTOR_FIELD] = term_weight
    return jsonl_lines


class SpladeDictionalizeTask(BaseTask):
    dataset = luigi.Parameter()
    encoder_path = luigi.Parameter()

    encode_batch_size = luigi.IntParameter(default=32)

    def requires(self) -> TaskOnKart:
        return SpladeEncodeTask()

    def _output_dir(self) -> Path:
        return self.make_output_dir(f"splade/{self.encoder_path}/{self.dataset}/dictionalize")

    def output(self):
        return self.make_target(self._output_dir() / "output.pkl")

    def data_path(self) -> Path:
        return Path(self.workspace_directory) / self._output_dir()

    def run(self):
        start = time.perf_counter()
        partial_files_manager: PartialFilesManager = self.load()
        splade_encoder = SpladeEncoder(
            encoder_path=self.encoder_path, device="cpu"
        )

        jsonl_path = self.data_path() / "data.jsonl"
        if not jsonl_path.parent.exists():
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        write_count = 0
        with jsonl_path.open("w") as f:
            max_workers = 24
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
                futures = []
                for path in partial_files_manager.yield_pathes():
                    # docs, model_outputs = torch.load(path)
                    # terms_weights = splade_encoder.model_outputs_to_dict(model_outputs)
                    # for doc, term_weight in zip(docs, terms_weights):
                    #     jsonl_line = doc.model_dump()
                    #     jsonl_line[VECTOR_FIELD] = term_weight
                    #     f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
                    #     write_count += 1
                    future = executor.submit(
                        dictionalize_worker,
                        path,
                        splade_encoder.vocab_dict,
                    )

                for future in as_completed(futures):
                    jsonl_lines = future.result()
                    for jsonl_line in jsonl_lines:
                        f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
                        write_count += 1

        self.dump((jsonl_path, write_count))
        end = time.perf_counter()
        logger.info("Dictionalize took %f seconds", end - start)


class SpladeEncodeTask(BaseTask):
    dataset = luigi.Parameter()
    encoder_path = luigi.Parameter()
    debug = luigi.BoolParameter(default=False)

    device = luigi.Parameter(default="cpu")
    corpus_load_batch_size = luigi.IntParameter(default=10_000)
    clear_cache = luigi.BoolParameter(default=False)

    def _output_dir(self) -> Path:
        return self.make_output_dir(f"splade/{self.encoder_path}/{self.dataset}/encode")

    def output(self):
        return self.make_target(str(self._output_dir() / "partial_files_manager.pkl"))

    def _get_text_for_encode(self, docs: Iterable[Doc]) -> list[str]:
        texts = []
        for doc in docs:
            text = " ".join(
                [getattr(doc, field, "") for field in ["title", "url", "text"]]
            )
            texts.append(text)
        return texts

    def run(self):
        dataset = get_dataset(self.dataset)
        splade_encoder = SpladeEncoder(
            encoder_path=self.encoder_path, device=self.device
        )

        partial_files_dir = Path(self.workspace_directory) / self._output_dir()
        with PartialFilesManager(partial_files_dir) as partial_files_manager:
            for batch_docs in tqdm(
                batched(dataset.corpus_iter(self.debug), self.corpus_load_batch_size),
                total=dataset.docs_count // self.corpus_load_batch_size,
                desc="SPLADE Encoding"
            ):
                encodable_texts = self._get_text_for_encode(batch_docs)
                model_outputs = splade_encoder.encode_texts(encodable_texts)
                new_file = partial_files_manager.new_file()
                torch.save((batch_docs, model_outputs), new_file)

        self.dump(partial_files_manager)
