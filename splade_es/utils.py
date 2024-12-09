from pathlib import Path
import json
import logging
import os
from typing import Iterable, Any, Sequence, Generator

from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from splade_es.dataset import DatasetBase

logger = logging.getLogger(__name__)


def make_dir_if_exists(path: Path) -> bool:
    """Create a directory if it does not exist.

    Args:
        path (Path): The path to the directory to create.

    Returns:
        bool: True if the directory was created, False otherwise
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return True
    return False


def dump_to_json(dict: dict, path: Path, indent: int = 2) -> None:
    if not path.parent.exists():
        make_dir_if_exists(path.parent)

    if path.suffix != ".json":
        logger.info("Changing file extension to .json")
        path = path.with_suffix(".json")

    if path.exists():
        logger.info("Overwriting existing file: %s", path)

    path.write_text(json.dumps(dict, indent=indent, ensure_ascii=False))


def get_elasticsearch_client() -> Elasticsearch:
    return Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9700"))


class ElasticsearchClient:
    def __init__(
        self, index_name: str, dataset: DatasetBase, index_schema: dict[str, dict]
    ) -> None:
        print(f"Creating Elasticsearch client with index {index_name}")
        self.client = get_elasticsearch_client()
        self.dataset = dataset
        self.index_schema = index_schema
        self.index_name = index_name

    def bulk(
        self,
        operations: Iterable[dict],
        index: str,
        refresh: bool = False,
        reset_index: bool = False,
    ) -> tuple[int, int | list[dict[str, Any]]]:
        self.setup_index(reset_index=reset_index)
        return bulk(self.client, operations, index=index, refresh=refresh)

    def msearch(self, searches: Sequence[dict]):
        return self.client.msearch(body=searches, index=self.index_name)

    def get(self, id: str, index: str):
        return self.client.get(index=index, id=id)

    def mget(self, ids: Sequence[str], index: str) -> list[dict[str, Any]]:
        def _produce_body(ids: Sequence) -> Generator[dict, None, None]:
            for id_ in ids:
                yield {"_id": id_, "_index": index}

        docs = self.client.mget(index=index, docs=list(_produce_body(ids)))["docs"]
        return [doc["_source"] for doc in docs if doc["found"]]

    def _make_properties(self) -> dict:
        properties = {}
        for field in self.dataset.doc_text_fields:
            properties[field] = {"type": "text", "analyzer": "standard"}
        return properties

    def setup_index(self, reset_index: bool = False) -> None:
        if self.client.indices.exists(index=self.index_name):
            logger.debug("Found exiting index %s", self.index_name)
            if reset_index:
                logger.info("Delete existing index %s", self.index_name)
                self.client.indices.delete(index=self.index_name)
            else:
                return

        mappings = self.index_schema["mappings"]
        mappings["properties"] = mappings["properties"] | self._make_properties()

        logger.info("Creating index %s", self.index_name)

        self.client.indices.create(
            index=self.index_name,
            settings=self.index_schema["settings"],
            mappings=mappings,
        )

        logger.info("complete index %s setup", self.index_name)


class PartialFilesManager:
    def __init__(
        self, file_dir: str, name: str = "partial", clear_cache: bool = False
    ) -> None:
        self.file_dir = Path(file_dir)
        self.name = name
        self._clear_cache = clear_cache

        self._write_count = 0

    def __enter__(self):
        if not self.file_dir.exists():
            self.file_dir.mkdir(parents=True, exist_ok=True)

        if self._clear_cache:
            self.clear_partial_files()
            self._write_count = 0

        if self._write_count != 0:
            raise ValueError("write count is not 0. Please do not reuse this object")

        logger.debug("Writing partial files to %s", self.file_dir)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._write_count = 0

    @property
    def pathes(self):
        return sorted(self.file_dir.iterdir(), key=lambda x: x.suffix)

    def clear_partial_files(self) -> None:
        for path in self.pathes:
            path.unlink()

    def yield_pathes(self) -> Generator[Path, None, None]:
        pathes = self.pathes
        for path in tqdm(pathes, total=len(pathes), desc="Yielding partial files"):
            if path.is_file() and path.stem == self.name:
                yield path

    def new_file(self) -> Path:
        path = self.file_dir / f"{self.name}.{self._write_count}"
        if path.exists():
            logger.warning("Found existing partial file %s", path)
        self._write_count += 1
        return path
