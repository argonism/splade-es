from abc import abstractmethod
from typing import Iterable, Optional
import logging

from elasticsearch import Elasticsearch

from splade_es.dataset.base import Doc, Query, DatasetBase


SearchResultType = dict[str, dict[str, float | int]]
logger = logging.getLogger(__name__)


class SearchModelBase(object):
    model_name = ""
    index_schema: dict[str, dict] = {}

    def __init__(
        self,
        es_client: Elasticsearch,
        dataset: DatasetBase,
        reset_index: bool = False,
        encoder_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.client = es_client
        self.encoder_path = encoder_path
        self.dataset = dataset

        self.setup_index(reset_index=reset_index)

    def __init_subclass__(cls, model_name: str, index_schema: dict, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.model_name = model_name
        cls.index_schema = index_schema

    @property
    def index_name(self) -> str:
        return f"{self.model_name}.{self.dataset.name}"

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

    @abstractmethod
    def index(self, docs: Iterable[Doc]) -> None: ...

    @abstractmethod
    def search(self, queries: Iterable[Query]) -> SearchResultType: ...

    def fetch_doc(self, doc_id: str) -> dict:
        res = self.client.get(index=self.index_name, id=doc_id)
        return res["_source"]
