from typing import Iterable, Generator, Type
from abc import abstractmethod

from pydantic import BaseModel, Field
from ranx import Qrels


class Doc(BaseModel):
    doc_id: str = Field(...)


class Query(BaseModel):
    query_id: str = Field(...)

    @property
    def id(self):
        return self.query_id

class DatasetBase:
    name: str
    doc_type: Type[Doc]

    def __init_subclass__(cls, name: str = "base", doc_type: Type[Doc] = Doc) -> None:
        cls.name = name
        cls.doc_type = doc_type

    @abstractmethod
    def corpus_iter(self) -> Generator[Doc, None, None]: ...

    @property
    @abstractmethod
    def queries(self) -> Iterable[Query]: ...

    @property
    @abstractmethod
    def qrels(self) -> Qrels: ...

    @abstractmethod
    def fetch_doc(self, doc_id: str) -> Doc: ...

    def get_query_table(self) -> dict[str, Query]:
        return {query.id: query for query in self.queries}

    @property
    def doc_text_fields(self) -> list[str]:
        return [
            field
            for field, info in self.doc_type.model_fields.items()
            if not info.annotation == "doc_id"
        ]

    @property
    @abstractmethod
    def docs_count(self) -> int: ...
