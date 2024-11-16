from typing import Iterable, Generator
from collections import defaultdict

from tqdm import tqdm
import ir_datasets

from src.dataset.base import DatasetBase, Doc, Query, Qrels


class BeirQuery(Query):
    text: str


class BeirDataset(DatasetBase, name="beir", doc_type=Doc):
    def __init__(self, dataset_key: str, query_type: type[Query] = BeirQuery) -> None:
        self.dataset_key = dataset_key
        self.dataset = ir_datasets.load(self.dataset_key)

        self.query_type = query_type
        self.query_fields = query_type.model_fields

    def corpus_iter(self) -> Generator[Doc, None, None]:
        for doc in tqdm(
            self.dataset.docs_iter(),
            total=self.dataset.docs_count(),
            desc=f"{self.dataset_key} corpus iter:",
        ):
            model_fields = {}
            for field in self.doc_type.model_fields:
                model_fields[field] = getattr(doc, field, None)

            yield self.doc_type(**model_fields)

    @property
    def queries(self) -> Iterable[Query]:
        queries: Query = []
        for query in self.dataset.queries_iter():
            model_fields = {}
            for field in self.query_fields:
                model_fields[field] = getattr(query, field)
            queries.append(self.query_type(**model_fields))

        return queries

    @property
    def qrels(self) -> Qrels:
        return Qrels.from_ir_datasets(self.dataset_key)

    @property
    def docs_count(self) -> int:
        return self.dataset.docs_count()


class NfcorpusDoc(Doc):
    text: str
    title: str
    url: str

    @property
    def id(self):
        return self.doc_id


class Nfcorpus(BeirDataset, name="nfcorpus", doc_type=NfcorpusDoc):
    def __init__(self) -> None:
        super().__init__("beir/nfcorpus/test")


class ArguanaDoc(Doc):
    doc_id: str
    text: str
    title: str

    @property
    def id(self):
        return self.doc_id


class Arguana(BeirDataset, name="arguana", doc_type=ArguanaDoc):
    def __init__(self) -> None:
        super().__init__("beir/arguana")
