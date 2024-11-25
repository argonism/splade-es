from collections import defaultdict
from typing import Generator, Iterable

import ir_datasets
from tqdm import tqdm

from splade_es.dataset.base import DatasetBase, Doc, Qrels, Query


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


class SciFactDoc(Doc):
    doc_id: str
    text: str
    title: str

    @property
    def id(self):
        return self.doc_id


class SciFact(BeirDataset, name="scifact", doc_type=SciFactDoc):
    def __init__(self) -> None:
        super().__init__("beir/scifact/test")


class ScidocsDoc(Doc):
    doc_id: str
    text: str
    title: str

    @property
    def id(self):
        return self.doc_id


class Scidocs(BeirDataset, name="scidocs", doc_type=ScidocsDoc):
    def __init__(self) -> None:
        super().__init__("beir/scidocs")


class TrecCovidDoc(Doc):
    doc_id: str
    text: str
    title: str

    @property
    def id(self):
        return self.doc_id


class TrecCovid(BeirDataset, name="trec-covid", doc_type=TrecCovidDoc):
    def __init__(self) -> None:
        super().__init__("beir/trec-covid")


class QuoraDoc(Doc):
    doc_id: str
    text: str

    @property
    def id(self):
        return self.doc_id


class Quora(BeirDataset, name="quora", doc_type=QuoraDoc):
    def __init__(self) -> None:
        super().__init__("beir/quora/test")


class FiQADoc(Doc):
    doc_id: str
    text: str

    @property
    def id(self):
        return self.doc_id


class FiQA(BeirDataset, name="fiqa", doc_type=FiQADoc):
    def __init__(self) -> None:
        super().__init__("beir/fiqa/test")


class Touchev2Doc(Doc):
    doc_id: str
    text: str
    title: str
    stance: str
    url: str

    @property
    def id(self):
        return self.doc_id


class Touchev2(BeirDataset, name="touche", doc_type=Touchev2Doc):
    def __init__(self) -> None:
        super().__init__("beir/webis-touche2020/v2")


class NQDoc(Doc):
    doc_id: str
    text: str
    title: str

    @property
    def id(self):
        return self.doc_id


class NQ(BeirDataset, name="nq", doc_type=NQDoc):
    def __init__(self) -> None:
        super().__init__("beir/nq")


class ClimateFeverDoc(Doc):
    doc_id: str
    text: str
    title: str

    @property
    def id(self):
        return self.doc_id


class ClimateFever(BeirDataset, name="climate-fever", doc_type=ClimateFeverDoc):
    def __init__(self) -> None:
        super().__init__("beir/climate-fever")
