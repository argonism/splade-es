import logging

from typing import Iterable
from tqdm import tqdm
from itertools import batched

from src.model.base import SearchModelBase, SearchResultType
from src.dataset.base import Doc, Query

logger = logging.getLogger(__name__)


INDEX_SCHEMA = {
    "settings": {},
    "mappings": {
        "properties": {
        }
    },
}


class ESBM25(SearchModelBase, model_name="bm25", index_schema=INDEX_SCHEMA):
    def index(self, docs: Iterable[Doc]) -> None:
        def make_bulk_insert_body(docs: Iterable[Doc]):
            for doc in docs:
                if not hasattr(doc, "id"):
                    raise ValueError(f"{doc} does not have id field")

                yield {"index": {"_index": self.index_name, "_id": doc.doc_id}}
                yield {
                    field: self._make_body(getattr(doc, field))
                    for field in self.dataset.doc_text_fields
                }

        self.client.bulk(operations=make_bulk_insert_body(docs), refresh=True)

    def search(self, queries: Iterable[Query]) -> SearchResultType:
        def yield_query(queries: Iterable):
            for query in queries:
                yield {"index": self.index_name}
                yield {
                    "query": {
                        "multi_match": {
                            "query": query.text,
                            "fields": self.dataset.doc_text_fields,
                        }
                    }
                }

        queries = list(queries)


        search_result = {}
        for batch_queries in tqdm(batched(queries, 500), desc="searching batch queries"):
            res = self.client.msearch(searches=yield_query(batch_queries), index=self.index_name)
            for response, query in zip(res["responses"], batch_queries):
                search_result[query.id] = {
                    hit["_id"]: hit["_score"] for hit in response["hits"]["hits"]
                }

        print("Retrieved %d search results", len(search_result))

        return search_result
