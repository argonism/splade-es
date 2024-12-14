import enum
from pathlib import Path
import json
from collections import defaultdict

import luigi
import ranx
from pydantic import BaseModel
from tqdm import tqdm
from mdfy import MdTable

from splade_es.tasks.base import BaseTask
from splade_es.tasks.splade import SpladeSearchTask, SpladeEncoder
from splade_es.tasks.splade_doc import SpladeDocSearchTask
from splade_es.tasks.splade_phrase import (
    SpladePhraseBoostSearchTask,
    SpladePhraseTitleBoostSearchTask,
    SpladeBoostPhraseSearchTask,
)
from splade_es.tasks.bm25 import BM25SearchTask
from splade_es.dataset import get_dataset, Query
from splade_es.utils import ElasticsearchClient


class SearchModels(enum.Enum):
    BM25 = "bm25"
    SPLADE = "splade"
    SPLADE_DOC = "splade-doc"
    SPLADE_PHRASE = "splade-phrase"
    SPLADE_TITLE = "splade-title"
    SPLADE_BOOST = "splade-boost"


class EvalResult:
    def __init__(
        self,
        scores: dict[str, float],
        scores_by_queries: dict[str, dict[str, float]],
        metrics: list[str],
        index_name: str,
    ):
        self.scores = scores
        self.scores_by_queries = scores_by_queries
        self.metrics = metrics
        self.index_name = index_name


def get_search_task(search_model: SearchModels) -> type[BaseTask]:
    if search_model == SearchModels.SPLADE:
        return SpladeSearchTask
    elif search_model == SearchModels.SPLADE_DOC:
        return SpladeDocSearchTask
    elif search_model == SearchModels.SPLADE_PHRASE:
        return SpladePhraseBoostSearchTask
    elif search_model == SearchModels.SPLADE_TITLE:
        return SpladePhraseTitleBoostSearchTask
    elif search_model == SearchModels.SPLADE_BOOST:
        return SpladeBoostPhraseSearchTask
    elif search_model == SearchModels.BM25:
        return BM25SearchTask
    else:
        raise ValueError(f"Unknown search model: {search_model}")


class EvaluateTask(BaseTask):
    dataset = luigi.Parameter()
    search_model = luigi.EnumParameter(enum=SearchModels)
    metrics = luigi.ListParameter(
        default=["ndcg", "ndcg@10", "ndcg@20", "recall", "recall@10", "recall@20"]
    )

    def requires(self):
        return get_search_task(self.search_model)(rerun=self.rerun)

    @property
    def _output_dir(self) -> Path:
        return Path(f"{self.search_model.value}/{self.dataset}")

    def output(self):
        return self.cache_path(str(self._output_dir / "evaluate.pkl"))

    def run(self):
        index_name, search_result = self.load()
        dataset = get_dataset(self.dataset)

        run = ranx.Run(search_result)
        eval_result = ranx.evaluate(dataset.qrels, run, metrics=self.metrics)

        for metric, value in eval_result.items():
            print(f"{metric}: {value}")

        json_output_base_path = Path(self.workspace_directory) / self.make_output_dir(
            self._output_dir
        )
        json_output_base_path.mkdir(parents=True, exist_ok=True)
        json_output_path = json_output_base_path / "evaluate.json"
        json_output_path.write_text(
            json.dumps(eval_result, indent=2, ensure_ascii=False)
        )
        json_output_path = json_output_base_path / "evaluate_by_query.json"
        json_output_path.write_text(
            json.dumps(run.scores, indent=2, ensure_ascii=False)
        )

        print(MdTable(eval_result, precision=2))

        self.dump(EvalResult(eval_result, run.scores, self.metrics, index_name))


class EnrichedEvalResultDoc(BaseModel):
    class PrettyEncoder(json.JSONEncoder):
        def _format_float_in_dict(self, dict_: dict) -> dict:
            for key, value in dict_.items():
                if isinstance(value, float):
                    dict_[key] = f"{value:.2f}"
                elif isinstance(value, dict):
                    dict_[key] = self._format_float_in_dict(value)
            return dict_

        def iterencode(self, o, **kwargs):
            if isinstance(o, dict):
                for key, value in o.items():
                    if isinstance(value, dict):
                        o[key] = self._format_float_in_dict(value)
                    elif isinstance(value, BaseModel):
                        o[key] = str(value)
            return super().iterencode(o, **kwargs)

    doc: dict
    rel: int | None
    score: float
    rank: int

    def __str__(self):
        return json.dumps(
            {
                "rank": self.rank,
                "Relevant": self.rel,
                "Doc": self.doc,
                "score": f"{self.score:.2f}",
            },
            ensure_ascii=False,
            indent=2,
            cls=self.PrettyEncoder,
        )


class EnrichedEvalResultQuery(BaseModel):
    enriched_result: list[EnrichedEvalResultDoc]
    query: Query
    eval_score: float


class EnrichedEvalResult(BaseModel):
    enriched_queries: dict[str, list[EnrichedEvalResultQuery]]
    metrics: list[str]
    index_name: str


class EnrichQueryWiseEvalResultTask(BaseTask):
    dataset = luigi.Parameter()
    search_model = luigi.EnumParameter(enum=SearchModels)

    def requires(self):
        return {
            "eval_result": EvaluateTask(
                dataset=self.dataset, search_model=self.search_model
            ),
            "search_result": get_search_task(self.search_model)(dataset=self.dataset),
        }

    def output(self):
        return self.cache_path(
            f"{self.search_model.value}/{self.dataset}/enriched_evaluate_by_query.pkl"
        )

    def run(self):
        result = self.load()
        eval_result: EvalResult = result["eval_result"]
        index_name, search_results = result["search_result"]

        # print(f"Enriching {search_results} search results")

        dataset = get_dataset(self.dataset)
        query_table = dataset.get_query_table()
        qrels = dataset.qrels

        es_client = ElasticsearchClient("", dataset, {})

        enriched_eval_result: dict[str, list[EnrichedEvalResultQuery]] = {}
        for metric, query_scores in eval_result.scores_by_queries.items():
            enriched_queries: list[EnrichedEvalResultQuery] = []
            for qid, eval_score in tqdm(query_scores.items()):
                query = query_table[qid]
                search_result = search_results[qid]
                if len(search_result) == 0:
                    enriched_queries.append(
                        EnrichedEvalResultQuery(
                            enriched_result=[], query=query, eval_score=eval_score
                        )
                    )
                    continue

                docs = {
                    doc["doc_id"]: doc
                    for doc in es_client.mget(list(search_result.keys()), index_name)
                }
                doc_rel = qrels[qid]

                enriched_docs = [
                    EnrichedEvalResultDoc(
                        doc=docs[docid],
                        rel=doc_rel[docid] if docid in doc_rel else None,
                        score=score,
                        rank=rank,
                    )
                    for rank, (docid, score) in enumerate(
                        sorted(search_result.items(), key=lambda x: x[1], reverse=True)
                    )
                ]
                enriched_queries.append(
                    EnrichedEvalResultQuery(
                        enriched_result=enriched_docs,
                        query=query,
                        eval_score=eval_score,
                    )
                )

            enriched_eval_result[metric] = enriched_queries

        self.dump(
            EnrichedEvalResult(
                enriched_queries=enriched_eval_result,
                metrics=eval_result.metrics,
                index_name=index_name[0],
            )
        )


class SpladeEnrichedEvalResultQuery(EnrichedEvalResultQuery):
    splade_query_vector: dict[str, float]


class SpladeEnrichedEvalResult(EnrichedEvalResult):
    enriched_queries: dict[str, list[SpladeEnrichedEvalResultQuery]]


class EnrichSpladeQueryResultTask(BaseTask):
    dataset = luigi.Parameter()
    search_model = luigi.EnumParameter(enum=SearchModels)

    encoder_path = luigi.Parameter()

    def requires(self):
        return EnrichQueryWiseEvalResultTask(
            dataset=self.dataset, search_model=self.search_model
        )

    def output(self):
        return self.cache_path(
            f"{self.search_model.value}/{self.dataset}/enriched_evaluate_by_query.pkl"
        )

    def run(self):
        result = self.load()
        eval_result: EnrichedEvalResult = result["eval_result"]
        splade_encoder = SpladeEncoder(encoder_path=self.encoder_path, device="cpu")

        metrics_splade_enriched_queries: dict[
            str, list[SpladeEnrichedEvalResultQuery]
        ] = {}
        for metric, enriched_queries in eval_result.enriched_queries.items():
            splade_enriched_queries = []
            for enriched_query in enriched_queries:
                query = enriched_query.query
                splade_dict = splade_encoder.encode_to_dict(query.text)

                splade_enriched_query = SpladeEnrichedEvalResultQuery(
                    **enriched_query.model_dump(), splade_query_vector=splade_dict
                )
                splade_enriched_queries.append(splade_enriched_query)
            metrics_splade_enriched_queries[metric] = splade_enriched_queries

        self.dump(
            SpladeEnrichedEvalResult(
                enriched_queries=metrics_splade_enriched_queries,
                metrics=eval_result.metrics,
                index_name=eval_result.index_name,
            )
        )
