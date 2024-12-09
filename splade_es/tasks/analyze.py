from typing import Callable

import luigi
from mdfy import MdTable

from splade_es.tasks.base import BaseTask
from splade_es.dataset import get_dataset, Query
from splade_es.tasks.evaluate import (
    SearchModels,
    EnrichedEvalResultDoc,
    EnrichedEvalResultQuery,
    EnrichedEvalResult,
    EnrichQueryWiseEvalResultTask,
)


class ErrorQuery:
    def __init__(
        self,
        query: Query,
        first: EnrichedEvalResultQuery,
        second: EnrichedEvalResultQuery,
    ) -> None:
        self.query = query
        self.first = first
        self.second = second

    def __str__(self) -> str:
        return ""

    def _top_docs(
        self, result_query: EnrichedEvalResultQuery, top_n: int
    ) -> list[EnrichedEvalResultDoc]:
        docs = sorted(result_query.enriched_result, key=lambda x: x.rank)[:top_n]
        return docs

    @property
    def first_top_docs(self, top_n: int = 3) -> list[EnrichedEvalResultDoc]:
        return self._top_docs(self.first, top_n=top_n)

    @property
    def second_top_docs(self, top_n: int = 3) -> list[EnrichedEvalResultDoc]:
        return self._top_docs(self.second, top_n=top_n)


class ErrorQueryResult:
    def __init__(
        self,
        first_model: SearchModels,
        second_model: SearchModels,
        first_good_second_bad: list[ErrorQuery],
        first_bad_second_good: list[ErrorQuery],
        both_bad: list[ErrorQuery],
        both_good: list[ErrorQuery],
    ) -> None:
        self.first_model = first_model
        self.second_model = second_model
        self.first_good_second_bad = first_good_second_bad
        self.first_bad_second_good = first_bad_second_good
        self.both_bad = both_bad
        self.both_good = both_good

    def sorted_first_good_second_bad(
        self, on_first: bool = True, limit: int = 10
    ) -> list[ErrorQuery]:
        return sorted(
            self.first_good_second_bad,
            key=lambda x: x.first.eval_score if on_first else x.second.eval_score,
            reverse=True,
        )[:limit]

    def sorted_first_bad_second_good(
        self, on_first: bool = False, limit: int = 10
    ) -> list[ErrorQuery]:
        return sorted(
            self.first_bad_second_good,
            key=lambda x: x.first.eval_score if on_first else x.second.eval_score,
            reverse=True,
        )[:limit]

    def sorted_both_bad(self, limit: int = 10) -> list[ErrorQuery]:
        return sorted(
            self.both_bad,
            key=lambda x: x.first.eval_score + x.second.eval_score,
            reverse=True,
        )[:limit]


class ErrorQueriesTask(BaseTask):
    dataset = luigi.Parameter()
    search_models = luigi.EnumListParameter(enum=SearchModels)

    bad_threshold = luigi.FloatParameter(default=0.1)

    def requires(self):
        if len(self.search_models) > 2:
            raise ValueError("Only 2 search models are supported")

        return {
            search_model: EnrichQueryWiseEvalResultTask(search_model=search_model)
            for search_model in self.search_models
        }

    def output(self):
        models = "_".join([model.value for model in self.search_models])
        return self.cache_path(f"analyze/{self.dataset}/{models}_error_queries.pkl")

    def run(self):
        def print_error_queries(
            error_queries: list[ErrorQuery],
            first_model: str,
            second_model: str,
            sort_func: Callable[[ErrorQuery], float] = lambda x: x.first.eval_score,
        ):
            table_rows = []
            for error_query in sorted(error_queries, key=sort_func):
                table_rows.append(
                    {
                        "Query": error_query.query.text,
                        f"{first_model}": error_query.first.eval_score,
                        f"{second_model}": error_query.second.eval_score,
                    }
                )
            print(MdTable(table_rows, precision=2))

        dataset = get_dataset(self.dataset)
        eval_results: dict[SearchModels, EnrichedEvalResult] = self.load()
        query_table = dataset.get_query_table()

        first_model = self.search_models[0]
        second_model = self.search_models[1]

        metric = "ndcg@10"
        eval_result_first = {
            result.query.query_id: result
            for result in eval_results[first_model].enriched_queries[metric]
        }
        eval_result_second = {
            result.query.query_id: result
            for result in eval_results[second_model].enriched_queries[metric]
        }

        if len(eval_result_first) != len(eval_result_second):
            raise ValueError(
                "The number of queries in the two search models are different"
            )

        first_good_second_bad: list[ErrorQuery] = []
        first_bad_second_good: list[ErrorQuery] = []
        both_bad: list[ErrorQuery] = []
        both_good: list[ErrorQuery] = []
        for qid, first_result in eval_result_first.items():
            if qid not in eval_result_second:
                print(
                    f"Query {first_result.query} is missing in the second search model"
                )

            second_result = eval_result_second[qid]

            first_score = first_result.eval_score
            second_score = second_result.eval_score

            if first_score > self.bad_threshold and second_score <= self.bad_threshold:
                first_good_second_bad.append(
                    ErrorQuery(query_table[qid], first_result, second_result)
                )
            elif (
                first_score <= self.bad_threshold and second_score > self.bad_threshold
            ):
                first_bad_second_good.append(
                    ErrorQuery(query_table[qid], first_result, second_result)
                )
            elif first_score < self.bad_threshold and second_score < self.bad_threshold:
                both_bad.append(
                    ErrorQuery(query_table[qid], first_result, second_result)
                )
            elif (
                first_score >= self.bad_threshold and second_score >= self.bad_threshold
            ):
                both_good.append(
                    ErrorQuery(query_table[qid], first_result, second_result)
                )
            else:
                raise ValueError("Unexpected case")

        result = ErrorQueryResult(
            first_model,
            second_model,
            first_good_second_bad,
            first_bad_second_good,
            both_bad,
            both_good,
        )

        print(
            f":::::::::::::::: {first_model} good {second_model} bad ::::::::::::::::"
        )
        print_error_queries(
            first_good_second_bad,
            first_model.value,
            second_model.value,
            sort_func=lambda x: -x.first.eval_score,
        )
        print(
            f":::::::::::::::: {first_model} bad {second_model} good ::::::::::::::::"
        )
        print_error_queries(
            first_bad_second_good,
            first_model.value,
            second_model.value,
            sort_func=lambda x: -x.second.eval_score,
        )
        print(f":::::::::::::::: both bad ::::::::::::::::")
        print_error_queries(
            both_bad,
            first_model.value,
            second_model.value,
            sort_func=lambda x: -(x.first.eval_score + x.second.eval_score),
        )

        self.dump(result)


class SampleEnrichErrorQueriesTask(BaseTask):
    dataset = luigi.Parameter()
    search_models = luigi.EnumListParameter(enum=SearchModels)

    def requires(self):
        return ErrorQueriesTask(search_models=[SearchModels.BM25, SearchModels.SPLADE])

    def output(self):
        models = "_".join([model.value for model in self.search_models])
        return self.cache_path(f"analyze/{self.dataset}/{models}_top_error_queries.txt")

    def run(self):
        result: ErrorQueryResult = self.load()

        show_result = []
        top_n = 3

        # print(
        #     f":::::::::::::::: {SearchModels.BM25} good {SearchModels.SPLADE} bad ::::::::::::::::"
        # )
        # for enriched_error_query in result.sorted_first_good_second_bad(
        #     SearchModels.SPLADE, limit=3
        # ):
        #     print(enriched_error_query)
        #     print()

        print(
            f":::::::::::::::: {SearchModels.BM25} bad {SearchModels.SPLADE} good ::::::::::::::::"
        )
        for splade_enriched_error_query in result.sorted_first_bad_second_good(limit=3):
            print(splade_enriched_error_query.query)
            for doc in splade_enriched_error_query.second_top_docs:
                print(doc)
                show_result.append(str(doc))
            print()

        # print(
        #     f":::::::::::::::: both {SearchModels.BM25} and {SearchModels.SPLADE} bad ::::::::::::::::"
        # )
        # for enriched_error_query in result.sorted_both_bad(
        #     SearchModels.SPLADE, limit=3
        # ):
        #     print(enriched_error_query)
        #     print()
        self.dump("\n".join(show_result))
