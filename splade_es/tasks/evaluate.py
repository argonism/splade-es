import time
import enum

import luigi
import ranx

from splade_es.tasks.base import BaseTask
from splade_es.tasks.splade import SpladeSearchTask
from splade_es.dataset import get_dataset


class SearchModels(enum.Enum):
    BM25 = "bm25"
    SPLADE = "splade"

def get_search_task(search_model: SearchModels) -> type[BaseTask]:
    if search_model == SearchModels.SPLADE:
        return SpladeSearchTask
    else:
        raise ValueError(f"Unknown search model: {search_model}")

class EvaluateTask(BaseTask):
    dataset = luigi.Parameter()
    search_model = luigi.EnumParameter(enum=SearchModels)
    metrics = luigi.ListParameter(default=["ndcg", "ndcg@10", "ndcg@20", "recall", "recall@10", "recall@20"])

    def requires(self):
        return get_search_task(self.search_model)(rerun=self.rerun)

    def output(self):
        return self.cache_path(f"{self.search_model}/{self.dataset}/evaluate.pkl")

    def run(self):
        search_result = self.load()
        dataset = get_dataset(self.dataset)

        run = ranx.Run(search_result)
        eval_result = ranx.evaluate(dataset.qrels, run, metrics=self.metrics)
        print(eval_result)
        self.dump(eval_result)
