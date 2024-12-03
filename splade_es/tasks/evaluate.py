import enum
from pathlib import Path
import json

import luigi
import ranx

from splade_es.tasks.base import BaseTask
from splade_es.tasks.splade import SpladeSearchTask
from splade_es.tasks.splade_doc import SpladeDocSearchTask
from splade_es.tasks.splade_phrase import SpladePhraseSearchTask
from splade_es.tasks.bm25 import BM25SearchTask
from splade_es.dataset import get_dataset


class SearchModels(enum.Enum):
    BM25 = "bm25"
    SPLADE = "splade"
    SPLADE_DOC = "splade-doc"
    SPLADE_PHRASE = "splade-phrase"


def get_search_task(search_model: SearchModels) -> type[BaseTask]:
    if search_model == SearchModels.SPLADE:
        return SpladeSearchTask
    elif search_model == SearchModels.SPLADE_DOC:
        return SpladeDocSearchTask
    elif search_model == SearchModels.SPLADE_PHRASE:
        return SpladePhraseSearchTask
    elif search_model == SearchModels.BM25:
        return BM25SearchTask
    else:
        raise ValueError(f"Unknown search model: {search_model}")

class EvaluateTask(BaseTask):
    dataset = luigi.Parameter()
    search_model = luigi.EnumParameter(enum=SearchModels)
    metrics = luigi.ListParameter(default=["ndcg", "ndcg@10", "ndcg@20", "recall", "recall@10", "recall@20"])

    def requires(self):
        return get_search_task(self.search_model)()

    @property
    def _output_dir(self) -> Path:
        return Path(f"{self.search_model.value}/{self.dataset}")

    def output(self):
        return self.cache_path(str(self._output_dir / "evaluate.pkl"))

    def run(self):
        search_result = self.load()
        dataset = get_dataset(self.dataset)

        run = ranx.Run(search_result)
        eval_result = ranx.evaluate(dataset.qrels, run, metrics=self.metrics)

        for metric, value in eval_result.items():
            print(f"{metric}: {value}")

        json_output_path = Path(self.workspace_directory) / self.make_output_dir(self._output_dir) / "evaluate.json"
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

        self.dump(eval_result)
