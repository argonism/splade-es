import logging
from argparse import ArgumentParser
from os import environ
from pathlib import Path
from typing import Iterable

import ranx
from elasticsearch import Elasticsearch
from pydantic import BaseModel, ConfigDict, Field

from splade_es.dataset import Doc, get_dataset
from splade_es.model import get_search_model
from splade_es.utils import dump_to_json, make_dir_if_exists

logger = logging.getLogger("splade_es")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s-[%(name)s][%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

output_base_dir = Path(__file__).parent / "output"
make_dir_if_exists(output_base_dir)

run_dir = output_base_dir / "run"
make_dir_if_exists(run_dir)

eval_dir = output_base_dir / "eval"
make_dir_if_exists(eval_dir)


class Args(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset: str = Field("nfcorpus", alias="d")
    model: str = Field("bm25", alias="m")
    device: str = Field("cpu")

    reset_index: bool = Field(False)
    debug: bool = Field(False)

    @classmethod
    def from_parse_args(cls) -> "Args":
        parser = ArgumentParser()
        for field_name, info in cls.model_fields.items():
            arg_params = []
            if info.alias is not None:
                arg_params.append(f"-{info.alias}")
            arg_params.append(f"--{field_name}")

            kwargs = {}
            if info.default is not None:
                kwargs["default"] = info.default
            if info.annotation is not None:
                if info.annotation is bool:
                    kwargs["action"] = "store_true"
                else:
                    kwargs["type"] = info.annotation

            parser.add_argument(*arg_params, **kwargs)

        return cls(**vars(parser.parse_args()))


def main(args: Args):
    es_client = Elasticsearch(environ["ELASTICSEARCH_URL"])

    dataset = get_dataset(args.dataset)

    search_model = get_search_model(args.model)(
        es_client, dataset, reset_index=args.reset_index
    )

    corpus: Iterable[Doc] = dataset.corpus_iter()
    if args.debug:
        corpus = [doc for i, doc in enumerate(dataset.corpus_iter()) if i < 3]

    search_model.index(corpus)
    search_result = search_model.search(dataset.queries)

    run_path = run_dir / f"{args.model}/{args.dataset}.json"
    dump_to_json(search_result, run_path)

    # Print 5 search result examples:
    for result in list(search_result.values())[:5]:
        logger.debug("%s", result)

    run = ranx.Run(search_result)
    metrics = [
        "ndcg",
        "ndcg@10",
        "ndcg@20",
        "recall",
        "recall@10",
        "recall@20",
    ]
    eval_result = ranx.evaluate(dataset.qrels, run, metrics=metrics)

    eval_path = eval_dir / f"{args.model}/{args.dataset}.json"
    dump_to_json(eval_result, eval_path)

    print(eval_result)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    args = Args.from_parse_args()
    logger.debug("args: %s", args)

    main(args)
