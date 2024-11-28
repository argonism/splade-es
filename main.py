import logging
import time
from argparse import ArgumentParser
from os import environ
from pathlib import Path
from types import UnionType
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
    suffix: str | None = Field(None)
    clear_cache: bool = Field(False)

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
                elif isinstance(info.annotation, UnionType):
                    if len(info.annotation.__args__ ) > 2:
                        raise ValueError("Annotation can not be more than 2")
                    kwargs["type"] = [type_ for type_ in info.annotation.__args__ if type_ is not None][0]
                else:
                    kwargs["type"] = info.annotation

            parser.add_argument(*arg_params, **kwargs)

        return cls(**vars(parser.parse_args()))


def main(args: Args):
    es_client = Elasticsearch(environ["ELASTICSEARCH_URL"])

    dataset = get_dataset(args.dataset)

    index_suffix = args.suffix if args.suffix else None
    if args.debug and index_suffix is None:
        index_suffix = "debug"
    search_model = get_search_model(args.model)(
        es_client,
        dataset,
        reset_index=args.reset_index,
        device=args.device,
        index_suffix=index_suffix
    )

    corpus: Iterable[Doc] = dataset.corpus_iter()
    if args.debug:
        corpus = []
        for i, doc in enumerate(dataset.corpus_iter()):
            if i > 10:
                break
            corpus.append(doc)

    start = time.perf_counter()
    search_model.index(corpus)
    end = time.perf_counter()
    logger.info("Indexing took %.2f seconds", end - start)

    start = time.perf_counter()
    search_result = search_model.search(dataset.queries)
    end = time.perf_counter()
    logger.info("Searching took %.2f seconds", end - start)

    run_path = run_dir / f"{search_model.index_name}.json"
    dump_to_json(search_result, run_path)

    # Print 5 search result examples:
    for result in list(search_result.values())[:5]:
        logger.debug("%s", {k:v for i, (k, v) in enumerate(result.items()) if i < 3})

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

    eval_path = eval_dir / f"{search_model.index_name}.json"
    dump_to_json(eval_result, eval_path)

    print(eval_result)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    args = Args.from_parse_args()
    logger.debug("args: %s", args)

    main(args)
