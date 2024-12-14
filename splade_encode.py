from argparse import ArgumentParser
import json
from collections import OrderedDict

from pydantic import BaseModel, ConfigDict, Field
from mdfy import MdTable

from splade_es.tasks.splade import SpladeEncoder


class Args(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field("", alias="t")
    encoder_path: str = Field("naver/splade-v3", alias="e")

    output_vector: bool = Field(False, alias="v")

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


def main(args: Args) -> None:
    encoder = SpladeEncoder(encoder_path=args.encoder_path, device="cpu")
    sparse_dict = encoder.encode_to_dict([args.text])[0]
    sorted_sparse_dict = sorted(sparse_dict.items(), key=lambda x: x[1], reverse=True)

    ordered_sparse_dict = OrderedDict(sorted_sparse_dict)
    for key, value in sorted_sparse_dict:
        print(f"{key}: {value:.2f}")

    print(MdTable(ordered_sparse_dict, precision=2))
    if args.output_vector:
        print(json.dumps(sparse_dict, ensure_ascii=False))


if __name__ == "__main__":
    args = Args.from_parse_args()
    print(args)
    main(args)
