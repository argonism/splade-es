from typing import TypeVar, Generic
from pathlib import Path

import luigi
import gokart
from gokart.target import TargetOnKart
from gokart.config_params import inherits_config_params


class MasterConfig(luigi.Config):
    output_dir = luigi.Parameter()
    dataset = luigi.Parameter()
    debug = luigi.BoolParameter(default=False)


G = TypeVar("G")


@inherits_config_params(MasterConfig)
class BaseTask(Generic[G], gokart.TaskOnKart[G]):
    output_dir = luigi.Parameter()
    debug = luigi.BoolParameter(default=False)

    @property
    def output_dir_path(self) -> Path:
        if self.debug:
            return Path(f"{self.output_dir}/debug")
        return Path(f"{self.output_dir}")

    def make_output_dir(self, path: str | Path) -> Path:
        return self.output_dir_path / path

    def cache_path(
        self, relative_file_path: str, use_unique_id: bool = False
    ) -> TargetOnKart:
        return super().make_target(
            str(self.make_output_dir(relative_file_path)), use_unique_id=use_unique_id
        )

    def workspace_path(self, relative_file_path: str) -> Path:
        return Path(self.workspace_directory) / self.make_output_dir(relative_file_path)
