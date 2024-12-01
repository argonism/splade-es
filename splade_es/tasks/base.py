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
    encoder_path = luigi.Parameter()

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

    def cache_path(self, relative_file_path: str) -> TargetOnKart:
        return super().make_target(str(self.output_dir_path / relative_file_path))
