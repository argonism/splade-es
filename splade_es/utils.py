import json
import logging
from abc import abstractmethod
from io import TextIOWrapper
from pathlib import Path
from typing import Generator, Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def make_dir_if_exists(path: Path) -> bool:
    """Create a directory if it does not exist.

    Args:
        path (Path): The path to the directory to create.

    Returns:
        bool: True if the directory was created, False otherwise
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return True
    return False


def dump_to_json(json_encodable: dict, path: Path, indent: int = 2) -> None:
    if not path.parent.exists():
        make_dir_if_exists(path.parent)

    if path.suffix != ".json":
        logger.info("Changing file extension to .json")
        path = path.with_suffix(".json")

    if path.exists():
        logger.info("Overwriting existing file: %s", path)

    path.write_text(json.dumps(json_encodable, indent=indent, ensure_ascii=False))


T = TypeVar("T", bound=BaseModel)

class MiddleFileStep(Generic[T]):
    def __init__(self, name: str, base_dir: Path, data_model: type[T]) -> None:
        self.path = base_dir / name
        self.write_count = 0
        self.data_model = data_model

    def close(self) -> None:
        self.write_count = 0

    @abstractmethod
    def writeone(self, data: T, *args, **kwargs) -> None: ...

    @abstractmethod
    def read(self) -> Generator[T, None, None]: ...

    def __enter__(self) -> "MiddleFileStep":
        if not self.path.parent.exists():
            make_dir_if_exists(self.path.parent)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MiddleFileJsonlStep(MiddleFileStep[T]):
    def __init__(self, name: str, base_dir: Path, data_model: type[T]) -> None:
        super().__init__(name, base_dir, data_model)
        make_dir_if_exists(self.path.parent)

        self._file: TextIOWrapper | None = None

    def __enter__(self) -> "MiddleFileStep":
        self._file = self.path.open("w+")

        return self

    def close(self) -> None:
        super().close()
        if self._file is None:
            return

        self._file.close()

    def writeone(self, data: T, *args, **kwargs) -> None:
        if self._file is None:
            raise ValueError("file is not opened")

        self._file.write(
            json.dumps(data.model_dump(), ensure_ascii=False) + "\n"
        )
        self.write_count += 1

    def read(self) -> Generator[T, None, None]:
        if self._file is None:
            raise ValueError("file is not opened")

        self._file.seek(0)
        for line in self._file:
            entry = json.loads(line)
            yield self.data_model(**entry)


class MiddleFileIncrementalFileStep(MiddleFileStep):
    def __init__(self, name: str, base_dir: Path) -> None:
        self.path = base_dir / name
        make_dir_if_exists(self.path)
        self._file: TextIOWrapper | None = None

        self.write_count = 0

    def __enter__(self) -> "MiddleFileStep":
        if not self.path.parent.exists():
            make_dir_if_exists(self.path.parent)

        return self

    def get_new_path_to_file_incremental(self) -> Path:
        new_path = self.path / f"middle.{self.write_count}"
        self.write_count += 1
        return new_path

    def read_each_files(self) -> Generator[Path, None, None]:
        for file_path in sorted(self.path.iterdir(), key=lambda x: x.suffix):
            yield file_path

    def close(self) -> None:
        if self._file is None:
            return

        self._file.close()
        self.write_count = 0

class MiddleFileHandler(object):
    def __init__(self, name: str) -> None:
        self.path = CACHE_DIR / name
        make_dir_if_exists(self.path.parent)

        self._steps: dict[str, MiddleFileStep] = {}
        self.step_order: dict[int, str] = {}

    def __enter__(self) -> "MiddleFileHandler":
        if not self.path.parent.exists():
            make_dir_if_exists(self.path.parent)

        return self

    def step_with_incremental_files(self, name: str) -> MiddleFileIncrementalFileStep:
        step = MiddleFileIncrementalFileStep(name, self.path)
        self._steps[name] = step
        self.step_order[len(self._steps)] = name

        return step

    def step_with_jsonl(self, name: str, data_model: type[T]) -> MiddleFileJsonlStep:
        step = MiddleFileJsonlStep(name, self.path, data_model)
        self._steps[name] = step
        self.step_order[len(self._steps)] = name

        return step

    def get_step(self, name: str) -> MiddleFileStep:
        return self._steps[name]

    def __exit__(self, exc_type, exc_val, exc_tb):
        for step in self._steps.values():
            step.close()
        return False

    @property
    def previous_step(self) -> MiddleFileStep | None:
        previous_step_num = len(self.step_order) - 1
        if previous_step_num <= 0:
            return None

        pre_step_name = self.step_order[previous_step_num]
        return self._steps[pre_step_name]

"""
stepを生成するタイミングで、外からpydantic.basemodelを受け取って、それをやりとりできるようにしたい
"""