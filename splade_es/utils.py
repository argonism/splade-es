from pathlib import Path
import json
import logging
from io import TextIOWrapper
from typing import Any, Generator

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


class MiddleFileHandler(object):
    def __init__(self, name: str) -> None:
        self.path = CACHE_DIR / name
        make_dir_if_exists(self.path.parent)
        self.file: TextIOWrapper | None = None

    def __enter__(self) -> "MiddleFileHandler":
        if not self.path.parent.exists():
            make_dir_if_exists(self.path.parent)

        self.file = self.path.open("w+", encoding="utf-8")
        return self

    def jsonl(self, row: dict[str, Any]) -> None:
        if self.file is None:
            raise ValueError("File is not open")

        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def yield_from_head_as_jsonl(self) -> Generator[dict[str, Any], None, None]:
        if self.file is None:
            raise ValueError("File is not open")

        self.file.seek(0)
        for line in self.file:
            yield json.loads(line)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False
