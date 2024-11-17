from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


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


def dump_to_json(dict: dict, path: Path, indent: int = 2) -> None:
    if not path.parent.exists():
        make_dir_if_exists(path.parent)

    if path.suffix != ".json":
        logger.info("Changing file extension to .json")
        path = path.with_suffix(".json")

    if path.exists():
        logger.info("Overwriting existing file: %s", path)

    path.write_text(json.dumps(dict, indent=indent, ensure_ascii=False))
