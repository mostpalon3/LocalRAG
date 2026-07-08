import os
from pathlib import Path
from typing import List

from config import DB_PATH


IGNORED_DIRECTORIES = {
    ".git",
    ".streamlit",
    ".streamlit_uploads",
    ".venv",
    "__pycache__",
    "build",
    Path(DB_PATH).as_posix(),
    "dist",
    "node_modules",
    "venv",
}


def collect_files(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [os.path.abspath(source_path)]

    if not os.path.isdir(source_path):
        return []

    collected_paths: List[str] = []
    for root, directories, filenames in os.walk(source_path):
        directories[:] = [
            directory
            for directory in directories
            if not _should_skip_path(os.path.join(root, directory))
        ]

        for filename in filenames:
            file_path = os.path.join(root, filename)
            if _should_skip_path(file_path):
                continue
            collected_paths.append(os.path.abspath(file_path))

    return sorted(collected_paths)


def _should_skip_path(path: str) -> bool:
    normalized_path = Path(path).as_posix()
    return any(ignored in normalized_path for ignored in IGNORED_DIRECTORIES)