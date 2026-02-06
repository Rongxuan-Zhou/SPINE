"""Minimal logging utilities stub."""

from __future__ import annotations


class PrintLogger:
    """Mirror stdout/stderr to a file-like object."""

    def __init__(self, path: str):
        self._file = open(path, "w", encoding="utf-8")

    def write(self, msg: str) -> None:
        self._file.write(msg)
        self._file.flush()

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
