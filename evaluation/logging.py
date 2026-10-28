"""Read and append validated evaluation records in UTF-8 JSONL files."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar, Union

from .schemas import EpisodeRecord, LLMCallRecord

Record = TypeVar("Record", LLMCallRecord, EpisodeRecord)
FilePath = Union[str, Path]


def _append(path: FilePath, record: Record, record_type: type[Record]) -> None:
    """Validate and append one record, creating the file when needed."""
    if not isinstance(record, record_type):
        raise TypeError(f"expected {record_type.__name__}")
    # Metadata dictionaries can change even on a frozen record.
    data = asdict(record)
    record_type(**data)
    line = json.dumps(data, ensure_ascii=False, allow_nan=False)
    with Path(path).open("a+b") as stream:
        # Separate the new record from an existing final line.
        stream.seek(0, 2)
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write((line + "\n").encode("utf-8"))


def _read(path: FilePath, record_type: type[Record]) -> list[Record]:
    """Read records, reporting invalid data with its path and line number."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("each record must be a JSON object")
                records.append(record_type(**data))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return records


def append_call(path: FilePath, record: LLMCallRecord) -> None:
    """Append one API-attempt record to a caller-selected JSONL file."""
    _append(path, record, LLMCallRecord)


def read_calls(path: FilePath) -> list[LLMCallRecord]:
    """Load call records in file order, skipping blank lines."""
    return _read(path, LLMCallRecord)


def append_episode(path: FilePath, record: EpisodeRecord) -> None:
    """Append one finished episode record to a caller-selected JSONL file."""
    _append(path, record, EpisodeRecord)


def read_episodes(path: FilePath) -> list[EpisodeRecord]:
    """Load episode records in file order, skipping blank lines."""
    return _read(path, EpisodeRecord)
