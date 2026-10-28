"""Store optional local diagnostics as deduplicated content-addressed JSON."""

import json
from pathlib import Path

from .experiment import fingerprint


class DiagnosticStore:
    """Write diagnostic payloads once, returning their SHA-256 references."""

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else None

    def put(self, value):
        """Store a JSON payload when enabled, otherwise return None."""
        if self.directory is None:
            return None
        reference = fingerprint(value)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (reference + ".json")
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        return reference
