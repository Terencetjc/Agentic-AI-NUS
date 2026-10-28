"""Run unittest discovery with network and runtime-dependency guards enabled."""

import builtins
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = {"openai", "pyserini", "huggingface_hub", "tiktoken", "torch",
             "llmlingua", "datasets", "transformers", "dotenv"}


def main():
    """Guard discovery and tests, failing if even a caught network attempt occurs."""
    attempts = []
    original_import = builtins.__import__

    def deny(*args, **kwargs):
        attempts.append("network access")
        raise AssertionError("Network access is forbidden during offline verification")

    def guarded_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in FORBIDDEN and root not in sys.modules:
            attempts.append("runtime dependency import: " + name)
            raise AssertionError("Unexpected runtime dependency during offline verification: " + name)
        return original_import(name, *args, **kwargs)

    with ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.socket.connect_ex",
                     "socket.create_connection", "socket.getaddrinfo"):
            stack.enter_context(patch(name, deny))
        stack.enter_context(patch("builtins.__import__", guarded_import))
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("Network/runtime-dependency attempts:", len(attempts))
    return 0 if result.wasSuccessful() and not attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
