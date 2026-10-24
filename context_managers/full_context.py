"""Condition A — full context (control). Keeps every observation verbatim."""
from .base import ContextManager


class FullContext(ContextManager):
    """Keeps everything verbatim. Exists so the loop treats A exactly like B, C, D."""
    name = "A_full"