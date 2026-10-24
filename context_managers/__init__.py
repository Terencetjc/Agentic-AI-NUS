"""
Context managers for the DSA4213 Group 28 study. One file per condition:

    base.py                   ContextManager interface, count_tokens, history_tokens
    full_context.py           A  FullContext
    memgpt.py          B  MemGPTStyleMemory      (needs rank_bm25)
    llmlingua_compression.py  C  LLMLinguaCompression   (needs llmlingua)
    srcm.py                   D  SRCM                   (add here, subclass ContextManager)

Use one directly:
    from context_managers.memgpt import MemGPTStyleMemory
or by condition letter:
    from context_managers import make_manager
    cm = make_manager("B", llm=my_logged_llm, max_tokens=24000)

Conditions are imported lazily, so using A or B never requires llmlingua to be installed.
"""
from __future__ import annotations

from typing import Callable, Optional

from .base import ContextManager, count_tokens, history_tokens


def make_manager(condition: str, llm: Optional[Callable] = None, **kw) -> ContextManager:
    if condition == "A":
        from .full_context import FullContext
        return FullContext()
    if condition == "B":
        assert llm is not None, "Condition B needs the logged llm() wrapper"
        from .memgpt import MemGPTStyleMemory
        return MemGPTStyleMemory(llm=llm, **kw)
    if condition == "C":
        from .llmlingua_compression import LLMLinguaCompression
        return LLMLinguaCompression(**kw)
    if condition == "D":
        from .srcm import SRCM  # the SRCM owner adds context_managers/srcm.py
        return SRCM(**kw)
    raise ValueError(f"unknown condition {condition!r}")


__all__ = ["ContextManager", "count_tokens", "history_tokens", "make_manager"]