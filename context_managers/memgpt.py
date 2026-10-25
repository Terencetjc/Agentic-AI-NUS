"""Condition B — LLM-managed memory (MemGPT-style). Needs: pip install rank_bm25"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .base import ContextManager, history_tokens


@dataclass
class _ArchiveEntry:
    text: str
    source: str           # e.g. "doc 12345" or "summary@turn 7"
    turn: int


class MemGPTStyleMemory(ContextManager):
    """
    When the working context passes `max_tokens`, the OLDEST tool observations (except the last
    `keep_recent`) are evicted until the context is under `target_tokens`. Before eviction, an LLM
    call writes what matters from the evicted text into an external archive (MemGPT's "archival
    memory"). The evicted raw text is archived too. The agent gets a `recall(query)` tool that
    searches the archive with BM25.

    Simplification vs. MemGPT (state it in the report): memory writes are triggered by the
    harness at a size limit, not chosen by the agent at arbitrary times.
    """
    name = "B_memgpt"

    SUMMARY_PROMPT = (
        "You manage memory for a research agent answering this question:\n\n{question}\n\n"
        "The observations below are about to be removed from the agent's working context. "
        "Write the facts that could still matter for answering the question as short bullet points, "
        "each ending with its source in brackets, e.g. [doc 123]. Keep names, numbers and dates exactly. "
        "If nothing is relevant, write NONE.\n\n{observations}"
    )

    def __init__(self, llm: Callable[[list[dict], str], str], max_tokens: int = 24000,
                 target_tokens: int = 12000, keep_recent: int = 3, recall_k: int = 3,
                 archive_raw: bool = True):
        self.llm = llm
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.keep_recent = keep_recent
        self.recall_k = recall_k
        self.archive_raw = archive_raw
        self.archive: list[_ArchiveEntry] = []
        self.evicted: dict[str, str] = {}  # tool_call_id -> placeholder (sticky)
        self._stats = {"evictions": 0, "memory_llm_calls": 0, "recall_calls": 0, "evicted_observations": 0}

    # -- tools --------------------------------------------------------------------------------
    def extra_tools(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search your archived memory (facts and documents removed from "
                               "context earlier). Use when you need something you saw before.",
                "parameters": {"type": "object",
                               "properties": {"query": {"type": "string"}},
                               "required": ["query"]},
            },
        }]

    def handle_tool(self, tool_name: str, args: dict) -> Optional[str]:
        if tool_name != "recall":
            return None
        self._stats["recall_calls"] += 1
        if not self.archive:
            return "Archive is empty."
        from rank_bm25 import BM25Okapi  # pip install rank_bm25
        tok = lambda s: re.findall(r"\w+", s.lower())
        bm = BM25Okapi([tok(e.text) for e in self.archive])
        scores = bm.get_scores(tok(args.get("query", "")))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.recall_k]
        return "\n\n".join(f"[{self.archive[i].source}] {self.archive[i].text[:1500]}" for i in top)

    # -- context building ---------------------------------------------------------------------
    def _apply(self, history: list[dict]) -> list[dict]:
        out = []
        for m in history:
            if m.get("role") == "tool" and m.get("tool_call_id") in self.evicted:
                m = dict(m, content=self.evicted[m["tool_call_id"]])
            out.append(m)
        return out

    def build_messages(self, history: list[dict]) -> list[dict]:
        msgs = self._apply(history)
        if history_tokens(msgs) <= self.max_tokens:
            return msgs  # unchanged since last compaction -> prompt cache still hits

        question = next((m["content"] for m in history if m.get("role") == "user"), "")
        tool_idx = [i for i, m in enumerate(msgs)
                    if m.get("role") == "tool" and m.get("tool_call_id") not in self.evicted
                    and m.get("name") != "recall"]
        candidates = tool_idx[: max(0, len(tool_idx) - self.keep_recent)]

        batch, turn = [], len(history)
        for i in candidates:  # oldest first
            if history_tokens(msgs) <= self.target_tokens:
                break
            m = msgs[i]
            src = f"doc {m['doc_id']}" if m.get("doc_id") else f"{m.get('name', 'tool')}@{i}"
            batch.append((src, m["content"]))
            if self.archive_raw:
                self.archive.append(_ArchiveEntry(m["content"], src, turn))
            ph = f"[Removed from context to save space: {src}. Use recall(query) to retrieve it.]"
            self.evicted[m["tool_call_id"]] = ph
            msgs[i] = dict(m, content=ph)

        if batch:
            self._stats["evictions"] += 1
            self._stats["evicted_observations"] += len(batch)
            obs = "\n\n".join(f"--- {s} ---\n{t[:6000]}" for s, t in batch)
            notes = self.llm(
                [{"role": "user", "content": self.SUMMARY_PROMPT.format(question=question, observations=obs)}],
                "memory_write",
            )
            self._stats["memory_llm_calls"] += 1
            if notes and notes.strip().upper() != "NONE":
                self.archive.append(_ArchiveEntry(notes, f"summary@turn {turn}", turn))
        return msgs

    def stats(self) -> dict:
        return dict(self._stats, archive_size=len(self.archive),
                    max_tokens=self.max_tokens, target_tokens=self.target_tokens,
                    keep_recent=self.keep_recent)