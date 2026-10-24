"""Condition C — statistical compression with LLMLingua. Needs: pip install llmlingua"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

from .base import ContextManager, count_tokens


class LLMLinguaCompression(ContextManager):
    """
    Compresses document text and search snippets with LLMLingua BEFORE they enter the context.

    Compressed documents are cached on disk keyed by (doc_id, model, rate), because the corpus is
    fixed: each document is compressed at most once across ALL runs. Compression time is logged
    separately (the proposal reports it separately from API cost).

    Two variants, pick one in the pilot and state it in the report:
      * variant="llmlingua2" (default, fast): token-classification compressor
        (microsoft/llmlingua-2-xlm-roberta-large-meetingbank). NOTE: this is NOT entropy-based;
        if you use it, change "entropy-based" in the report to "learned token-level compression".
      * variant="llmlingua": the original perplexity/entropy method with a small causal LM
        (e.g. "Qwen/Qwen2.5-0.5B"). Matches the proposal's wording; slower.
    """
    name = "C_llmlingua"

    def __init__(self, rate: float = 0.33, variant: str = "llmlingua2",
                 model_name: Optional[str] = None, device: str = "cpu",
                 cache_dir: str = "cache/llmlingua", compress_search: bool = True,
                 min_tokens: int = 80):
        from llmlingua import PromptCompressor  # pip install llmlingua
        import time
        self._time = time
        self.rate = rate
        self.variant = variant
        self.compress_search = compress_search
        self.min_tokens = min_tokens  # don't bother compressing tiny texts
        if variant == "llmlingua2":
            model_name = model_name or "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
            self.comp = PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map=device)
        else:
            model_name = model_name or "Qwen/Qwen2.5-0.5B"
            self.comp = PromptCompressor(model_name=model_name, device_map=device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._stats = {"compress_calls": 0, "cache_hits": 0, "compress_seconds": 0.0,
                       "tokens_before": 0, "tokens_after": 0}

    # -- core ---------------------------------------------------------------------------------
    def _compress(self, text: str, key: str) -> str:
        if count_tokens(text) < self.min_tokens:
            return text
        h = hashlib.sha1(f"{key}|{self.model_name}|{self.rate}".encode()).hexdigest()
        path = os.path.join(self.cache_dir, h + ".txt")
        if os.path.exists(path):
            self._stats["cache_hits"] += 1
            with open(path, encoding="utf-8") as f:
                out = f.read()
        else:
            t0 = self._time.time()
            if self.variant == "llmlingua2":
                res = self.comp.compress_prompt(
                    [text], rate=self.rate,
                    force_tokens=["\n", ".", "?", "!"],
                    force_reserve_digit=True,   # keep numbers: the main risk named in the proposal
                    drop_consecutive=True,
                )
            else:
                res = self.comp.compress_prompt([text], rate=self.rate)
            out = res["compressed_prompt"]
            self._stats["compress_calls"] += 1
            self._stats["compress_seconds"] += self._time.time() - t0
            with open(path, "w", encoding="utf-8") as f:
                f.write(out)
        self._stats["tokens_before"] += count_tokens(text)
        self._stats["tokens_after"] += count_tokens(out)
        return out

    def on_observation(self, tool_name: str, args: dict, text: str) -> str:
        if tool_name == "open_document":
            doc_id = str(args.get("doc_id"))
            return f"[doc {doc_id}] " + self._compress(text, key=f"doc:{doc_id}")
        if tool_name == "search" and self.compress_search:
            # keep the doc IDs readable; compress only the snippet text after each ID line
            return self._compress(text, key=f"search:{args.get('query')}")
        return text  # notes, calculator, errors stay verbatim

    def stats(self) -> dict:
        return dict(self._stats)

    def reset(self) -> None:
        self._stats = {k: 0 for k in self._stats}
        self._stats["compress_seconds"] = 0.0