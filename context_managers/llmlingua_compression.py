"""Condition C — statistical compression with LLMLingua. Needs: pip install llmlingua

Speed fixes (after the first pilot, where compression took ~800 s of each 900 s question):
  * The compressor model is loaded ONCE per process and shared, even if the agent creates a
    new manager for every question.
  * Only opened documents are compressed by default. Search snippets are short, and compressing
    them was most of the calls (set compress_search=True to restore the old behaviour).
  * device="auto" uses the Mac GPU ("mps") when available, otherwise the CPU.
  * Compressed documents are cached on disk, so a document is compressed once across all runs.

Compression runs locally, so its time is reported separately (stats()["compress_seconds"])
and should NOT count toward the agent's API cost or timeout.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

from .base import ContextManager, count_tokens

# One loaded model per (variant, model, device) for the whole process.
_COMPRESSORS: dict = {}


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_compressor(variant: str, model_name: str, device: str):
    key = (variant, model_name, device)
    if key not in _COMPRESSORS:
        from llmlingua import PromptCompressor
        kwargs = {"model_name": model_name, "device_map": device}
        if variant == "llmlingua2":
            kwargs["use_llmlingua2"] = True
        try:
            _COMPRESSORS[key] = PromptCompressor(**kwargs)
        except Exception as exc:
            if device == "cpu":
                raise
            print(f"[llmlingua] could not load on {device} ({exc}); falling back to CPU", flush=True)
            kwargs["device_map"] = "cpu"
            _COMPRESSORS[key] = PromptCompressor(**kwargs)
    return _COMPRESSORS[key]


class LLMLinguaCompression(ContextManager):
    """
    Compresses opened documents (and optionally search snippets) BEFORE they enter the context.

    variant="llmlingua2" (default, fast): learned token-classification compressor
        (microsoft/llmlingua-2-xlm-roberta-large-meetingbank). Not entropy-based: describe it as
        "learned token-level compression" in the report.
    variant="llmlingua": the original perplexity/entropy method with a small causal LM.
    """
    name = "C_llmlingua"

    def __init__(self, rate: float = 0.33, variant: str = "llmlingua2",
                 model_name: Optional[str] = None, device: str = "auto",
                 cache_dir: str = "cache/llmlingua", compress_search: bool = False,
                 min_tokens: int = 80):
        self.rate = rate
        self.variant = variant
        self.compress_search = compress_search
        self.min_tokens = min_tokens
        if model_name is None:
            model_name = ("microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
                          if variant == "llmlingua2" else "Qwen/Qwen2.5-0.5B")
        self.model_name = model_name
        self.device = _pick_device(device)
        self.comp = _load_compressor(variant, model_name, self.device)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self._stats = {"compress_calls": 0, "cache_hits": 0, "compress_seconds": 0.0,
                       "tokens_before": 0, "tokens_after": 0, "device": self.device}

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
            t0 = time.monotonic()
            if self.variant == "llmlingua2":
                # LLMLingua-2 splits long text into 512-token chunks internally; the
                # "sequence length is longer than 512" tokenizer warning is harmless.
                res = self.comp.compress_prompt(
                    [text], rate=self.rate,
                    force_tokens=["\n", ".", "?", "!"],
                    force_reserve_digit=True,   # keep numbers
                    drop_consecutive=True,
                )
            else:
                res = self.comp.compress_prompt([text], rate=self.rate)
            out = res["compressed_prompt"]
            self._stats["compress_calls"] += 1
            self._stats["compress_seconds"] += time.monotonic() - t0
            with open(path, "w", encoding="utf-8") as f:
                f.write(out)
        self._stats["tokens_before"] += count_tokens(text)
        self._stats["tokens_after"] += count_tokens(out)
        return out

    def on_observation(self, tool_name: str, args: dict, text: str) -> str:
        if tool_name == "open_document":
            doc_id = str(args.get("doc_id"))
            if text.startswith("Document not found"):
                return text
            return f"[doc {doc_id}] " + self._compress(text, key=f"doc:{doc_id}")
        if tool_name == "search" and self.compress_search:
            return self._compress(text, key=f"search:{args.get('query')}")
        return text  # snippets (by default), notes, calculator output and errors stay verbatim

    def stats(self) -> dict:
        s = dict(self._stats)
        s["compress_seconds"] = round(s["compress_seconds"], 2)
        return s