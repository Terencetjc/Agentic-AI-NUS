"""Condition C — statistical compression with LLMLingua. Needs: pip install llmlingua"""
from __future__ import annotations

import hashlib
import os
import re
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
                 min_tokens: int = 80, chunk_tokens: int = 400):
        from llmlingua import PromptCompressor  # pip install llmlingua
        import time
        self._time = time
        self.rate = rate
        self.variant = variant
        self.compress_search = compress_search
        self.min_tokens = min_tokens  # don't bother compressing tiny texts
        self.chunk_tokens = chunk_tokens
        if not 32 <= chunk_tokens <= 400:
            raise ValueError("chunk_tokens must be between 32 and 400 for LLMLingua2")
        if variant == "llmlingua2":
            model_name = model_name or "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
            self.comp = PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map=device)
        else:
            model_name = model_name or "Qwen/Qwen2.5-0.5B"
            self.comp = PromptCompressor(model_name=model_name, device_map=device)
        self.model_name = model_name
        self.chunk_tokenizer = None
        if variant == "llmlingua2":
            from transformers import AutoTokenizer
            self.chunk_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._stats = {"compress_calls": 0, "cache_hits": 0, "compress_seconds": 0.0,
                       "tokens_before": 0, "tokens_after": 0, "chunks_processed": 0}

    def _chunks(self, text: str) -> list[str]:
        """Keep every token; never silently truncate a long observation."""
        if self.variant != "llmlingua2":
            return [text]
        ids = self.chunk_tokenizer.encode(text, add_special_tokens=False)
        return [self.chunk_tokenizer.decode(ids[i:i + self.chunk_tokens],
                skip_special_tokens=True) for i in range(0, len(ids), self.chunk_tokens)]

    def _compress(self, text: str, key: str) -> str:
        if count_tokens(text) < self.min_tokens:
            return text
        # Include a content hash: different observations with the same query must not
        # reuse an unrelated cached result. Version the key to invalidate old caches.
        cache_key = f"v2|{key}|{self.variant}|{self.model_name}|{self.rate}|{self.chunk_tokens}|{hashlib.sha256(text.encode()).hexdigest()}"
        h = hashlib.sha256(cache_key.encode()).hexdigest()
        path = os.path.join(self.cache_dir, h + ".txt")
        if os.path.exists(path):
            self._stats["cache_hits"] += 1
            with open(path, encoding="utf-8") as f:
                out = f.read()
        else:
            t0 = self._time.time()
            chunks = self._chunks(text)
            outputs = []
            for chunk in chunks:
                if self.variant == "llmlingua2":
                    res = self.comp.compress_prompt(
                        [chunk], rate=self.rate,
                        force_tokens=["\n", ".", "?", "!"],
                        force_reserve_digit=True,
                        drop_consecutive=True,
                    )
                else:
                    res = self.comp.compress_prompt([chunk], rate=self.rate)
                outputs.append(res["compressed_prompt"])
            out = "\n".join(outputs)
            self._stats["chunks_processed"] += len(chunks)
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
            # Preserve document identifiers exactly; compress only the snippet text.
            parts = re.split(r"(?=Document [^\n:]+:)", text)
            result = []
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                match = re.match(r"(Document [^\n:]+:)\s*", part)
                if match:
                    doc_label = match.group(1)
                    snippet = part[match.end():]
                    result.append(doc_label + " " + self._compress(
                        snippet, key=f"search:{args.get('query')}:{i}"))
                else:
                    result.append(self._compress(part, key=f"search:{args.get('query')}:{i}"))
            return "\n\n".join(result)
        return text

    def stats(self) -> dict:
        return dict(self._stats)

    def reset(self) -> None:
        self._stats = {k: 0 for k in self._stats}
        self._stats["compress_seconds"] = 0.0