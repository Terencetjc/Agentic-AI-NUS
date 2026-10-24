# context_managers — context-management conditions A, B, C

This package decides **what the model sees on each call**. The agent (`agent.py`) does the searching and answering. A context manager only changes how past observations are represented in the prompt. That is the one variable our study changes, so everything else stays identical across conditions.

| Condition | File | Class | Idea |
|---|---|---|---|
| A — Full context (control) | `full_context.py` | `FullContext` | Keep every observation verbatim |
| B — LLM-managed memory | `memgpt.py` | `MemGPTStyleMemory` | Past a size limit, an LLM saves key facts to an archive and old observations are removed; agent can `recall` them |
| C — Statistical compression | `llmlingua_compression.py` | `LLMLinguaCompression` | Compress documents with LLMLingua before they enter the context |
| D — SRCM | `srcm.py` (SRCM owner) | `SRCM` | Deterministic structural compaction (our method) |

## The interface (`base.py`)

Every condition subclasses `ContextManager` and the agent calls it at fixed points, so the agent never needs to know which condition is running:

| Method | Called by the agent | Default | Overridden by |
|---|---|---|---|
| `on_observation(tool, args, text) -> str` | after a tool returns, before the result is added to history | returns `text` unchanged | C |
| `build_messages(history) -> list` | before every LLM call | returns `history` unchanged | B |
| `extra_tools() -> list` | when building the tool list | `[]` | B (`recall`) |
| `handle_tool(name, args) -> str \| None` | on every tool call; `None` means "not mine" | `None` | B |
| `stats() -> dict` / `reset()` | after each question / between questions | `{}` | B, C |

`history` is a list of OpenAI-style chat messages. Tool messages carry two extra fields set by the agent: `name` (the tool name) and `doc_id` (for `open_document`). `agent.py` strips these before sending to the API.

Design rules every condition follows:

- **Never delete messages, only replace their content.** Deleting would separate a tool result from the assistant call that requested it, and the API rejects that.
- **Replacements are sticky.** Once an observation is evicted or compressed it stays that way. The prompt is then unchanged between compactions, so prompt caching still works.
- **One token counter for all conditions** (`count_tokens`, tiktoken `o200k_base`, falling back to chars/4 if unavailable), so size thresholds and savings are comparable.
- **Manager LLM calls are billed to the condition.** B receives the agent's logging wrapper as `llm(messages, purpose)`, so its memory-writing calls count toward B's tokens and cost.

## How each condition works

### A — `FullContext`
Inherits every default, so the full trajectory is sent on every call. It is the reference point for "% token reduction" and "% of baseline accuracy".

### B — `MemGPTStyleMemory` (after MemGPT, Packer et al. 2023)
On each `build_messages` call:

1. If the context is at most `max_tokens` (default 24,000), send it unchanged.
2. Otherwise, take the oldest tool observations, skipping the last `keep_recent` (default 3) and any `recall` results. Evict them until the context is under `target_tokens` (default 12,000).
3. Make **one LLM call** to write the facts from the evicted text that matter for the question, as bullet points with sources, into the archive. The raw evicted text is archived too.
4. Replace each evicted observation with a placeholder: `[Removed from context to save space: doc 123. Use recall(query) to retrieve it.]`

`recall(query)` searches the archive with BM25 (`rank_bm25`) and returns the top `recall_k` (default 3) entries.

Stats: `evictions`, `memory_llm_calls`, `recall_calls`, `archive_size`.

**Difference from MemGPT:** in MemGPT the agent decides when to save memory. Here the harness triggers it at a size limit. This keeps runs comparable and should be stated in the report.

### C — `LLMLinguaCompression` (after LLMLingua, Jiang et al. 2023)
In `on_observation`:

- `open_document`: the text is compressed and prefixed with `[doc <id>]`, so the ID is never lost.
- `search`: the snippets are compressed (turn off with `compress_search=False`).
- Notes, calculator output and errors are not compressed. Texts under `min_tokens` (default 80) are left alone.

Settings:

- **`rate`** (default 0.33) is the target fraction kept. The *measured* ratio is usually higher, because numbers (`force_reserve_digit=True`) and punctuation are always kept. Report the measured `tokens_after / tokens_before` from `stats()`.
- **`variant`**: `"llmlingua2"` (default) uses `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`, a fast learned token classifier. `"llmlingua"` uses the original perplexity-based method with a small causal LM (default `Qwen/Qwen2.5-0.5B`). Describe whichever we use accurately in the report: only the original is entropy-based.
- **`device`**: `"cpu"` by default.

**Caching:** each compressed document is saved to `cache/llmlingua/` under a hash of (doc ID, model, rate). The corpus is fixed, so a document is compressed at most once across all runs. Compression runs locally, so its time is logged separately (`compress_seconds`) and is not part of API cost.

Stats: `compress_calls`, `cache_hits`, `compress_seconds`, `tokens_before`, `tokens_after`.

## Usage

```python
from context_managers import make_manager

cm = make_manager("A")
cm = make_manager("B", llm=logged_llm, max_tokens=24000, target_tokens=12000, keep_recent=3)
cm = make_manager("C", rate=0.33, variant="llmlingua2")

# or import one class directly
from context_managers.memgpt import MemGPTStyleMemory
```

From the batch runner, settings are passed as JSON without editing code:

```
python agent.py --condition C --ids pilot_ids.txt --out runs/C_r033 --cm-args '{"rate": 0.33}'
```

Conditions are imported lazily, so running A or B does not require `llmlingua`.

## Adding SRCM (condition D)

Create `srcm.py` with `class SRCM(ContextManager)` and override the methods you need. Most likely that is `build_messages` for compaction and `on_observation` to record metadata. `make_manager("D")` and `agent.py --condition D` already point to it.

## Requirements

`tiktoken`, `rank_bm25` (B), `llmlingua` (C). The first run of C downloads the compressor model (a few GB).

## Parameters to freeze after the pilot

| Condition | Parameter | Pilot values | Frozen |
|---|---|---|---|
| B | `max_tokens` / `target_tokens` | 16k/8k, 32k/16k | |
| B | `keep_recent` | 3 | |
| C | `rate` | 0.33, 0.5 | |
| C | `variant` | llmlingua2 | |