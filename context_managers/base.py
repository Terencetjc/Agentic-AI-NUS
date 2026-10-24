"""
context_managers/base.py — shared interface and token counting for all conditions for the DSA4213 Group 28 context-management study.

All conditions share ONE interface, so the agent loop never needs to know which one it is using.
The agent loop (owned by the agent person) should call the manager at exactly three points:

    obs_text = cm.on_observation(tool_name, args, raw_text)   # before a tool result enters history
    messages = cm.build_messages(history)                     # before every LLM call
    out      = cm.handle_tool(tool_name, args)                # first, for any tool call; None = not mine

plus  tools = BASE_TOOLS + cm.extra_tools()  when building the tool list.

`history` is the full, raw list of OpenAI-style chat messages:
    {"role": "system" | "user" | "assistant" | "tool", "content": str, ...}
  assistant messages may carry "tool_calls"; tool messages carry "tool_call_id" and
  (our convention) "name" = the tool name and "doc_id" when the tool was open_document.

Managers never delete messages (that would break tool_call / tool pairing); they only replace
`content`. Replacements are sticky, so the prefix stays identical between compactions and
prompt caching keeps working.

LLM calls made BY a manager (Condition B's memory writes) must go through the same logging
wrapper as the agent, so their cost counts toward that condition. Pass that wrapper in as
`llm(messages: list[dict], purpose: str) -> str`.
"""

from __future__ import annotations

import json
from typing import Optional

# --------------------------------------------------------------------------------------------
# Token counting (same counter for every condition, so comparisons are fair)
# --------------------------------------------------------------------------------------------
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:  # offline or tiktoken missing: rough approximation, fine for thresholds
    def count_tokens(text: str) -> int:
        return max(1, len(text or "") // 4)


def history_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += count_tokens(m.get("content") or "")
        if m.get("tool_calls"):
            total += count_tokens(json.dumps(m["tool_calls"]))
    return total


# --------------------------------------------------------------------------------------------
# Base interface
# --------------------------------------------------------------------------------------------
class ContextManager:
    name = "base"

    def on_observation(self, tool_name: str, args: dict, text: str) -> str:
        """Transform a tool result BEFORE it is appended to history. Default: unchanged."""
        return text

    def build_messages(self, history: list[dict]) -> list[dict]:
        """Return the messages actually sent to the model. Default: full history."""
        return history

    def extra_tools(self) -> list[dict]:
        """Extra tool schemas this condition adds (e.g. recall for B)."""
        return []

    def handle_tool(self, tool_name: str, args: dict) -> Optional[str]:
        """Handle a tool this manager owns; return None if it isn't ours."""
        return None

    def stats(self) -> dict:
        """Anything worth logging per episode (compactions, cache hits, ...)."""
        return {}

    def reset(self) -> None:
        """Clear per-question state when one manager is reused across questions."""