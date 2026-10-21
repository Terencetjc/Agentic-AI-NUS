"""BrowseComp-Plus agents 
Run: python agent.py baseline|llmlingua|memgpt
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
MODEL = "anthropic/claude-sonnet-4.5"
MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"
if MODE not in ("baseline", "llmlingua", "memgpt"):
    raise ValueError("Choose baseline, llmlingua, or memgpt")

# Download the official index once; subsequent runs use the local copy.
print("Checking prebuilt BrowseComp-Plus BM25 index...", flush=True)
snapshot_download(
    repo_id="Tevatron/browsecomp-plus-indexes",
    repo_type="dataset",
    allow_patterns="bm25/*",
    local_dir="indexes",
)
# Lucene index contains segments_N. Locate it even if nested in bm25/.
index_roots = list(Path("indexes/bm25").rglob("segments_*"))
index_dirs = [p.parent for p in index_roots if p.is_file()]
if not index_dirs:
    raise RuntimeError("BM25 index files not found under indexes/bm25. Check the download.")
searcher = LuceneSearcher(str(index_dirs[0]))
searcher.set_bm25(k1=0.9, b=0.4)
print("BM25 index ready.", flush=True)

compressor = None
if MODE == "llmlingua":
    from llmlingua import PromptCompressor
    compressor = PromptCompressor()

usage = {"input": 0, "output": 0, "cached": 0, "calls": 0, "seconds": 0.0}

def ask(messages, tools=None):
    params = {"model": MODEL, "messages": messages}
    if tools:
        params["tools"] = tools
    start = time.monotonic()
    response = client.chat.completions.create(**params)
    usage["seconds"] += time.monotonic() - start
    usage["calls"] += 1
    if response.usage:
        u = response.usage
        usage["input"] += u.prompt_tokens or 0
        usage["output"] += u.completion_tokens or 0
        details = getattr(u, "prompt_tokens_details", None)
        usage["cached"] += getattr(details, "cached_tokens", 0) or 0
    return response.choices[0].message

def search(query):
    hits = searcher.search(query, k=3)
    output = []
    for hit in hits:
        doc = searcher.doc(hit.docid)
        raw = doc.raw() if doc else None
        if raw:
            try:
                data = json.loads(raw)
                content = data.get("contents") or data.get("text") or raw
            except json.JSONDecodeError:
                content = raw
        else:
            content = "[Text not stored in index]"
        output.append(f"Document {hit.docid}:\n{content[:2000]}")
    return "\n\n".join(output) or "No matching documents found."

TOOLS = [{"type": "function", "function": {
    "name": "search_corpus", "description": "Search BrowseComp-Plus documents",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
}}]
if MODE == "memgpt":
    TOOLS.append({"type": "function", "function": {
        "name": "recall_memory", "description": "Recall archived research notes",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }})

question = input("You: ")
messages = [
    {"role": "system", "content": "Answer using only retrieved BrowseComp-Plus documents. If evidence is insufficient, say so."},
    {"role": "user", "content": question},
]
archive = []

for step in range(20):
    # Simplified MemGPT-style memory: archive older completed exchanges.
    if MODE == "memgpt" and len(json.dumps(messages)) > 12000 and len(messages) > 4:
        # Only archive when the last assistant tool calls have all been answered.
        old = messages[2:-2]
        if old:
            summary = ask([
                {"role": "system", "content": "Summarize important research facts with document IDs. Do not invent facts."},
                {"role": "user", "content": json.dumps(old, ensure_ascii=False)[:20000]},
            ]).content or ""
            archive.append(summary)
            messages = messages[:2] + [{"role": "system", "content": "Saved research memory: " + "\n".join(archive)}] + messages[-2:]

    message = ask(messages, TOOLS)
    messages.append(message.model_dump(exclude_none=True))
    if not message.tool_calls:
        print("\nAnswer:", message.content)
        break

    for call in message.tool_calls:
        try:
            args = json.loads(call.function.arguments)
            if call.function.name == "recall_memory":
                result = "\n".join(archive[-5:]) or "No archived memories yet."
            elif call.function.name == "search_corpus":
                print("Searching:", args["query"])
                result = search(args["query"])
                if MODE == "llmlingua":
                    result = compressor.compress_prompt(result, rate=0.5)["compressed_prompt"]
            else:
                result = "Unknown tool"
        except Exception as exc:
            result = f"Tool error: {exc}"
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
else:
    print("Stopped: step limit")

print("\nToken usage:", usage)
print("Total tokens:", usage["input"] + usage["output"])
