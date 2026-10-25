"""Shared BrowseComp-Plus research agent for Conditions A/B/C/D.

Examples:
    python agent.py baseline
    python agent.py memgpt
    python agent.py llmlingua
    python agent.py baseline --questions questions.jsonl --limit 5
    python agent.py baseline --questions questions.jsonl --limit 120

JSONL questions: one object per line, e.g. {"id":"q1","question":"...","answer":"..."}.
The answer field is stored as reference metadata; this script does NOT grade answers.
"""
import argparse
import ast
import json
import operator
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, BadRequestError
from pyserini.search.lucene import LuceneSearcher

from context_managers import make_manager
from context_managers.base import history_tokens

SYSTEM = ("Answer the question using only BrowseComp-Plus documents. Search and open documents "
          "as needed. Record important evidence with take_note. When ready, call submit_answer. "
          "If evidence is insufficient, submit an answer stating that. Do not invent citations.")
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")


def schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}

BASE_TOOLS = [
    schema("search", "BM25 search. Returns document IDs and short snippets.",
           {"query": {"type": "string"}}, ["query"]),
    schema("open_document", "Read a document by its ID from a previous search.",
           {"doc_id": {"type": "string"}}, ["doc_id"]),
    schema("take_note", "Save a research fact and its source in the scratchpad.",
           {"text": {"type": "string"}, "doc_id": {"type": "string"}}, ["text", "doc_id"]),
    schema("calculate", "Calculate an arithmetic expression.",
           {"expression": {"type": "string"}}, ["expression"]),
    schema("submit_answer", "Finish the research episode with a short answer.",
           {"answer": {"type": "string"}}, ["answer"]),
]

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def calculate(expr):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("Only basic arithmetic is allowed")
    if len(expr) > 150:
        raise ValueError("Expression too long")
    return str(visit(ast.parse(expr, mode="eval")))


def load_searcher():
    index_path = Path("indexes/bm25")
    if not list(index_path.rglob("segments_*")):
        print("Downloading prebuilt BM25 index (first run only)...", flush=True)
        snapshot_download(repo_id="Tevatron/browsecomp-plus-indexes", repo_type="dataset",
                          allow_patterns="bm25/*", local_dir="indexes")
    roots = [p.parent for p in index_path.rglob("segments_*") if p.is_file()]
    if not roots:
        raise RuntimeError("No BM25 Lucene index found in indexes/bm25")
    searcher = LuceneSearcher(str(roots[0]))
    searcher.set_bm25(k1=0.9, b=0.4)
    return searcher


def get_document(searcher, doc_id):
    doc = searcher.doc(str(doc_id))
    if doc is None:
        return None
    raw = doc.raw()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return str(obj.get("contents") or obj.get("text") or obj.get("body") or raw)
    except (ValueError, TypeError):
        pass
    return str(raw)


def load_questions(path, limit, seed=None, exclude_ids=None):
    if not path:
        return [{"id": "manual", "question": input("You: ").strip()}]
    items = []
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            items = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get("questions", [])
    if exclude_ids:
        items = [item for i, item in enumerate(items) if str(item.get("id", i + 1)) not in exclude_ids] if all(isinstance(item, dict) for item in items) else items
    if seed is not None:
        random.Random(seed).shuffle(items)
    result = []
    for i, item in enumerate(items[:limit] if limit else items):
        if isinstance(item, str):
            item = {"question": item}
        question = item.get("question") or item.get("query")
        if not question:
            raise ValueError(f"Question #{i + 1} has no 'question' field")
        result.append({"id": str(item.get("id", i + 1)), "question": question,
                       "reference_answer": item.get("answer")})
    return result


class ResearchAgent:
    def __init__(self, searcher, client, condition, args):
        self.searcher, self.client, self.condition, self.args = searcher, client, condition, args
        self.events = []
        self.reset_usage()
        self.cm = make_manager(condition, llm=self.memory_llm, **json.loads(args.cm_args))
        self.tools = BASE_TOOLS + self.cm.extra_tools()

    def reset_usage(self):
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
                      "api_calls": 0, "api_seconds": 0.0, "estimated_cost_usd": 0.0}
        self.events = []

    def api_call(self, messages, purpose="agent", tools=None, step=None):
        # Internal metadata is used by context managers, not accepted by the API.
        clean = [{k: v for k, v in m.items() if k in
                  ("role", "content", "tool_calls", "tool_call_id", "name") and
                  not (k == "name" and m.get("role") == "tool")}
                 for m in messages]
        params = {"model": MODEL, "messages": clean, "temperature": 0}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        t0 = time.monotonic()
        for attempt in range(self.args.api_retries + 1):
            try:
                response = self.client.chat.completions.create(**params)
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                self.events.append({"purpose": purpose, "step": step, "retry": attempt + 1,
                                    "error_type": type(exc).__name__, "error": str(exc)[:500]})
                if attempt >= self.args.api_retries:
                    raise
                delay = min(30, 2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
        seconds = time.monotonic() - t0
        u = response.usage
        inp = (u.prompt_tokens or 0) if u else 0
        out = (u.completion_tokens or 0) if u else 0
        details = getattr(u, "prompt_tokens_details", None) if u else None
        cached = getattr(details, "cached_tokens", 0) or 0
        self.usage["input_tokens"] += inp
        self.usage["output_tokens"] += out
        self.usage["cached_input_tokens"] += cached
        self.usage["api_calls"] += 1
        self.usage["api_seconds"] += seconds
        # Rates are supplied explicitly; defaults of zero avoid inventing provider prices.
        cost = (max(0, inp - cached) * self.args.input_per_million +
                cached * self.args.cached_per_million +
                out * self.args.output_per_million) / 1_000_000
        self.usage["estimated_cost_usd"] += cost
        self.events.append({"purpose": purpose, "step": step, "input_tokens": inp,
                            "output_tokens": out, "cached_input_tokens": cached,
                            "context_estimated_tokens": history_tokens(messages),
                            "seconds": round(seconds, 3), "estimated_cost_usd": round(cost, 8)})
        return response.choices[0].message

    def memory_llm(self, messages, purpose):
        return self.api_call(messages, purpose=purpose).content or ""

    def run(self, task):
        # Instantiate a new manager for each question: no memory leaks between benchmark runs.
        self.reset_usage()
        self.cm = make_manager(self.condition, llm=self.memory_llm,
                               **json.loads(self.args.cm_args))
        self.tools = BASE_TOOLS + self.cm.extra_tools()
        history = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": task["question"]}]
        notes = []
        final_answer, stop_reason, steps = None, "step_limit", 0
        started = time.monotonic()
        for step in range(1, self.args.max_steps + 1):
            if time.monotonic() - started >= self.args.timeout:
                stop_reason = "timeout"
                break
            steps = step
            try:
                messages = self.cm.build_messages(history)
                assistant = self.api_call(messages, tools=self.tools, step=step)
            except BadRequestError as exc:
                msg = str(exc).lower()
                stop_reason = "context_overflow" if any(t in msg for t in
                    ("context length", "context window", "maximum context", "too many tokens")) else "api_error"
                self.events.append({"step": step, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
                break
            except Exception as exc:
                stop_reason = "api_error"
                self.events.append({"step": step, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
                break
            history.append(assistant.model_dump(exclude_none=True))
            if not assistant.tool_calls:
                # Require explicit submit_answer as specified by the proposal.
                if assistant.content:
                    print("Agent message (not submitted):", assistant.content[:300])
                history.append({"role": "user", "content":
                    "Please continue using the available tools. Finish by calling submit_answer."})
                continue
            for call in assistant.tool_calls:
                name = call.function.name
                arg = {}
                try:
                    arg = json.loads(call.function.arguments)
                    if not isinstance(arg, dict):
                        raise ValueError("Tool arguments must be an object")
                    result = self.cm.handle_tool(name, arg)
                    if result is None:
                        if name == "search":
                            hits = self.searcher.search(arg["query"], k=self.args.top_k)
                            snippets = []
                            for hit in hits:
                                text = get_document(self.searcher, hit.docid)
                                snippet = (text[:self.args.snippet_chars] if text else
                                           "[Text unavailable in index]")
                                snippets.append(f"Document {hit.docid}: {snippet}")
                            result = "\n\n".join(snippets) or "No matching documents."
                            print("Searching:", arg["query"])
                        elif name == "open_document":
                            text = get_document(self.searcher, arg["doc_id"])
                            result = text[:self.args.document_chars] if text else "Document not found or text unavailable."
                        elif name == "take_note":
                            note = {"fact": arg["text"], "source": str(arg["doc_id"])}
                            notes.append(note)
                            result = "Saved note: " + json.dumps(note, ensure_ascii=False)
                        elif name == "calculate":
                            result = calculate(arg["expression"])
                        elif name == "submit_answer":
                            final_answer = str(arg["answer"])
                            stop_reason = "submitted"
                            result = "Answer submitted."
                        else:
                            result = "Unknown tool: " + name
                    result = self.cm.on_observation(name, arg, str(result))
                except Exception as exc:
                    result = f"Tool error: {exc}"
                # Metadata is retained internally and stripped from API messages.
                msg = {"role": "tool", "tool_call_id": call.id, "content": result,
                       "name": name}
                if name == "open_document" and isinstance(arg, dict) and "doc_id" in arg:
                    msg["doc_id"] = str(arg["doc_id"])
                history.append(msg)
                self.events.append({"step": step, "tool": name, "args": arg,
                                    "result_chars": len(str(result)),
                                    "tool_error": str(result).startswith("Tool error:")})
            if final_answer is not None:
                break
        total_seconds = time.monotonic() - started
        return {"id": task["id"], "question": task["question"],
                "reference_answer": task.get("reference_answer"), "condition": self.condition,
                "answer": final_answer, "stop_reason": stop_reason, "steps": steps,
                "notes": notes, "usage": dict(self.usage), "manager_stats": self.cm.stats(),
                "total_seconds": round(total_seconds, 3), "events": self.events,
                "history": history if self.args.save_history else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["baseline", "memgpt", "llmlingua", "srcm"])
    p.add_argument("--questions", help="Path to JSONL or JSON question list; otherwise interactive")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Shuffle question order deterministically")
    p.add_argument("--exclude-ids", default=None, help="Text file of pilot IDs to exclude")
    p.add_argument("--api-retries", type=int, default=3)
    p.add_argument("--save-history", action="store_true", help="Store full research history; large output")
    p.add_argument("--resume", action="store_true", help="Append and skip IDs already in --out")
    p.add_argument("--out", default=None, help="Output JSONL path")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--timeout", type=int, default=900, help="Per-question timeout in seconds")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--snippet-chars", type=int, default=400)
    p.add_argument("--document-chars", type=int, default=12000)
    p.add_argument("--cm-args", default="{}", help='JSON manager settings, e.g. {"rate":0.5}')
    p.add_argument("--input-per-million", type=float, default=0.0)
    p.add_argument("--cached-per-million", type=float, default=0.0)
    p.add_argument("--output-per-million", type=float, default=0.0)
    args = p.parse_args()
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        p.error("Set OPENROUTER_API_KEY in .env")
    condition = {"baseline": "A", "memgpt": "B", "llmlingua": "C", "srcm": "D"}[args.mode]
    excluded = set()
    if args.exclude_ids:
        excluded = {line.strip() for line in Path(args.exclude_ids).read_text(encoding="utf-8").splitlines() if line.strip()}
    questions = load_questions(args.questions, args.limit, args.seed, excluded)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    searcher = load_searcher()
    agent = ResearchAgent(searcher, client, condition, args)
    out = Path(args.out or f"runs/{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.resume:
        p.error(f"Output already exists: {out}. Choose a different --out or pass --resume.")
    completed = set()
    if args.resume and out.exists():
        with out.open(encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    completed.add(str(json.loads(line)["id"]))
    if not any((args.input_per_million, args.cached_per_million, args.output_per_million)):
        print("WARNING: No API pricing supplied. estimated_cost_usd will be null, not $0.")
    with out.open("a" if args.resume else "x", encoding="utf-8") as f:
        for i, question in enumerate(questions, 1):
            if str(question["id"]) in completed:
                print("Skipping completed:", question["id"])
                continue
            print(f"\n[{i}/{len(questions)}] Condition {condition}: {question['id']}", flush=True)
            result = agent.run(question)
            if not any((args.input_per_million, args.cached_per_million, args.output_per_million)):
                result["usage"]["estimated_cost_usd"] = None
            print("Answer:", result["answer"])
            print("Stop:", result["stop_reason"], "Usage:", result["usage"])
            print("Context manager stats:", result["manager_stats"])
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    print("\nSaved:", out)
    print("Note: Reference answers are stored, but benchmark grading is not implemented.")


if __name__ == "__main__":
    main()
