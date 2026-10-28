import argparse
import ast
import json
import operator
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.diagnostics import DiagnosticStore
from evaluation.experiment import (code_revision, estimate_cost, fingerprint, manager_settings,
                                  pricing_from_args, research_configuration, resolve_model)
from evaluation.metrics import complete_sum

SYSTEM = ("Answer questions using only BrowseComp-Plus documents. "
          "Search for relevant documents using BM25 and open promising documents to read them. "
          "Avoid repeating searches that have already returned insufficient evidence. "
          "If a search was repeated, open a relevant document or try a substantially different query. "
          "Use take_note to save important facts and document IDs. "
          "When you have enough evidence, call submit_answer. "
          "If evidence remains insufficient, submit an answer stating that. Do not invent citations.")


def bootstrap_model(args):
    """Load local configuration before resolving the research model."""
    from dotenv import load_dotenv
    load_dotenv()
    args.model, args.model_source = resolve_model(args.model, os.environ)
    return args.model


def create_client():
    """Create the configured client when experimental execution is requested."""
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def make_manager(*args, **kwargs):
    """Load the selected context manager at runtime."""
    from context_managers import make_manager as factory
    return factory(*args, **kwargs)


def history_tokens(messages):
    """Load the shared threshold counter at runtime."""
    from context_managers.base import history_tokens as counter
    return counter(messages)

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

def load_searcher(directory="indexes/bm25"):
    """Open an existing local BM25 index; resource preparation is separate."""
    index_path = Path(directory)
    roots = [p.parent for p in index_path.rglob("segments_*") if p.is_file()]
    if not roots:
        raise RuntimeError(f"No local BM25 Lucene index found in {index_path}")
    from pyserini.search.lucene import LuceneSearcher
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
    def __init__(self, searcher, client, condition, args, manager_factory=None,
                 token_counter=None, clock=None, diagnostics=None):
        self.searcher, self.client, self.condition, self.args = searcher, client, condition, args
        self.events = []
        self.model = args.model
        self.manager_factory = manager_factory or make_manager
        self.token_counter = token_counter or history_tokens
        self.clock = clock or time
        self.diagnostics = diagnostics or DiagnosticStore()
        self.pricing = pricing_from_args(args)
        self.config_id = getattr(args, "config_id", None)
        self.reset_usage()

    def reset_usage(self):
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
                      "api_calls": 0, "api_seconds": 0.0, "estimated_cost_usd": 0.0}
        self.events = []

    def api_call(self, messages, purpose="agent", tools=None, step=None):
        """Record each wrapper attempt while preserving existing SDK retry behaviour."""
        clean = [{k: v for k, v in m.items() if k in
                  ("role", "content", "tool_calls", "tool_call_id", "name") and
                  not (k == "name" and m.get("role") == "tool")}
                 for m in messages]
        params = {"model": self.model, "messages": clean, "temperature": 0,
                  "max_tokens": self.args.max_output_tokens}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        request_ref = self.diagnostics.put(params)
        for attempt in range(self.args.api_retries + 1):
            t0 = self.clock.monotonic()
            event = {"kind": "llm_call", "call_id": f"{self.episode_id}:call:{self.call_number}",
                     "purpose": purpose, "step": step, "attempt": attempt + 1,
                     "requested_model": self.model, "returned_model": None, "response_id": None,
                     "input_tokens": None, "output_tokens": None, "cached_input_tokens": None,
                     "estimated_cost_usd": None, "context_estimated_tokens": self.token_counter(messages),
                     "model_input_ref": request_ref, "measurement_scope": "wrapper attempt; SDK may retry internally",
                     "sdk_max_retries": getattr(self.client, "max_retries", None)}
            self.call_number += 1
            try:
                response = self.client.chat.completions.create(**params)
            except Exception as exc:
                event.update(status="error", seconds=self.clock.monotonic() - t0,
                             error_type=type(exc).__name__, error=self._error_text(exc))
                self.events.append(event)
                self._update_usage()
                retryable = any(cls.__name__ in ("RateLimitError", "APIConnectionError",
                                "APITimeoutError", "InternalServerError") for cls in type(exc).__mro__)
                if not retryable or attempt >= self.args.api_retries:
                    raise
                delay = min(30, 2 ** attempt) + random.uniform(0, 0.5)
                wait_started = self.clock.monotonic()
                self.clock.sleep(delay)
                self.events.append({"kind": "retry_wait", "seconds": self.clock.monotonic() - wait_started,
                                    "purpose": purpose, "step": step})
                continue
            seconds = self.clock.monotonic() - t0
            usage = response.usage
            inputs = getattr(usage, "prompt_tokens", None)
            outputs = getattr(usage, "completion_tokens", None)
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
            measurements = {"input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached}
            invalid = []
            for key, value in measurements.items():
                if value is not None and (type(value) is not int or value < 0):
                    measurements[key] = None
                    invalid.append(key)
            inputs, outputs, cached = (measurements[key] for key in
                                       ("input_tokens", "output_tokens", "cached_input_tokens"))
            if inputs is not None and cached is not None and cached > inputs:
                cached = None
                invalid.append("cached_input_tokens exceeds input_tokens")
            event.update(status="success", seconds=seconds, input_tokens=inputs, output_tokens=outputs,
                         cached_input_tokens=cached, returned_model=getattr(response, "model", None),
                         response_id=getattr(response, "id", None),
                         estimated_cost_usd=estimate_cost(inputs, outputs, cached, self.pricing),
                         usage_validation_errors=invalid)
            event["response_message_ref"] = self.diagnostics.put(response.choices[0].message.model_dump(exclude_none=True))
            self.events.append(event)
            self._update_usage()
            return response.choices[0].message

    @staticmethod
    def _error_text(exc):
        """Retain a bounded error description with common credential forms redacted."""
        import re
        text = re.sub(r"sk-[A-Za-z0-9_-]+|Bearer\s+\S+", "[redacted]", str(exc), flags=re.I)
        return text[:1000]

    def _update_usage(self):
        """Reconcile attempt measurements without replacing unknown values."""
        calls = [e for e in self.events if e.get("kind") == "llm_call"]
        self.usage = {key: complete_sum(e[key] for e in calls) for key in
                      ("input_tokens", "output_tokens", "cached_input_tokens", "estimated_cost_usd")}
        self.usage.update(api_calls=len(calls), successful_calls=sum(e["status"] == "success" for e in calls),
                          api_seconds=sum(e["seconds"] for e in calls),
                          retry_wait_seconds=sum(e["seconds"] for e in self.events if e.get("kind") == "retry_wait"))

    def memory_llm(self, messages, purpose):
        return self.api_call(messages, purpose=purpose).content or ""

    def run(self, task):

        self.reset_usage()
        self.call_number = 0
        self.episode_id = f"{self.condition}:{task['id']}"
        initialization_started = self.clock.monotonic()
        self.cm = self.manager_factory(self.condition, llm=self.memory_llm,
                                       **manager_settings(self.args, self.condition))
        self.tools = BASE_TOOLS + self.cm.extra_tools()
        initialization_seconds = self.clock.monotonic() - initialization_started
        history = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": task["question"]}]
        notes = []
        search_cache = {}
        repeated_searches = 0
        final_answer, stop_reason, steps = None, "step_limit", 0
        started = self.clock.monotonic()
        for step in range(1, self.args.max_steps + 1):
            if self.clock.monotonic() - started >= self.args.timeout:
                stop_reason = "timeout"
                break
            steps = step
            try:
                messages = self.cm.build_messages(history)
                assistant = self.api_call(messages, tools=self.tools, step=step)
            except Exception as exc:
                stop_reason = "api_error"
                if any(cls.__name__ == "BadRequestError" for cls in type(exc).__mro__):
                    msg = str(exc).lower()
                    if any(t in msg for t in ("context length", "context window", "maximum context", "too many tokens")):
                        stop_reason = "context_overflow"
                self.events.append({"kind": "episode_error", "step": step,
                                    "error_type": type(exc).__name__, "error": self._error_text(exc)})
                break
            history.append(assistant.model_dump(exclude_none=True))
            if not assistant.tool_calls:

                if assistant.content:
                    print("Agent message (not submitted):", assistant.content[:300])
                history.append({"role": "user", "content":
                    "Please continue using the available tools. Finish by calling submit_answer."})
                continue
            for call in assistant.tool_calls:
                name = call.function.name
                arg = {}
                detail = {}
                tool_error = False
                raw_result_ref = None
                try:
                    arg = json.loads(call.function.arguments)
                    if not isinstance(arg, dict):
                        raise ValueError("Tool arguments must be an object")
                    result = self.cm.handle_tool(name, arg)
                    if result is None:
                        if name == "search":
                            query = arg["query"]
                            key = " ".join(query.casefold().split())
                            if key in search_cache:
                                repeated_searches += 1
                                ids = search_cache[key]
                                result = ("This exact search was already performed. "
                                          "Do not repeat it. Open one of these documents "
                                          "or use a substantially different query. "
                                          "Previously returned document IDs: " +
                                          (", ".join(ids) if ids else "none"))
                                print("Repeated search:", query)
                                detail.update(returned_docids=ids, repeated_search=True)
                            else:
                                hits = self.searcher.search(query, k=self.args.top_k)
                                snippets = []
                                ids = []
                                for hit in hits:
                                    ids.append(str(hit.docid))
                                    text = get_document(self.searcher, hit.docid)
                                    snippet = (text[:self.args.snippet_chars] if text else
                                               "[Text unavailable in index]")
                                    snippets.append(f"Document {hit.docid}: {snippet}")
                                search_cache[key] = ids
                                detail.update(returned_docids=ids, repeated_search=False)
                                result = "\n\n".join(snippets) or "No matching documents."
                                print("Searching:", query)
                        elif name == "open_document":
                            text = get_document(self.searcher, arg["doc_id"])
                            result = text[:self.args.document_chars] if text else "Document not found or text unavailable."
                            tool_error = not bool(text)
                            length = len(text) if text is not None else None
                            detail.update(document_status="found" if text else "unavailable",
                                          original_chars=length, returned_start=0 if text else None,
                                          returned_end=len(result) if text else None,
                                          truncated=(length > len(result)) if text else None)
                        elif name == "take_note":
                            note = {"fact": arg["text"], "source": str(arg["doc_id"])}
                            notes.append(note)
                            result = "Saved note: " + json.dumps(note, ensure_ascii=False)
                        elif name == "calculate":
                            result = calculate(arg["expression"])
                        elif name == "submit_answer":
                            if not isinstance(arg.get("answer"), str) or not arg["answer"].strip():
                                raise ValueError("Submitted answer must be a nonempty string")
                            final_answer = arg["answer"]
                            stop_reason = "submitted"
                            result = "Answer submitted."
                        else:
                            result = "Unknown tool: " + name
                            tool_error = True
                    raw_result_ref = self.diagnostics.put(str(result))
                    result = self.cm.on_observation(name, arg, str(result))
                except Exception as exc:
                    result = f"Tool error: {self._error_text(exc)}"
                    tool_error = True
                    detail.update(error_type=type(exc).__name__)
                    if name == "open_document":
                        detail.update(document_status="error", original_chars=None,
                                      returned_start=None, returned_end=None, truncated=None)
                    elif name == "search":
                        detail["returned_docids"] = None

                msg = {"role": "tool", "tool_call_id": call.id, "content": result,
                       "name": name}
                if name == "open_document" and isinstance(arg, dict) and "doc_id" in arg:
                    msg["doc_id"] = str(arg["doc_id"])
                history.append(msg)
                self.events.append({"step": step, "tool": name, "args": arg,
                                    "result_chars": len(str(result)),
                                    "tool_error": tool_error, "status": "error" if tool_error else "success",
                                    "raw_result_ref": raw_result_ref,
                                    "model_result_ref": self.diagnostics.put(str(result)), **detail})
                if final_answer is not None:
                    for skipped in assistant.tool_calls[assistant.tool_calls.index(call) + 1:]:
                        self.events.append({"step": step, "tool": skipped.function.name,
                                            "status": "skipped", "reason": "answer_submitted"})
                    break
            if final_answer is not None:
                break
        total_seconds = self.clock.monotonic() - started
        self._update_usage()
        return {"schema_version": 2, "episode_id": self.episode_id,
                "id": task["id"], "question": task["question"],
                "requested_model": self.model, "config_id": self.config_id,
                "model_source": getattr(self.args, "model_source", None),
                "research_config": getattr(self.args, "research_config", None),
                "code_revision": getattr(self.args, "code_revision", None), "pricing": self.pricing,
                "reference_answer": task.get("reference_answer"), "condition": self.condition,
                "answer": final_answer, "stop_reason": stop_reason, "steps": steps,
                "notes": notes, "repeated_searches": repeated_searches,
                "unique_searches": len(search_cache),
                "usage": dict(self.usage), "manager_stats": self.cm.stats(),
                "manager_runtime": {name: getattr(self.cm, name, None) for name in ("name", "device", "model_name")},
                "total_seconds": round(total_seconds, 3), "events": self.events,
                "initialization_seconds": initialization_seconds,
                "diagnostics_enabled": self.diagnostics.directory is not None,
                "history_capture_enabled": bool(self.args.save_history),
                "timeout_policy": "existing between-iteration elapsed check",
                "timeout_overrun_seconds": max(0, total_seconds - self.args.timeout),
                "history_ref": self.diagnostics.put(history) if self.args.save_history else None}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["baseline", "memgpt", "llmlingua", "srcm"])
    p.add_argument("--questions", help="Path to JSONL or JSON question list; otherwise interactive")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Shuffle question order deterministically")
    p.add_argument("--exclude-ids", default=None, help="Text file of pilot IDs to exclude")
    p.add_argument("--api-retries", type=int, default=3)
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--save-history", action="store_true", help="Store full research history; large output")
    p.add_argument("--resume", action="store_true", help="Append and skip IDs already in --out")
    p.add_argument("--out", default=None, help="Output JSONL path")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--timeout", type=int, default=900, help="Per-question timeout in seconds")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--snippet-chars", type=int, default=400)
    p.add_argument("--document-chars", type=int, default=12000)
    p.add_argument("--cm-args", default="{}", help='JSON manager settings, e.g. {"rate":0.5}')
    p.add_argument("--condition-cm-args", default="{}", help='Per-condition JSON settings, e.g. {"B":{},"C":{},"D":{}}')
    p.add_argument("--model", default=None, help="Research model; overrides OPENROUTER_MODEL after .env loading")
    p.add_argument("--index-dir", default="indexes/bm25")
    p.add_argument("--diagnostics", action="store_true", help="Save deduplicated model inputs and tool results locally")
    p.add_argument("--input-per-million", type=float, default=None)
    p.add_argument("--cached-per-million", type=float, default=None)
    p.add_argument("--output-per-million", type=float, default=None)
    args = p.parse_args()
    bootstrap_model(args)
    if not os.getenv("OPENROUTER_API_KEY"):
        p.error("Set OPENROUTER_API_KEY in .env")
    condition = {"baseline": "A", "memgpt": "B", "llmlingua": "C", "srcm": "D"}[args.mode]
    excluded = set()
    if args.exclude_ids:
        excluded = {line.strip() for line in Path(args.exclude_ids).read_text(encoding="utf-8").splitlines() if line.strip()}
    questions = load_questions(args.questions, args.limit, args.seed, excluded)
    out = Path(args.out or f"runs/{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl")
    from evaluation.experiment import read_records, validate_resume
    root = Path(__file__).parent
    args.research_config = research_configuration(args, root, SYSTEM, BASE_TOOLS, [condition])
    args.code_revision = code_revision(root)
    args.config_id = fingerprint({"common": args.research_config["common"],
                                  "method": args.research_config["methods"][condition]})
    manifest = {"schema_version": 2, "dataset_sha256": fingerprint(questions), "seed": args.seed,
                "selected_ids": [q["id"] for q in questions], "excluded_ids": sorted(excluded),
                "research_config": args.research_config, "pricing": pricing_from_args(args)}
    manifest_path = out.with_suffix(".manifest.json")
    completed = read_records(out)
    if out.exists() and not args.resume:
        p.error("Output exists; select a new path or explicitly resume")
    if out.exists() or manifest_path.exists():
        if not manifest_path.exists():
            p.error("Legacy output has no configuration manifest; select a new output path")
        validate_resume(json.loads(manifest_path.read_text()), manifest, {condition: completed})
    searcher = load_searcher(args.index_dir)
    client = create_client()
    diagnostics = DiagnosticStore(out.parent / ".diagnostics" if args.diagnostics or args.save_history else None)
    agent = ResearchAgent(searcher, client, condition, args, diagnostics=diagnostics)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with out.open("a" if args.resume else "x", encoding="utf-8") as f:
        for i, question in enumerate(questions, 1):
            if str(question["id"]) in completed:
                print("Skipping completed:", question["id"])
                continue
            print(f"\n[{i}/{len(questions)}] Condition {condition}: {question['id']}", flush=True)
            result = agent.run(question)
            print("Answer:", result["answer"])
            print("Stop:", result["stop_reason"], "Usage:", result["usage"])
            print("Context manager stats:", result["manager_stats"])
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    print("\nSaved:", out)
    print("Note: Reference answers are stored, but benchmark grading is not implemented.")

if __name__ == "__main__":
    main()
