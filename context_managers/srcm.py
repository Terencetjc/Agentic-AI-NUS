from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .base import ContextManager, count_tokens, history_tokens

_STOP = set(
    "a an the of and or in on at to for from by with as is are was were be been being this that these those it its "
    "which who whom whose what when where why how did does do has have had not no but if then than so such into about "
    "also can could would should may might will shall their his her they them he she you your i we our us there here "
    "any all each other more most some one two".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_DOCID = re.compile(r"(?:doc(?:ument)?[_ ]?id|docid|doc|document)\W{0,3}(\d+)", re.I)
_TAG = re.compile(
    r"^\s*(CLAIM|FINDING|EVIDENCE|RESULT|DEFINITION|ASSUMPTION|LIMITATION|CONTRADICTION|CONFLICT|"
    r"OPEN|TODO|QUESTION|DONE|ANSWERED|RESOLVED)\s*[:\-]\s*(.*)$",
    re.I | re.S,
)
_TAG_KIND = {
    "CLAIM": "claim", "FINDING": "claim", "EVIDENCE": "claim", "RESULT": "claim",
    "DEFINITION": "definition", "ASSUMPTION": "assumption", "LIMITATION": "limitation",
    "CONTRADICTION": "contradiction", "CONFLICT": "contradiction",
    "OPEN": "open", "TODO": "open", "QUESTION": "open",
    "DONE": "done", "ANSWERED": "done", "RESOLVED": "done",
}
_COMPACTABLE = {"search", "search_corpus", "open_document"}


_BLOCK = re.compile(r"(?:^|\n)(?:Document\s+(\d+):[ \t]*\n?|\[docid:\s*(\d+)\][ \t]*)")


def _split_docs(text: str) -> list:
    """Split a search result into (doc_id, body) blocks. Handles 'Document 123:\\n...' and '[docid: 123] ...'."""
    ms = list(_BLOCK.finditer(text))
    out = []
    for j, m in enumerate(ms):
        end = ms[j + 1].start() if j + 1 < len(ms) else len(text)
        out.append((m.group(1) or m.group(2), text[m.end():end].strip()))
    return out


def _kw(text: str) -> frozenset:
    """Content words (no stopwords, no pure numbers)."""
    return frozenset(w for w in _WORD.findall(text.lower())
                     if w not in _STOP and len(w) > 2 and not w.isdigit())


def _nums(text: str) -> frozenset:
    return frozenset(n.replace(",", "").rstrip(".") for n in _NUM.findall(text))


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class _Fact:
    text: str
    sources: list = field(default_factory=list)
    kw: frozenset = frozenset()
    nums: frozenset = frozenset()


class SRCM(ContextManager):
    name = "D_srcm"

    def __init__(self, K: int = 3, threshold: int = 20000, max_facts_per_doc: int = 4,
                 max_state_tokens: int = 4000, fact_chars: int = 320,
                 dup_similarity: float = 0.9, conflict_similarity: float = 0.6,
                 inject_state: bool = True, source_pointers: bool = True,
                 merge_duplicates: bool = True, keep_conflicts: bool = True,
                 collapse_done: bool = True):
        self.K = K
        self.threshold = threshold
        self.max_facts_per_doc = max_facts_per_doc
        self.max_state_tokens = max_state_tokens
        self.fact_chars = fact_chars
        self.dup_similarity = dup_similarity
        self.conflict_similarity = conflict_similarity
        self.inject_state = inject_state
        self.source_pointers = source_pointers
        self.merge_duplicates = merge_duplicates
        self.keep_conflicts = keep_conflicts
        self.collapse_done = collapse_done
        self.reset()

    # ------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._seen = 0                      # history index already ingested
        self._calls: dict = {}              # tool_call_id -> (tool name, args)
        self.question = ""
        self._q_kw: frozenset = frozenset()
        # Research State Store
        self.claims: list[_Fact] = []
        self.evidence: list[_Fact] = []     # verbatim sentences from documents
        self.definitions: list[_Fact] = []
        self.assumptions: list[_Fact] = []
        self.limitations: list[_Fact] = []
        self.contradictions: list[str] = [] # explicit (tagged) + auto-detected, rendered side by side
        self.calcs: list[tuple] = []        # (expression, result) verbatim
        self.open_q: list[_Fact] = []
        self.done: list[_Fact] = []
        self.sources: dict[str, str] = {}   # doc_id -> title ("" if only seen in search)
        self.searches: list[tuple] = []     # (query, [doc ids])
        # compaction bookkeeping (sticky)
        self._pointer: dict[str, str] = {}  # tool_call_id -> stub text (computed at ingest)
        self.compacted: dict[str, str] = {}
        self.snapshot = ""
        self._conflict_keys: set = set()
        self._stats = {"compactions": 0, "observations_compacted": 0, "tokens_removed": 0,
                       "duplicates_merged": 0, "conflicts_flagged": 0, "facts_kept": 0,
                       "state_tokens": 0}

    # ------------------------------------------------------------------------------------------
    # Ingestion: read new messages into the Research State Store
    # ------------------------------------------------------------------------------------------
    def _ingest(self, history: list[dict]) -> None:
        for i in range(self._seen, len(history)):
            m = history[i]
            role = m.get("role")
            if role == "user" and not self.question:
                self.question = m.get("content") or ""
                self._q_kw = _kw(self.question)
            elif role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    self._calls[tc.get("id")] = (fn.get("name", ""), args)
            elif role == "tool":
                name, args = self._calls.get(m.get("tool_call_id"), (m.get("name", ""), {}))
                name = m.get("name") or name
                key = m.get("tool_call_id") or f"idx{i}"
                self._observe(key, name, args, m.get("content") or "", m.get("doc_id"))
        self._seen = len(history)

    def _observe(self, key: str, name: str, args: dict, text: str, doc_id: Optional[str]) -> None:
        if name == "open_document":
            did = str(doc_id or args.get("doc_id") or args.get("docid") or "?")
            title = _clip(next((ln for ln in text.splitlines() if ln.strip()), ""), 100)
            if self.sources.get(did, "") == "":
                self.sources[did] = title
            kept = self._extract_facts(text, did)
            self._pointer[key] = (f"[doc {did} | {title} | opened; {kept} key sentence(s) kept verbatim in "
                                  f"Research State. Re-open the doc if you need more.]")
        elif name in ("search", "search_corpus"):
            query = str(args.get("query", ""))
            blocks = _split_docs(text)
            ids = list(dict.fromkeys(d for d, _ in blocks)) or list(dict.fromkeys(_DOCID.findall(text)))
            kept = 0
            for did, body in blocks:  # documents arrive inside search results: extract facts from each
                title = _clip(next((ln for ln in body.splitlines() if ln.strip()), ""), 100)
                if self.sources.get(did, "") == "":
                    self.sources[did] = title
                kept += self._extract_facts(body, did)
            for d in ids:
                self.sources.setdefault(d, "")
            self.searches.append((query, ids))
            self._pointer[key] = (f"[search '{_clip(query, 80)}' | {len(ids)} doc(s), {kept} key sentence(s) kept "
                                  f"verbatim in Research State | doc ids: {', '.join(ids) if ids else 'none'}. "
                                  f"Re-run the search if you need more.]")
        elif name == "take_note":
            src = f"doc {args['doc_id']}" if args.get("doc_id") else "note"
            self._ingest_note(self._note_text(args) or text, src)
        elif name in ("calculator", "calculate"):
            expr = str(args.get("expression") or next(iter(args.values()), ""))
            self.calcs.append((expr, _clip(text, 200)))  # R3: verbatim, never compressed

    @staticmethod
    def _note_text(args: dict) -> str:
        for k in ("note", "text", "content", "notes"):
            if isinstance(args.get(k), str):
                return args[k]
        return " ".join(str(v) for v in args.values() if isinstance(v, str))

    # -- notes --------------------------------------------------------------------------------
    def _ingest_note(self, raw: str, source: str = "note") -> None:
        raw = raw.strip()
        if not raw:
            return
        mt = _TAG.match(raw)
        kind, body = (_TAG_KIND[mt.group(1).upper()], mt.group(2).strip()) if mt else ("claim", raw)
        if not body:
            return
        fact = _Fact(body, [source], _kw(body), _nums(body))
        if kind == "claim":
            self._add_fact(self.claims, fact)
        elif kind == "definition":
            self._add_fact(self.definitions, fact)
        elif kind == "assumption":
            self._add_fact(self.assumptions, fact)
        elif kind == "limitation":
            self._add_fact(self.limitations, fact)
        elif kind == "contradiction":
            if body not in self.contradictions:
                self.contradictions.append(body)
        elif kind == "open":
            self._add_fact(self.open_q, fact)
            self._q_kw |= fact.kw
        elif kind == "done":
            self.open_q = [q for q in self.open_q
                           if _jaccard(q.kw, fact.kw) < 0.4 and not (q.kw and q.kw <= fact.kw)]
            self._add_fact(self.done, fact)

    # -- documents ----------------------------------------------------------------------------
    def _extract_facts(self, text: str, doc_id: str) -> int:
        sents = [s.strip() for s in _SENT.split(text) if 25 <= len(s.strip())]
        scored = []
        for pos, s in enumerate(sents):
            k, n = _kw(s), _nums(s)
            score = 2 * len(k & self._q_kw) + (1 if n else 0)
            if score > 0:
                scored.append((-score, pos, s))
        scored.sort()
        chosen = sorted(scored[: self.max_facts_per_doc], key=lambda t: t[1])
        kept = 0
        for _, _, s in chosen:
            s = s if len(s) <= self.fact_chars else s[: self.fact_chars - 1] + "…"
            self._add_fact(self.evidence, _Fact(s, [f"doc {doc_id}"], _kw(s), _nums(s)))
            kept += 1
        self._stats["facts_kept"] += kept
        return kept

    # -- R1 merge duplicates, R4 flag conflicts -------------------------------------------------
    def _add_fact(self, store: list, fact: _Fact) -> None:
        if self.merge_duplicates:
            norm = _norm(fact.text)
            for old in store:
                if _norm(old.text) == norm or (
                        old.kw and _jaccard(old.kw, fact.kw) >= self.dup_similarity and old.nums == fact.nums):
                    for s in fact.sources:
                        if s not in old.sources:
                            old.sources.append(s)
                    self._stats["duplicates_merged"] += 1
                    return
        if self.keep_conflicts and fact.nums and store is self.evidence:
            for old in self.evidence + self.claims:
                if old.nums and old.nums != fact.nums and _jaccard(old.kw, fact.kw) >= self.conflict_similarity:
                    ck = (_norm(old.text), _norm(fact.text))
                    if ck not in self._conflict_keys:
                        self._conflict_keys.add(ck)
                        self.contradictions.append(
                            f"(A) {old.text} [{', '.join(old.sources)}]  <->  (B) {fact.text} [{', '.join(fact.sources)}]")
                        self._stats["conflicts_flagged"] += 1
        store.append(fact)

    # ------------------------------------------------------------------------------------------
    # Rendering the state
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _fmt(f: _Fact) -> str:
        return f"- {f.text} [{', '.join(f.sources)}]"

    def _render_state(self) -> str:
        ev = list(self.evidence)
        while True:
            text = self._compose(ev)
            if count_tokens(text) <= self.max_state_tokens:
                break
            # trim oldest evidence WITHOUT numbers first (R3: numbers are protected), then give up
            drop = next((j for j, f in enumerate(ev) if not f.nums), None)
            if drop is None:
                break
            ev.pop(drop)
        self._stats["state_tokens"] = count_tokens(text)
        return text

    def _compose(self, ev: list[_Fact]) -> str:
        out = ["[RESEARCH STATE (SRCM). Facts below were extracted verbatim from earlier tool results, which "
               "have been collapsed to source pointers. Bracketed items are sources.]"]

        def section(title, lines):
            if lines:
                out.append(f"\n{title}:\n" + "\n".join(lines))

        section("Open questions", [self._fmt(f) for f in self.open_q])
        section("Claims / notes", [self._fmt(f) for f in self.claims])
        section("Evidence (verbatim)", [self._fmt(f) for f in ev if not f.nums])
        numeric = [self._fmt(f) for f in ev if f.nums]
        numeric += [f"- calc: {e} = {r}" for e, r in self.calcs]
        section("Numerical values & calculations (exact)", numeric)
        section("Conflicting findings (kept side by side)", [f"- {c}" for c in self.contradictions])
        section("Definitions", [self._fmt(f) for f in self.definitions])
        section("Assumptions", [self._fmt(f) for f in self.assumptions])
        section("Limitations", [self._fmt(f) for f in self.limitations])
        if self.done:
            if self.collapse_done:
                section("Completed subtasks (collapsed)", [f"- {_clip(f.text, 100)}" for f in self.done])
            else:
                section("Completed subtasks", [self._fmt(f) for f in self.done])
        if self.sources:
            srcs = [f"{d}: {t}" if t else d for d, t in self.sources.items()]
            out.append("\nSources seen (doc ids): " + "; ".join(srcs))
        if self.searches:
            out.append("Queries already run: " + "; ".join(f"'{_clip(q, 60)}'" for q, _ in self.searches))
        return "\n".join(out)

    # ------------------------------------------------------------------------------------------
    # build_messages: sticky replacements + state snapshot in the first user message
    # ------------------------------------------------------------------------------------------
    def _apply(self, history: list[dict]) -> list[dict]:
        out, first_user = [], True
        for i, m in enumerate(history):
            key = m.get("tool_call_id") or f"idx{i}"
            if m.get("role") == "tool" and key in self.compacted:
                m = dict(m, content=self.compacted[key])
            elif m.get("role") == "user" and first_user:
                first_user = False
                if self.snapshot:
                    m = dict(m, content=(m.get("content") or "") + "\n\n" + self.snapshot)
            out.append(m)
        return out

    def build_messages(self, history: list[dict]) -> list[dict]:
        self._ingest(history)
        msgs = self._apply(history)
        if history_tokens(msgs) <= self.threshold:
            return msgs  # unchanged since last compaction -> prompt cache still hits

        idx = [i for i, m in enumerate(history)
               if m.get("role") == "tool" and (m.get("name") or self._calls.get(m.get("tool_call_id"), ("",))[0])
               in _COMPACTABLE and (m.get("tool_call_id") or f"idx{i}") not in self.compacted]
        victims = idx[: max(0, len(idx) - self.K)]  # the K most recent observations stay verbatim
        if not victims:
            return msgs
        for i in victims:
            m = history[i]
            key = m.get("tool_call_id") or f"idx{i}"
            stub = self._pointer.get(key, "[tool result collapsed]") if self.source_pointers else "[collapsed]"
            self._stats["tokens_removed"] += count_tokens(m.get("content") or "") - count_tokens(stub)
            self.compacted[key] = stub
        self._stats["compactions"] += 1
        self._stats["observations_compacted"] += len(victims)
        if self.inject_state:
            self.snapshot = self._render_state()
        return self._apply(history)

    def stats(self) -> dict:
        return dict(self._stats, sources=len(self.sources), evidence=len(self.evidence),
                    claims=len(self.claims), open_questions=len(self.open_q))
