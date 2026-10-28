"""Condition D — SRCM: Structured Research Context Manager (our method). Version 3.

Deterministic middleware. No LLM calls, no learned or statistical compressor, no extra tools.
It uses only structure already in the research trace: tool type, document ID, search query,
recency, document reopening, and whether a document was cited in a take_note call.

PROPOSAL RULES (Condition D, Component 6). Applied only when the context passes the threshold:
  P1  The last K search/open_document results are kept verbatim.                        keep_recent
  P2  Search hits already shown earlier are reduced to their document ID.               dedup_search
  P3  If a document is reopened, earlier copies are replaced by a pointer.              reopen_pointer
  P4  Older documents are reduced to their ID plus the few sentences that overlap most
      with the search queries that retrieved them.
  P5  Documents opened but never cited in a note are reduced to one line.               reduce_uncited
  P6  The scratchpad is kept as a structured list of (fact, source) pairs.              protect_scratchpad
  P7  Compaction only runs past a size threshold; between compactions the messages sent
      to the model do not change, so prompt caching keeps working.

FRAMEWORK FIGURE (Fig 2): the Research State Store holds the scratchpad (fact, source), calculation
results, conflicting findings, open questions / completed subtasks (optional note tags) and the
sources read. Reduction rules: (1) merge duplicates, (2) keep source pointers, (3) never alter
exact numbers, (4) conflicts side by side, (5) collapse finished subtasks.

WHAT IS NEW IN V3 (no rule changes; implementation only)
  * Cache-aware placement (Component 7 lists what the rebuilt context contains, not where).
    The original question stays untouched at the top, and the Research State sits right after the
    last compacted result, just before the K verbatim results. Everything above the oldest newly
    compacted result is byte-identical between compactions, so it stays cached. v1/v2 put the
    state in the first message, so every compaction re-billed the whole conversation.
  * Better sentence choice for P4/P5: query terms are weighted by how rare they are inside the
    document (a name or year beats "born" or "award"), quoted query phrases score extra, and
    near-duplicate sentences are skipped.
  * Shorter placeholders: the "how to reopen" instruction appears once in the state header,
    not in every placeholder.
  * Hysteresis: after a compaction the next one waits for `min_growth` new tokens, so a large
    verbatim window cannot make it fire on every step (each firing would re-bill the cache).

Shared rules (base.py): messages are never deleted or inserted, only `content` is replaced,
and replacements are sticky.

OPT-IN, NOT IN THE PROPOSAL (off by default; needs team approval before use in reported runs)
  dedup_on_arrival  apply P2 when a search result arrives instead of at the next compaction.

Removed from the team's v1: ingest_filter (it shrank every document on arrival, so the last K were
never verbatim, no threshold applied, and a reopened document could never return its full text).

Settings via --cm-args, e.g. '{"K":3,"threshold":8000}'. Proposal ablations:
  no recency preservation     '{"keep_recent":false}'
  no search-result dedup      '{"dedup_search":false}'
  no scratchpad protection    '{"protect_scratchpad":false}'
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .base import ContextManager, count_tokens, history_tokens

SEARCH_TOOLS = {"search", "search_corpus"}
DOC_TOOLS = {"open_document"}
NOTE_TOOLS = {"take_note"}
CALC_TOOLS = {"calculate", "calculator"}

_STOP = set(
    "a an the of and or in on at to for from by with as is are was were be been being this that these those it its "
    "which who whom whose what when where why how did does do has have had not no but if then than so such into about "
    "also can could would should may might will shall their his her they them he she you your i we our us there here "
    "any all each other more most some one two after before during between over under s".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_HIT = re.compile(r"(?:^|\n)(?:Document\s+([^\s:]+):|\[docid:\s*([^\]\s]+)\])[ \t]*")
_QUOTED = re.compile(r'"([^"]{3,80})"')
_IDLIKE = re.compile(r"\b\d{3,}\b")
_TAG = re.compile(
    r"^\s*(CLAIM|FINDING|EVIDENCE|RESULT|DEFINITION|ASSUMPTION|LIMITATION|CONTRADICTION|CONFLICT|"
    r"OPEN|TODO|QUESTION|DONE|ANSWERED|RESOLVED)\s*[:\-]\s*(.*)$", re.I | re.S)
_TAG_KIND = {"CLAIM": "fact", "FINDING": "fact", "EVIDENCE": "fact", "RESULT": "fact",
             "DEFINITION": "definition", "ASSUMPTION": "assumption", "LIMITATION": "limitation",
             "CONTRADICTION": "contradiction", "CONFLICT": "contradiction",
             "OPEN": "open", "TODO": "open", "QUESTION": "open",
             "DONE": "done", "ANSWERED": "done", "RESOLVED": "done"}


# ----------------------------------------------------------------------------------------------
# Small text helpers
# ----------------------------------------------------------------------------------------------
def _terms(text: str) -> frozenset:
    """Content terms, numbers included (years and IDs matter in queries)."""
    return frozenset(w for w in _WORD.findall((text or "").lower())
                     if w not in _STOP and (len(w) > 2 or w.isdigit()))


def _kw(text: str) -> frozenset:
    return frozenset(w for w in _terms(text) if not w.isdigit())


def _nums(text: str) -> frozenset:
    return frozenset(n.replace(",", "").rstrip(".") for n in _NUM.findall(text or ""))


def _norm(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def _jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _clip(text: str, n: int) -> str:
    """Shorten at a word boundary, so no word or number is ever cut in half (Fig rule 3)."""
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    cut = text.rfind(" ", 0, n)
    return text[: cut if cut > 0 else n] + " …"


def _sentences(text: str) -> list:
    out = []
    for part in _SENT.split(text or ""):
        s = part.strip(" \t#*-|>")
        letters = sum(c.isalpha() for c in s)
        if len(s) >= 25 and letters >= 0.5 * len(s):        # skip separators, tables, link lists
            out.append(_clip(s, 300))
    return out


def _title(text: str) -> str:
    lines = [ln.strip(" #*") for ln in (text or "").splitlines()[:8]]
    lines = [ln for ln in lines if ln and ln != "---"]
    for ln in lines:
        if ln.lower().startswith("title:"):
            return _clip(ln[6:].strip(), 60)
    return _clip(lines[0], 60) if lines else ""


def _split_hits(text: str) -> list:
    """'Document 123: snippet' blocks -> [(doc_id, block_text)]."""
    ms = list(_HIT.finditer(text or ""))
    out = []
    for j, m in enumerate(ms):
        start = m.start() + (1 if text[m.start()] == "\n" else 0)
        end = ms[j + 1].start() if j + 1 < len(ms) else len(text)
        out.append((m.group(1) or m.group(2), text[start:end].strip()))
    return out


@dataclass
class _Note:
    fact: str
    sources: list = field(default_factory=list)
    kw: frozenset = frozenset()
    nums: frozenset = frozenset()


# ----------------------------------------------------------------------------------------------
class SRCM(ContextManager):
    name = "D_srcm"

    def __init__(self, K: int = 3, threshold: int = 8000, min_growth: int = 4000,
                 keep_sentences: int = 3, max_state_tokens: int = 1200,
                 keep_recent: bool = True, dedup_search: bool = True, reopen_pointer: bool = True,
                 reduce_uncited: bool = True, protect_scratchpad: bool = True,
                 research_state: bool = True, dedup_on_arrival: bool = False):
        if K < 0 or threshold < 1 or keep_sentences < 1:
            raise ValueError("K must be >= 0, threshold >= 1, keep_sentences >= 1")
        self.K = K
        self.threshold = threshold
        self.min_growth = min_growth
        self.keep_sentences = keep_sentences
        self.max_state_tokens = max_state_tokens
        self.keep_recent = keep_recent
        self.dedup_search = dedup_search
        self.reopen_pointer = reopen_pointer
        self.reduce_uncited = reduce_uncited
        self.protect_scratchpad = protect_scratchpad
        self.research_state = research_state
        # OPT-IN, NOT IN THE PROPOSAL: apply P2 when a search result arrives instead of at the next
        # compaction. Never edits earlier messages, so it costs no cache. Off by default.
        self.dedup_on_arrival = dedup_on_arrival
        self.reset()

    def reset(self) -> None:
        self._seen = 0
        self._calls: dict = {}
        self.question = ""
        self._obs: list = []                 # one record per tool result, in order
        self.notes: list = []                # structured scratchpad: (fact, source) pairs
        self.calcs: list = []
        self.contradictions: list = []
        self.open_q: list = []
        self.done: list = []
        self.other: dict = {"definition": [], "assumption": [], "limitation": []}
        self.cited: set = set()              # doc ids used in a note
        self.first_seen: dict = {}           # doc id -> order of first search hit / open
        self.doc_queries: dict = {}          # doc id -> queries whose results contained it
        self.opened: dict = {}               # doc id -> orders of successful opens
        self.titles: dict = {}
        self.queries: list = []
        self.compacted: dict = {}            # tool_call_id -> replacement (sticky)
        self._settled: set = set()           # results that never need changing
        self.snapshot = ""
        self.state_host = None               # tool_call_id whose content carries the state
        self._floor = 0                      # hysteresis: no compaction below this size
        self._arrived: set = set()           # doc ids already shown (dedup_on_arrival only)
        self._stats = {"compactions": 0, "observations_compacted": 0, "search_hits_deduped": 0,
                       "docs_reduced": 0, "docs_one_line": 0, "reopen_pointers": 0,
                       "notes_compacted": 0, "tokens_removed": 0, "state_tokens": 0}

    # ------------------------------------------------------------------------------------------
    # Reading the trace
    # ------------------------------------------------------------------------------------------
    def _ingest(self, history: list) -> None:
        for i in range(self._seen, len(history)):
            m = history[i]
            role = m.get("role")
            if role == "user" and not self.question:
                self.question = m.get("content") or ""
            elif role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except (ValueError, TypeError):
                        args = {}
                    self._calls[tc.get("id")] = (fn.get("name", ""), args if isinstance(args, dict) else {})
            elif role == "tool":
                name, args = self._calls.get(m.get("tool_call_id"), (m.get("name", ""), {}))
                self._record(m.get("tool_call_id") or f"idx{i}", m.get("name") or name, args,
                             m.get("content") or "", m.get("doc_id"))
        self._seen = len(history)

    def _record(self, key, name, args, text, doc_id) -> None:
        order = len(self._obs)
        failed = text.startswith(("Tool error", "Document not found", "Unknown tool"))
        rec = {"key": key, "name": name, "order": order, "text": text, "ok": not failed}
        if name in SEARCH_TOOLS and not failed:
            q = str(args.get("query", ""))
            if q and q not in self.queries:
                self.queries.append(q)
            rec["query"] = q
            rec["hits"] = _split_hits(text)          # empty for the agent's "repeated search" notice
            for d, _ in rec["hits"]:
                self.first_seen.setdefault(d, order)
                qs = self.doc_queries.setdefault(d, [])
                if q and q not in qs:
                    qs.append(q)
        elif name in DOC_TOOLS and not failed:
            d = str(doc_id or args.get("doc_id") or "")
            rec["doc_id"] = d
            self.opened.setdefault(d, []).append(order)
            self.first_seen.setdefault(d, order)
            self.titles.setdefault(d, _title(text))
        elif name in NOTE_TOOLS and not failed:
            fact = str(args.get("text") or args.get("note") or "").strip()
            src = str(args.get("doc_id") or "").strip()
            if src:
                self.cited.add(src)
            for tok in _IDLIKE.findall(fact):        # doc ids mentioned inside the note text
                if tok in self.opened:
                    self.cited.add(tok)
            self._add_note(fact, f"doc {src}" if src else "no source")
        elif name in CALC_TOOLS and not failed:
            calc = (str(args.get("expression", "")), _clip(text, 120))
            if calc not in self.calcs:               # Fig rule 1
                self.calcs.append(calc)
        self._obs.append(rec)

    def _add_note(self, raw: str, source: str) -> None:
        mt = _TAG.match(raw)
        kind, body = (_TAG_KIND[mt.group(1).upper()], mt.group(2).strip()) if mt else ("fact", raw)
        if not body:
            return
        note = _Note(body, [source], _kw(body), _nums(body))
        if kind == "contradiction":
            self.contradictions.append(f"{body} [{source}]")
        elif kind == "open":
            self.open_q.append(note)
        elif kind == "done":                         # Fig rule 5
            self.open_q = [q for q in self.open_q if _jaccard(q.kw, note.kw) < 0.4]
            self.done.append(note)
        elif kind in self.other:
            self.other[kind].append(note)
        else:
            for old in self.notes:                   # Fig rule 1: merge duplicate findings
                if _norm(old.fact) == _norm(body) or (_jaccard(old.kw, note.kw) >= 0.9 and old.nums == note.nums):
                    if source not in old.sources:
                        old.sources.append(source)
                    return
            for old in self.notes:                   # Fig rule 4: conflicting findings side by side
                if (old.nums and note.nums and old.nums != note.nums
                        and _jaccard(old.kw, note.kw) >= 0.5 and source not in old.sources):
                    self.contradictions.append(
                        f"{old.fact} [{', '.join(old.sources)}] <-> {body} [{source}]")
            self.notes.append(note)

    def on_observation(self, tool_name: str, args: dict, text: str) -> str:
        """Default: unchanged (proposal). With dedup_on_arrival, repeated search hits arrive as IDs."""
        if not (self.dedup_on_arrival and self.dedup_search):
            return text
        if tool_name in DOC_TOOLS:
            self._arrived.add(str(args.get("doc_id", "")))
            return text
        if tool_name not in SEARCH_TOOLS:
            return text
        hits = _split_hits(text)
        if not hits:
            return text
        self._arrived |= set(self.first_seen)
        parts, repeated = [], 0
        for d, block in hits:
            if d in self._arrived:
                parts.append(f"Document {d}: (shown earlier)")
                repeated += 1
            else:
                parts.append(block)
                self._arrived.add(d)
        if not repeated:
            return text
        self._stats["search_hits_deduped"] += repeated
        return "\n\n".join(parts)

    # ------------------------------------------------------------------------------------------
    # P4 / P5 sentence selection
    # ------------------------------------------------------------------------------------------
    def _select(self, doc_id: str, text: str, n: int) -> list:
        """Sentences that best match the queries that retrieved this document (P4).
        Query terms are weighted by rarity inside the document; quoted phrases score extra."""
        sents = _sentences(text)
        if not sents:
            return []
        queries = self.doc_queries.get(doc_id) or self.queries          # fallback: all queries so far
        qterms = set().union(*(_terms(q) for q in queries)) if queries else set()
        if not qterms:
            qterms = set(_terms(self.question))
        phrases = [p.lower() for q in queries for p in _QUOTED.findall(q)]
        tie = set(_terms(self.question)) - qterms     # tie-breaker only; queries decide (P4)
        toks = [_terms(s) for s in sents]
        df = Counter(t for ts in toks for t in ts if t in qterms)
        n_s = len(sents)
        weight = {t: math.log((n_s + 1) / (c + 0.5)) for t, c in df.items()}
        scored = []
        for i, (s, ts) in enumerate(zip(sents, toks)):
            hit = ts & qterms
            if not hit:
                continue
            low = s.lower()
            score = (sum(weight[t] for t in hit) + 1.5 * sum(p in low for p in phrases)
                     + 0.01 * len(ts & tie))
            scored.append((score, -i, i, s, ts))
        scored.sort(reverse=True)
        if not scored:
            return []
        # Keep only sentences that match distinctive query terms: a term found in most sentences of
        # the document (weight < log 2) cannot make a sentence relevant on its own.
        bar = max(0.3 * scored[0][0], math.log(2))
        chosen = []
        for score, _, i, s, ts in scored:
            if score < bar:
                break
            if any(_jaccard(ts, c[2]) > 0.7 for c in chosen):   # skip near-duplicates
                continue
            chosen.append((i, s, ts))
            if len(chosen) >= n:
                break
        return [s for _, s, _ in sorted(chosen)]

    # ------------------------------------------------------------------------------------------
    # Reduction rules
    # ------------------------------------------------------------------------------------------
    def _reduce(self, r: dict, recent: set):
        """Return (replacement, stat). replacement None = leave for a later compaction, "" = keep verbatim."""
        name, key = r["name"], r["key"]
        if not r["ok"]:
            return "", None                              # errors stay verbatim
        if name in DOC_TOOLS and r.get("doc_id"):
            if key in recent:
                return None, None                        # P1
            d = r["doc_id"]
            title = self.titles.get(d, "")
            if self.reopen_pointer and r["order"] != self.opened[d][-1]:
                return f"[doc {d} · opened again later; see the later copy]", "reopen_pointers"   # P3
            if d in self.cited or not self.reduce_uncited:
                lines = [f"[doc {d} · {title} · reduced]"]
                picked = self._select(d, r["text"], self.keep_sentences) or _sentences(r["text"])[:1]
                lines += [f"- {s}" for s in picked]
                return "\n".join(lines), "docs_reduced"                                         # P4
            best = self._select(d, r["text"], 1)
            return (f"[doc {d} · {title} · not cited; one line] "
                    + (_clip(best[0], 200) if best else "")), "docs_one_line"                    # P5
        if name in SEARCH_TOOLS:
            if key in recent:
                return None, None                        # P1
            hits = r.get("hits") or []
            if not self.dedup_search or not hits:
                return "", None
            repeated = [d for d, _ in hits if self.first_seen.get(d, r["order"]) < r["order"]]
            if not repeated:
                return "", None                          # first appearances stay verbatim
            stat = ("search_hits_deduped", len(repeated))                                     # P2
            if len(repeated) == len(hits):
                return f"[search '{_clip(r.get('query', ''), 60)}' · all shown earlier: {', '.join(repeated)}]", stat
            return "\n\n".join(f"Document {d}: (shown earlier)" if d in repeated else block
                               for d, block in hits), stat
        if name in NOTE_TOOLS:
            # The fact is already in the agent's take_note call and (P6) in the scratchpad.
            return ("Saved to scratchpad." if self.protect_scratchpad else "[note saved earlier]"), "notes_compacted"
        return "", None                                  # calculations and anything else stay verbatim

    def _compact(self) -> int:
        compactable = [r for r in self._obs if r["ok"] and r["name"] in SEARCH_TOOLS | DOC_TOOLS]
        recent = ({r["key"] for r in compactable[-self.K:]}
                  if self.keep_recent and self.K > 0 else set())
        changed = 0
        for r in self._obs:
            key = r["key"]
            if key in self.compacted or key in self._settled:
                continue
            new, stat = self._reduce(r, recent)
            if new is None:
                continue
            if new and len(new) < len(r["text"]):
                self.compacted[key] = new
                changed += 1
                if stat:
                    name, n = stat if isinstance(stat, tuple) else (stat, 1)
                    self._stats[name] += n
            else:
                self._settled.add(key)
        return changed

    # ------------------------------------------------------------------------------------------
    # Research State (structured scratchpad)
    # ------------------------------------------------------------------------------------------
    def _render_state(self) -> str:
        for level in range(3):                           # notes are never trimmed
            text = self._compose(level)
            if count_tokens(text) <= self.max_state_tokens:
                break
        self._stats["state_tokens"] = count_tokens(text)
        return text

    def _compose(self, level: int) -> str:
        out = ["[SRCM RESEARCH STATE. Older results above are reduced; exact facts are kept below. "
               "open_document(id) returns any document's full text.]"]

        def sec(title, lines):
            if lines:
                out.append(f"{title}:\n" + "\n".join(lines))

        if self.protect_scratchpad:
            sec("Scratchpad (fact | source)", [f"- {n.fact} | {', '.join(n.sources)}" for n in self.notes])
            sec("Calculations", [f"- {e} = {r}" for e, r in self.calcs])
            sec("Conflicting findings", [f"- {c}" for c in self.contradictions])
            sec("Open questions", [f"- {n.fact}" for n in self.open_q])
            sec("Completed subtasks", [f"- {_clip(n.fact, 80)}" for n in self.done])
            for kind, items in self.other.items():
                sec(kind.capitalize() + "s", [f"- {n.fact} | {', '.join(n.sources)}" for n in items])
        if self.opened and level < 2:
            cited = [d for d in self.opened if d in self.cited]
            other = [d for d in self.opened if d not in self.cited]
            line = "Documents read: " + ", ".join(cited) + " (cited in notes)" if cited else "Documents read:"
            if other:
                line += ("; " if cited else " ") + ", ".join(other) + " (not cited)"
            out.append(line)
        return "\n".join(out)

    # ------------------------------------------------------------------------------------------
    # build_messages
    # ------------------------------------------------------------------------------------------
    def _apply(self, history: list) -> list:
        out = []
        for i, m in enumerate(history):
            if m.get("role") == "tool":
                key = m.get("tool_call_id") or f"idx{i}"
                if key in self.compacted:
                    content = self.compacted[key]
                    if key == self.state_host and self.snapshot:
                        content = content + "\n\n" + self.snapshot
                    m = dict(m, content=content)
            out.append(m)
        return out

    def build_messages(self, history: list) -> list:
        self._ingest(history)
        msgs = self._apply(history)
        size = history_tokens(msgs)
        if size <= self.threshold or size < self._floor:   # P7: unchanged between compactions
            return msgs
        if self._compact():
            if self.research_state:
                # Host the state on the newest compacted result: just above the K verbatim results.
                order = {r["key"]: r["order"] for r in self._obs}
                self.state_host = max(self.compacted, key=lambda k: order.get(k, -1))
                self.snapshot = self._render_state()
            msgs = self._apply(history)
            post = history_tokens(msgs)
            self._stats["compactions"] += 1
            self._stats["tokens_removed"] += size - post
            self._stats["observations_compacted"] = len(self.compacted)
        else:
            post = size
        self._floor = post + self.min_growth
        return msgs

    def stats(self) -> dict:
        return dict(self._stats, notes=len(self.notes), contradictions=len(self.contradictions),
                    docs_opened=len(self.opened), docs_cited=len(self.cited & set(self.opened)),
                    queries=len(self.queries), K=self.K, threshold=self.threshold,
                    min_growth=self.min_growth, keep_sentences=self.keep_sentences)
