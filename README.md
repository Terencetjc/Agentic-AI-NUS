# Context Management for Research Agents — DSA4213 Group 28

How do different ways of managing an agent's context affect **token use, API cost and accuracy** on multi-step research questions?

We run one research agent on [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) under four context-management conditions. Everything else is held fixed: model, prompt, tools, corpus, search index and limits.

| Condition | What the model sees | Code |
|---|---|---|
| **A** Full context (control) | Everything, verbatim | `context_managers/full_context.py` |
| **B** LLM-managed memory (MemGPT-style) | Old observations moved to a searchable archive past a size limit | `context_managers/memgpt.py` |
| **C** Statistical compression (LLMLingua) | Documents compressed before entering context | `context_managers/llmlingua_compression.py` |
| **D** SRCM, our method | Deterministic structural compaction of the research trace | `context_managers/srcm.py` |

**Team:** Terence, Xuanxi, Xylia, Keak Jie Yi

---

## How the pieces fit

```
                 ┌──────────────────────────── agent.py ───────────────────────────┐
question ──────► │  loop:  messages = cm.build_messages(history)   ◄── B, D compact │
                 │         reply    = LLM(messages, tools)          (logged: tokens,│
                 │         for each tool call:                       cost, latency) │
                 │             out = cm.handle_tool(...)   or run the tool  ◄── B   │
                 │             out = cm.on_observation(...)              ◄── C      │
                 │             history.append(out)                                  │
                 │  until submit_answer / step limit / timeout                      │
                 └──────────────────────────────┬───────────────────────────────────┘
                                                ▼
                                   <run-dir>/<condition>.jsonl
                                                ▼
                                   grading + analysis
```

`agent.py` owns the loop, the tools, search and logging. `context_managers/` owns only **what the model sees**. Each condition implements the same interface (`context_managers/base.py`), so the agent never needs to know which one it is running. Details are in [`context_managers/README.md`](context_managers/README.md).

## Repository layout

```
agent.py                  batch runner: agent loop, tools, BM25 search, logging
context_managers/         conditions A–D behind one interface (see its README)
requirements.txt
.env                      OPENROUTER_API_KEY=...   (not committed)
data/questions_830.jsonl   prepared questions        (not committed)
indexes/bm25/             prepared local BM25 index (not committed)
cache/llmlingua/          compressed docs for C      (not committed)
runs/                     run outputs; sharing follows the group policy
evaluation/               records, accounting, diagnostics and offline reports
tests/                    synthetic offline unit and regression tests
```

---

## Setup (once per machine)

**1. Python packages**
```bash
pip install -r requirements.txt
```

**2. Java 21.** Pyserini's BM25 search runs on Java.
```bash
brew install openjdk@21            # macOS
# sudo apt install -y openjdk-21-jdk   # Ubuntu
java -version                      # should say 21
```

**3. API key.** Create `.env` in the project root:
```
OPENROUTER_API_KEY=<set-locally>
OPENROUTER_MODEL=<research-model-id>
```

**4. Questions.** Run the preparation script manually to download and decrypt the
BrowseComp-Plus test questions:
```bash
python prepare_questions.py --out data/questions_830.jsonl
```
Keep prepared questions and answers outside version control.

**5. BM25 index.** Prepare the `bm25/` directory from
[`Tevatron/browsecomp-plus-indexes`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes)
under `indexes/bm25/`. Execution opens an existing local index; index downloads
are a separate preparation step. Condition C also requires its configured local
compressor resources. See [`context_managers/README.md`](context_managers/README.md)
for method details.

**6. Offline verification.** Run from the repository root:
```bash
python -B tests/run_offline.py
```
The runner guards network connections and runtime dependency imports before test
discovery. Tests use synthetic records, fake clients and temporary directories.

---

## Running experiments

### Fix the question selection and configuration

Development and main question sets must be disjoint. Keep the selected ID files
and agreed settings fixed for each run. Set `QUESTION_COUNT` to the number of IDs
in `selected_ids.txt` and `RESEARCH_MODEL` to the agreed provider model identifier.
Check configuration and local index presence first:

```bash
python benchmark.py --questions data/questions_830.jsonl --ids selected_ids.txt \
    --count "$QUESTION_COUNT" --model "$RESEARCH_MODEL" \
    --out-dir experiments/selected-set --preflight
```

`--preflight` writes no files and initializes no provider client, searcher or
context manager. `--dry-run` saves the manifest and selection without model
requests. Run research separately after reviewing the configuration and budget:

```bash
python benchmark.py --questions data/questions_830.jsonl --ids selected_ids.txt \
    --count "$QUESTION_COUNT" --model "$RESEARCH_MODEL" \
    --out-dir experiments/selected-set --conditions A B C D --diagnostics
```

These research commands incur API charges. Supply the agreed
`--input-per-million`, `--cached-per-million`, and `--output-per-million` rates for
cost estimates; unavailable costs remain unknown. `--max-budget-usd` checks
estimated spending between episodes and can be exceeded within an episode.
Grading expenditure is accounted separately.

Condition subsets can share a run directory when settings and selection match.
Repeat the same command to resume; saved episodes are retained, including failed
ones. Research-affecting changes and legacy manifests with missing configuration
metadata require a new directory. An existing manifest's question set is fixed.

### Model and method settings

Model precedence is `--model`, then `OPENROUTER_MODEL` after `.env` loading, then
`openai/gpt-4.1-mini`. An inherited environment variable takes precedence over
`.env`. `AGENT_MODEL` is replaced by `OPENROUTER_MODEL`.

`--cm-args` remains available for shared settings. `--condition-cm-args` supplies
per-condition overrides, for example `'{"B":{},"C":{},"D":{}}'`; empty objects
retain the method owners' defaults.

| Setting | Current default |
|---|---|
| Max agent steps | 40 |
| Episode timeout | 900 seconds, checked between iterations |
| Search results per query | `benchmark.py`: 3; `agent.py`: 5; 400-character prefixes |
| Opened document length | 12,000 characters |
| API retries | 3 outer retries, plus unchanged SDK retry behavior |
| Maximum output tokens per request | 2,048 |

Freeze the model, method settings, dataset/index, prompts, tools, and limits
before paired experiments. The manifest records these settings and relevant
implementation hashes; pricing and diagnostic-output options stay separate.

### Grade saved answers separately

After all requested condition files are complete, an offline summary reuses only
matching cached grades:

```bash
python evaluate.py --run-dir experiments/selected-set \
    --judge-model "$JUDGE_MODEL" --offline
```

Set `JUDGE_MODEL` to the group's agreed judge. Omit `--offline` only when grading
requests are intended; that invocation can incur API charges. The existing
semantic grading criterion is preserved. Ungraded completed answers remain
unknown, and unsuccessful episodes count as incorrect. Grades are invalidated
when answers, references, questions, or grading configuration change.

## Output and accounting

The benchmark writes `manifest.json`, `selected_ids.txt`, and one JSONL file per
condition in the caller-selected output directory. Each line is an episode.

| Field | Contents |
|---|---|
| `id`, `episode_id`, `condition`, `answer`, `stop_reason` | Task identity, submitted answer and termination |
| `requested_model`, `config_id`, `research_config`, `code_revision` | Model and configuration provenance |
| `usage` | Input/output/cache totals, estimated cost, observed call attempts and API time |
| `events` | Call outcomes, model IDs, usage, latency, search queries, document IDs and tool status |
| `manager_stats`, `manager_runtime` | Method counters and resolved compressor/device details |
| `total_seconds`, `initialization_seconds` | Episode duration and separate manager initialization |
| `notes`, diagnostic references | Research notes and optional content-addressed payloads |

Total tokens are input plus output; cached input is a subset. Missing
measurements propagate to dependent totals. Research and memory calls share the
same accounting path. SDK-internal retries are contained within an observed
wrapper attempt. Episode timing starts after manager initialization; in-flight
calls and synchronous compression can overrun the between-iteration timeout.

`--diagnostics` saves exact request bodies and pre/post-management tool results
under the run's `.diagnostics/` directory. `--save-history` adds final history.
Keep these payloads local; they can contain benchmark text and sensitive content.

The standard-library [`evaluation/` module](evaluation/README.md) documents
schemas, metric formulas, compatibility checks and offline aggregation across
compatible run directories. Its tests require neither credentials nor resources.

---

## Team workflow on GitHub

### Who owns what
| Area | Files | Owner |
|---|---|---|
| Agent loop, tools, logging | `agent.py` | Agent (Terence) |
| Conditions A, B, C | `context_managers/` except `srcm.py` | Conditions (Jie Yi) |
| SRCM + ablations | `context_managers/srcm.py` | SRCM (Xylia) |
| Data, pilot, grading, analysis | `scripts/`, `analysis/`, ID files | Pipeline & evaluation (Xuanxi) |

### Never commit
```
.env  data/  topics-qrels/  indexes/  cache/  __pycache__/
```
The group's existing experiment-sharing policy is preserved: share final run
artifacts only when the team agrees. Development directories under `runs/dev_*/`,
root-level `runs/dev_*.json` reports, and optional `.diagnostics/` payloads stay
local. Other experiment directories remain available for the group's workflow.
Existing data, index, cache and credential ignore rules continue to apply;
previously tracked artifacts remain unchanged.

---

## References
- Yao et al. (2023). ReAct. arXiv:2210.03629
- Packer et al. (2023). MemGPT. arXiv:2310.08560
- Jiang et al. (2023). LLMLingua. arXiv:2310.05736
- Chen et al. (2025). BrowseComp-Plus. arXiv:2508.06600
- Liu et al. (2023). Lost in the Middle. arXiv:2307.03172
