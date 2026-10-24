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
                                   runs/<condition>/<query_id>.json
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
topics-qrels/queries.tsv  decrypted questions       (not committed)
indexes/                  BM25 index, auto-downloaded (not committed)
cache/llmlingua/          compressed docs for C      (not committed)
runs/                     one JSON log per question per condition
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
OPENROUTER_API_KEY=sk-or-...
AGENT_MODEL=<model chosen in the pilot>
```

**4. Questions.** The BrowseComp-Plus questions are encrypted. Decrypt them with the authors' script:
```bash
git clone https://github.com/texttron/BrowseComp-Plus.git ../BrowseComp-Plus
cd ../BrowseComp-Plus
pip install datasets
python scripts_build_index/decrypt_dataset.py \
    --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
cd -
mkdir -p topics-qrels && cp ../BrowseComp-Plus/topics-qrels/queries.tsv topics-qrels/
```
> ⚠️ Never commit decrypted questions or answers. The encryption exists to keep them out of future models' training data.

**5. BM25 index.** Nothing to do: `agent.py` downloads it to `indexes/` on first run.

**6. Check the install**
```bash
python -c "from context_managers import make_manager; print(make_manager('A').name)"   # → A_full
```

---

## Running experiments

### Choose the question sets (once, then commit the ID files)
Pilot and main sets **must not overlap**. Settings are tuned on the pilot, and results are reported on the main set.
```bash
python -c "
import random
ids=[l.split('\t')[0] for l in open('topics-qrels/queries.tsv') if '\t' in l]
random.Random(4213).shuffle(ids)
open('pilot_ids.txt','w').write('\n'.join(ids[:10]))
open('main_ids.txt','w').write('\n'.join(ids[10:210]))
"
```

### Pilot (tune, then freeze)
```bash
python agent.py --condition A --ids pilot_ids.txt --out runs/pilot/A
python agent.py --condition B --ids pilot_ids.txt --out runs/pilot/B_16k --cm-args '{"max_tokens":16000,"target_tokens":8000}'
python agent.py --condition B --ids pilot_ids.txt --out runs/pilot/B_32k --cm-args '{"max_tokens":32000,"target_tokens":16000}'
python agent.py --condition C --ids pilot_ids.txt --out runs/pilot/C_r033 --cm-args '{"rate":0.33}'
python agent.py --condition C --ids pilot_ids.txt --out runs/pilot/C_r05  --cm-args '{"rate":0.5}'
python agent.py --condition D --ids pilot_ids.txt --out runs/pilot/D
```
Record the chosen settings in the **Frozen settings** table below before starting the main run.

### Main run
```bash
python agent.py --condition A --ids main_ids.txt --out runs/main/A
# ...same for B, C, D using the frozen --cm-args
```

**Run in parallel:** open several terminals and give each a shard:
```bash
python agent.py --condition A --ids main_ids.txt --out runs/main/A --shard 0/4
python agent.py --condition A --ids main_ids.txt --out runs/main/A --shard 1/4   # ...2/4, 3/4
```

**Resume after a crash:** rerun the same command. Questions that already have a log are skipped.

### Fixed run settings (top of `agent.py`, the same for every condition)
| Setting | Value |
|---|---|
| Max agent steps | 40 |
| Timeout per question | 15 min |
| Search results per query | 5 (300-char snippets) |
| Opened document length | 6,000 chars |
| API retries | 6, exponential backoff (not counted as steps) |

### Frozen settings (fill in after the pilot)
| | Value |
|---|---|
| Model (`AGENT_MODEL`) | |
| B `--cm-args` | |
| C `--cm-args` | |
| D `--cm-args` | |
| Pilot / main question counts | 10 / |

---

## Output: one JSON file per question

`runs/<set>/<condition>/<query_id>.json`

| Field | Meaning |
|---|---|
| `query_id`, `status`, `tool_call_counts`, `retrieved_docids`, `result` | Fields read by the BrowseComp-Plus grader. `status` is `completed`, `step_limit`, `timeout` or `context_exceeded` |
| `usage` | Totals: `input`, `cached`, `output` tokens, `cost` ($), `agent_calls`, `memory_calls` |
| `calls` | Per API call: tokens, cost, context size, latency, purpose (`agent` or `memory_write`) |
| `manager_stats` | Condition-specific counters (B evictions/recalls, C compression ratio and time) |
| `notes`, `history` | The agent's scratchpad and full trajectory, for failure analysis |

Runs that end without an answer are scored as incorrect, and their cost still counts.

---

## Team workflow on GitHub

### Who owns what
| Area | Files | Owner |
|---|---|---|
| Agent loop, tools, logging | `agent.py` | Agent |
| Conditions A, B, C | `context_managers/` except `srcm.py` | Conditions (Jie Yi) |
| SRCM + ablations | `context_managers/srcm.py` | SRCM |
| Data, pilot, grading, analysis | `scripts/`, `analysis/`, ID files | Pipeline & evaluation |

Only edit files you own. If you need a change in someone else's file, ask them or open an issue. Because each person works in different files, merges rarely conflict.

### Daily loop
```bash
git checkout main && git pull                 # start from the latest main
git checkout -b <your-branch>                 # e.g. srcm, grading, cm-pilot-tuning
# ...work...
git add <your files>                          # add specific files; avoid `git add .`
git status                                    # check: no .env, data/, cache/, indexes/
git commit -m "Short description of the change"
git push -u origin <your-branch>              # first push; afterwards just `git push`
```
Then open a **Pull Request** on GitHub and ask the owner of any affected area to review before merging.

**If `main` changed while you were working:**
```bash
git checkout main && git pull
git checkout <your-branch> && git merge main
```

### Never commit
```
.env  data/  topics-qrels/  indexes/  cache/  __pycache__/
```
These are already in `.gitignore`. Runs: commit only the final `runs/main/` logs if the team agrees (they are needed to reproduce the figures). Pilot runs can stay local.

---

## Adding SRCM (condition D)

1. Create `context_managers/srcm.py` with `class SRCM(ContextManager)`.
2. Override `build_messages(history)` for compaction. Read `name` and `doc_id` from tool messages, and the agent's notes from `take_note` calls.
3. Follow the shared rules: never delete messages (only replace `content`), and keep replacements sticky so prompt caching survives.
4. Pass settings through `--cm-args`, e.g. `'{"K":3,"threshold":20000}'`. Ablation switches go in the same way.

`agent.py --condition D` and `make_manager("D")` already point to it.

---

## References
- Yao et al. (2023). ReAct. arXiv:2210.03629
- Packer et al. (2023). MemGPT. arXiv:2310.08560
- Jiang et al. (2023). LLMLingua. arXiv:2310.05736
- Chen et al. (2025). BrowseComp-Plus. arXiv:2508.06600
- Liu et al. (2023). Lost in the Middle. arXiv:2307.03172
