# Evaluation module

Python 3.10+, standard library only. This package validates records and computes
accounting summaries independently of provider and retrieval libraries. Examples
and tests use synthetic data and temporary files.

`adapters.py` connects the shared agent's JSONL records to the typed schemas.
Unavailable legacy model identities remain `None`; `model_identity_complete`
exposes coverage, while `model_ids` contains only observed identifiers.

## Files and public interface

| File | Responsibility |
| --- | --- |
| `schemas.py` | Validated input dataclasses and derived-summary dataclasses |
| `logging.py` | Typed, single-writer JSONL append/read functions |
| `metrics.py` | Formulas and missing-value handling |
| `aggregation.py` | Calls to episodes to conditions, plus baseline ratios |
| `__init__.py` | Public record, storage, and aggregation exports |
| `adapters.py` | Normalize shared-agent records for the existing aggregators |
| `experiment.py` | Model/configuration resolution, fingerprints, resume and pricing |
| `diagnostics.py` | Optional content-addressed local request/result storage |
| `pilot_report.py` | Offline summaries across compatible, disjoint run selections |

Public functions:

```python
append_call(path, record)       # None; one LLMCallRecord
append_episode(path, record)    # None; one EpisodeRecord
read_calls(path)                # list[LLMCallRecord]
read_episodes(path)             # list[EpisodeRecord]
aggregate_episodes(calls, episodes)  # list[EpisodeSummary], input episode order
aggregate_conditions(summaries)     # list[ConditionSummary], condition-name order
compare_conditions(method, baseline)  # ComparisonSummary; explicit baseline
```

Paths are caller-selected strings or `Path` objects. Create parent directories
before appending and use a single writer per file. Store calls and episodes in
separate files. After grading, update loaded episodes with `dataclasses.replace`
before aggregation. Each episode ID must occur once; duplicate records raise an
error rather than updating earlier entries.

## Input schema

Required fields have no default. All optional fields default to `None`.

### `LLMCallRecord`

One record represents one **observed API attempt**, whether successful or failed.

| Field | Type / meaning |
| --- | --- |
| `call_id`, `episode_id` | Required nonempty strings; identify the attempt and episode |
| `model_id` | Nonempty model identifier when known; `None` preserves unavailable legacy identity |
| `purpose` | Required `research` or `memory`; grading is outside this stream |
| `status` | Required `success` or `error` for this observed attempt |
| `config_id` | Optional nonempty identifier referencing settings maintained outside this package |
| `step_index` | Optional nonnegative integer agent-step index; use a consistent convention |
| `input_tokens` | Optional total input usage, **including cached input tokens** |
| `output_tokens` | Optional output usage |
| `cached_input_tokens` | Optional cached subset of input usage |
| `context_size_tokens` | Optional measured context size sent for this call |
| `api_cost_usd` | Optional externally reported or calculated cost in US dollars |
| `latency_seconds` | Optional elapsed duration of this attempt in seconds |
| `error_message` | Optional string describing failure |
| `metadata` | Optional JSON-compatible object |

Input/output/cache/context counts must be nonnegative integers. Cached input
cannot exceed input when both are known. Error attempts with known usage or cost
remain in totals. Report unknown billing as `None`. The caller supplies the final
episode outcome separately from individual call outcomes.

### `EpisodeRecord`

| Field | Type / meaning |
| --- | --- |
| `episode_id`, `question_id`, `condition` | Required nonempty identifier strings |
| `status` | Required `completed`, `step_limit`, `context_limit`, `timeout`, or `error` |
| `final_answer` | Optional string preserving the submitted answer; required non-null for `completed` |
| `correct` | Optional external boolean grade; `None` means ungraded for completed episodes |
| `latency_seconds` | Optional end-to-end episode duration in seconds |
| `agent_steps`, `tool_calls`, `searches`, `documents_opened` | Optional nonnegative integer counts supplied by the agent |
| `metadata` | Optional JSON-compatible object |

`completed` means an answer was submitted. Correctness comes from grading. Every other
status has an effective incorrect outcome even if `correct=None`; `correct=True`
is rejected for these statuses. An unsuccessful episode may retain answer text
for diagnosis, but the non-completed status still makes it incorrect.

All supplied durations/costs must be finite and nonnegative. Booleans are rejected
as numerical measurements. Metadata accepts JSON objects with string keys,
arrays, strings, finite numbers, booleans and null. Records are frozen dataclasses;
nested metadata should also be treated
as immutable. Storage revalidates records before writing.

## JSONL contract

One JSON object per UTF-8 line; documented nulls and metadata survive round trips.
Unknown top-level fields are rejected. Put intentional supplementary fields in
`metadata`.
Readers skip blank lines and raise `ValueError` including the path and line number
for malformed records. Missing required fields are errors. Missing optional fields
use their defaults. Missing files raise normal filesystem exceptions.

## Accounting and outputs

`EpisodeSummary` contains the original `episode` record (including its answer),
call/error-call counts, distinct observed `model_ids` and `config_ids`, token and
cost totals, and summed model-call latency. `None` in `config_ids` explicitly
means a call had no configuration identifier. Context measurements remain in the
individual call records.

`ConditionSummary` contains question IDs, episode and grading counts, observed
model/config identifiers, usage/cost totals, accuracy, cost per correct task,
behaviour-count totals, and latency summaries. `graded_count` counts **known
effective outcomes**, including failed episodes automatically scored incorrect;
`correct_count` counts known correct answers even if other episodes await grading. If any
completed episode is ungraded, accuracy and cost per successful task are `None`.

| Metric | Definition |
| --- | --- |
| Total tokens | Total input + total output; never add cached input again |
| Accuracy | Correct episodes / all episodes, including unsuccessful episodes |
| Accuracy retention | Method accuracy / baseline accuracy |
| Token reduction | `1 - method total tokens / baseline total tokens` |
| Total API cost | All research and memory attempt costs, including failures |
| Cost per successful task | Cost of all episodes / correct episodes |

All ratios use fractions (0.5 means 50%). Zero denominators yield `None`.
Savings may be negative and accuracy retention may exceed one. The proposal's
joint target is reduction >= 0.50 and retention >= 0.90.

Missing values propagate **only to affected metrics**. For example, unknown cost
makes total cost and cost per success unknown, while known tokens and accuracy
remain usable. Input/output totals remain usable when cached usage is unknown.
Every sum requires complete measurements. Zero is known; `None` is unavailable.

`model_call_latency_seconds` sums attempt durations. `episode_latency` summarizes
end-to-end episode durations; `model_call_time_per_episode` summarizes per-episode
model-call totals. These are distinct scopes. Each `LatencySummary` has `count`,
`measured_count`, `mean`, `median`, `minimum`, and `maximum`. If any value in that
summary is missing, all four statistics are `None` and the counts expose missing
coverage. Empty latency collections also have undefined statistics. Behaviour
totals require every episode to supply that count. Agent counters are supplied
separately from model-call counts.

An explicitly empty call collection sums to zero tokens, cost, and model-call
time, with empty model/config ID lists. The integrator must ensure capture
completeness before interpreting an empty list as a call-free episode. An empty
episode collection produces an empty condition list; `accuracy([])` is undefined.

Aggregation rejects duplicate call/episode IDs, calls without matching episodes,
and repeated questions within a condition. Keep repeated experimental runs in
separate aggregation inputs. Comparisons require identical question sets and
use ratios of aggregate totals over those questions.

Derived summaries can be serialized with `dataclasses.asdict()` and ordinary
`json.dumps(..., allow_nan=False)`. JSONL readers accept the two input record types.
Create summaries through aggregation rather than constructing them manually.

## Synthetic usage example

Run from the repository root. `TemporaryDirectory` keeps this example portable
and removes its synthetic files after use.

```python
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from evaluation import (
    LLMCallRecord, EpisodeRecord, append_call, append_episode,
    read_calls, read_episodes, aggregate_episodes, aggregate_conditions,
)

call = LLMCallRecord(
    call_id="synthetic-c1", episode_id="synthetic-e1",
    model_id="synthetic-model", config_id="synthetic-settings-v1",
    purpose="research", status="success", input_tokens=100,
    cached_input_tokens=40, output_tokens=20, context_size_tokens=95,
    api_cost_usd=0.01, latency_seconds=0.5,
)
episode = EpisodeRecord(
    episode_id="synthetic-e1", question_id="synthetic-q1",
    condition="baseline", status="completed", final_answer="Synthetic answer",
    correct=True, latency_seconds=2.0,
)
with TemporaryDirectory() as directory:
    calls_path = Path(directory) / "calls.jsonl"
    episodes_path = Path(directory) / "episodes.jsonl"
    append_call(calls_path, call)
    append_episode(episodes_path, episode)
    summaries = aggregate_episodes(read_calls(calls_path), read_episodes(episodes_path))
    condition = aggregate_conditions(summaries)[0]
    assert condition.total_tokens == 120  # Cached tokens are included in input.
    print(json.dumps(asdict(condition), indent=2, allow_nan=False))
```

## Shared-agent integration

`ResearchAgent.api_call()` records both research requests and the memory callback
used by condition B. `normalize_run()` maps these events and episode outcomes to
`LLMCallRecord` and `EpisodeRecord`; `evaluate.build_summary()` reuses the shared
aggregators. Grading calls use a separate accounting stream.

| Agent data | Evaluation handoff |
| --- | --- |
| Research or memory wrapper attempt | Usage, outcome, model identity and latency in `LLMCallRecord` |
| Post-management messages | `context_estimated_tokens`; this is a threshold estimate, distinct from provider usage |
| Submitted answer and termination | `final_answer` and normalized episode status |
| `cm.stats()` | Episode metadata and optional method-operation reporting |
| `manager_runtime` | Resolved method name and compressor/device details |
| `initialization_seconds` | Manager initialization, separate from the existing episode timing boundary |

The SDK retry policy is preserved. An observed wrapper attempt can contain
SDK-internal retries; those subattempts are not separately visible. Missing usage
or costs remain unknown, including failed attempts. Cached input is charged as a
subset of total input when the supplied pricing distinguishes cache reads.

Episode duration starts after manager initialization. Local compression is
inside episode duration and separately reported by the method; avoid adding it
twice. Timeouts and spending checks occur between iterations or episodes and
cannot interrupt synchronous work or guarantee a hard bill cap. Record cache-hit
and initialization behavior when comparing latency.

## Configuration and storage

The entry points load `.env` before resolving CLI/environment/default model
precedence. Configuration fingerprints include common prompts, tools, model
settings, retrieval and stopping limits, method defaults/overrides, SDK version,
and relevant source hashes. Index identity uses local file metadata rather than
a full corpus content hash. Pricing and diagnostic options remain separate.

Resume requires matching selection, dataset and research configuration. Legacy
manifests without the required provenance remain available for audit and require
a fresh directory for new research. Repeated questions within a condition are
rejected. Current execution uses existing local index resources; package imports
for tests and CLI help defer providers, searchers and context managers.

Ordinary episode records contain accounting and tool metadata. Opt-in diagnostics
store the exact outgoing request body, response message, and tool outputs before
and after context management. Content hashes deduplicate repeated payloads;
keep their `.diagnostics/` directory alongside the run when moving local files.

## Grading and compatible-run reports

The grader preserves its existing semantic-equivalence criteria and decoding
settings. Cached labels are bound to question, answer, reference, termination
status, judge model, prompt, decoding settings and parser version. Mismatched
labels are excluded. A cached null judgment remains unknown and requires review
before a separately authorized retry. Raw verdicts, returned model identifiers,
usage and pricing assumptions support audit. `evaluate.py --offline` creates no
provider client and uses only matching cached labels.

`python -m evaluation.pilot_report --help` describes the offline multi-directory
report. Supply `--judge-model` explicitly to select matching cached grades.
Its selection manifest supplies expected IDs only. Every source run must
have compatible schema-v2 dataset/configuration/revision metadata, and its ID
selection must be disjoint from the other sources and the caller-supplied main
selection. The report validates existing records, exposes missing episodes, and
reuses `evaluate.build_summary()` and cached-grade checks. Source runs and grades
are read without modification. The CLI prints JSON; `--require-complete` returns
a nonzero exit status when research episodes remain missing.

Reports include condition-level metrics, separately accounted grading costs,
per-episode usage/outcomes, initialization-inclusive timing, and method-operation
counters. Unknown measurements remain unknown. The caller remains responsible
for selecting comparable environments and cache policies; matching IDs and
configuration metadata alone cannot establish full experimental comparability.

## Tests

From the repository root:

```sh
python -B tests/run_offline.py
```

This guards network connections and runtime-only imports before discovering the
synthetic unit and regression tests. The tests require no credentials, downloaded
resources, saved experiments, or model packages.
