import argparse
import json
import os
import re
from dataclasses import asdict

from evaluation.adapters import summarize_runs
from evaluation.aggregation import compare_conditions
from evaluation.experiment import estimate_cost, fingerprint, read_records
from evaluation.metrics import complete_sum
from pathlib import Path

CONDITIONS = ('A', 'B', 'C', 'D')
GRADER_PROMPT = '''Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: {response}
[correct_answer]: {correct_answer}
Your judgement must be in the format and criteria specified below:
extracted_final_answer: Extract the final answer from [response], or None if absent.
reasoning: Compare only the extracted answer to [correct_answer]. Do not independently solve the question.
correct: Answer yes if the extracted answer matches the correct answer, including semantically equivalent wording or a small margin of error for numerical problems; otherwise no.
confidence: Your confidence as a percentage.'''

def load_results(folder, conditions=CONDITIONS):
    """Load every required condition file with unique episode IDs."""
    folder = Path(folder)
    runs = {}
    for condition in conditions:
        path = folder / f"{condition}.jsonl"
        if not path.exists():
            raise ValueError(f"Missing required condition file: {path}")
        runs[condition] = read_records(path)
        if any(row.get("condition") != condition for row in runs[condition].values()):
            raise ValueError(f"Condition labels differ from {path}")
    return runs


def grading_configuration(judge_model):
    """Identify the existing grading criteria and decoding settings."""
    return {"model": judge_model, "prompt": GRADER_PROMPT, "temperature": 0,
            "max_tokens": 500, "parser_version": 1,
            "protocol": "llm_judge_nonofficial_settings"}


def grade_key(row, config):
    """Bind a cached grade to its answer, reference, question and protocol."""
    return fingerprint({"question": row.get("question"), "answer": row.get("answer"),
                        "reference": row.get("reference_answer"), "stop_reason": row.get("stop_reason"),
                        "configuration": config})


def cached_grades(path, rows_by_condition, config):
    """Reuse only grades with matching content and protocol fingerprints."""
    grades = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            grade_record = json.loads(line)
            condition, qid = grade_record["condition"], str(grade_record["id"])
            row = rows_by_condition.get(condition, {}).get(qid)
            if row is not None and grade_record.get("grade_key") == grade_key(row, config):
                grades[(condition, qid)] = grade_record
    return grades


def grade(row, judge_client=None, judge_model=None, pricing=None):
    """Grade one saved answer using the unchanged prompt and decoding settings."""
    answer = row.get("answer")
    if row.get("stop_reason") != "submitted" or not isinstance(answer, str) or not answer.strip():
        return False, "not_submitted", {"estimated_cost_usd": 0.0, "api_calls": 0}
    reference = row.get("reference_answer")
    if reference is None or not str(reference).strip():
        return None, "no_reference", {"estimated_cost_usd": 0.0, "api_calls": 0}
    import time
    started = time.monotonic()
    try:
        response = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": GRADER_PROMPT.format(
                question=row["question"], response=answer, correct_answer=reference)}],
            temperature=0, max_tokens=500)
    except Exception as exc:
        return None, "judge_error", {"api_calls": 1, "seconds": time.monotonic() - started,
                                     "error_type": type(exc).__name__, "estimated_cost_usd": None}
    verdict = response.choices[0].message.content or ""
    usage = response.usage
    inputs = getattr(usage, "prompt_tokens", None)
    outputs = getattr(usage, "completion_tokens", None)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    details = {"raw_verdict": verdict, "requested_model": judge_model,
               "returned_model": getattr(response, "model", None), "response_id": getattr(response, "id", None),
               "input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached,
               "seconds": time.monotonic() - started, "api_calls": 1,
               "measurement_scope": "logical judge call; SDK may retry internally",
               "estimated_cost_usd": estimate_cost(inputs, outputs, cached, pricing or {
                   "input_per_million": None, "cached_per_million": None, "output_per_million": None})}
    match = re.search(r"(?im)^correct\s*:\s*(yes|no)\b", verdict)
    extracted = re.search(r"(?im)^extracted_final_answer\s*:\s*(.*)$", verdict)
    reasoning = re.search(r"(?ims)^reasoning\s*:\s*(.*?)(?=^correct\s*:|\Z)", verdict)
    details.update(extracted_final_answer=extracted.group(1).strip() if extracted else None,
                   reasoning=reasoning.group(1).strip() if reasoning else None)
    if not match:
        return None, "judge_unparseable", details
    return match.group(1).lower() == "yes", "llm_judge_nonofficial_settings", details


def build_summary(runs, ids, grades, seed=None):
    """Summarize saved rows through the independent evaluation module."""
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate question IDs")
    expected = set(ids)
    all_rows = []
    for condition, rows in runs.items():
        if set(rows) != expected:
            raise ValueError(f"{condition} has mismatched question IDs")
        config_ids = {row.get("config_id") for row in rows.values()}
        if len(config_ids) > 1:
            raise ValueError(f"Mixed research configurations within {condition}")
        for row in rows.values():
            configuration = row.get("research_config")
            if configuration is not None:
                expected_id = fingerprint({"common": configuration["common"],
                                           "method": configuration["methods"][condition]})
                if row.get("config_id") != expected_id:
                    raise ValueError(f"Invalid configuration fingerprint in {condition}/{row['id']}")
        all_rows.extend(rows[qid] for qid in ids)
    for qid in ids:
        versions = {(runs[c][qid].get("question"), runs[c][qid].get("reference_answer")) for c in runs}
        if len(versions) != 1:
            raise ValueError(f"Question/reference contents differ across conditions: {qid}")
    common_versions = {fingerprint(row["research_config"]["common"]) for row in all_rows
                       if row.get("research_config") is not None}
    if len(common_versions) > 1:
        raise ValueError("Research settings differ across conditions")
    if common_versions and any(row.get("research_config") is None for row in all_rows):
        raise ValueError("Legacy and configured runs require separate comparisons")
    outcomes = {key: value["correct"] for key, value in grades.items()}
    summaries = summarize_runs(all_rows, outcomes)
    baseline = next((item for item in summaries if item.condition == "A"), None)
    result = {"seed": seed, "selected_count": len(ids), "grading": "llm_judge_nonofficial_settings",
              "configuration_complete": all(row.get("config_id") for row in all_rows), "conditions": {}}
    for item in summaries:
        values = asdict(item)
        values.update(questions=item.episode_count, graded=item.graded_count, correct=item.correct_count,
                      avg_tokens=(item.total_tokens / item.episode_count if item.total_tokens is not None else None),
                      api_calls=item.call_count, avg_wall_seconds=item.episode_latency.mean,
                      total_cost_usd=item.api_cost_usd, cost_per_correct_usd=item.cost_per_successful_task_usd)
        rows = runs[item.condition].values()
        reasons = sorted({row.get("stop_reason") for row in rows})
        values["stop_reasons"] = {reason: sum(row.get("stop_reason") == reason for row in rows) for reason in reasons}
        if baseline is not None:
            comparison = compare_conditions(item, baseline)
            values.update(token_reduction_vs_A=comparison.token_reduction,
                          accuracy_retention_vs_A=comparison.accuracy_retention)
        result["conditions"][item.condition] = values
    details = [record.get("judge_usage", {}) for record in grades.values()]
    result["grading_usage"] = {"estimated_cost_usd": complete_sum(d.get("estimated_cost_usd") for d in details),
                               "api_calls": complete_sum(d.get("api_calls") for d in details)}
    return result


def grading_usage(records):
    """Sum observed judge expenditure, including superseded cached grades."""
    details = [record.get("judge_usage", {}) for record in records]
    return {"estimated_cost_usd": complete_sum(d.get("estimated_cost_usd") for d in details),
            "api_calls": complete_sum(d.get("api_calls") for d in details)}


def evaluate(folder, judge_model, conditions=CONDITIONS, offline=False, judge_client=None, pricing=None):
    """Validate saved runs, grade uncached answers, and write a separate summary."""
    folder = Path(folder)
    runs = load_results(folder, conditions)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["selected_ids"]
    if any(set(rows) != set(ids) for rows in runs.values()):
        raise ValueError("Run IDs differ from the manifest")
    build_summary(runs, ids, {}, manifest.get("seed"))
    config = grading_configuration(judge_model)
    grades_path = folder / "grades_judge.jsonl"
    grades = cached_grades(grades_path, runs, config)
    new_grades = []
    if not offline:
        pending = [(c, qid) for c in runs for qid in ids if (c, qid) not in grades]
        needs_client = any(runs[c][qid].get("stop_reason") == "submitted" and
                           runs[c][qid].get("answer") and runs[c][qid].get("reference_answer")
                           for c, qid in pending)
        if pending and needs_client and judge_client is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv()
            judge_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        if pending:
            with grades_path.open("a", encoding="utf-8") as stream:
                for condition, qid in pending:
                    row = runs[condition][qid]
                    correct, method, usage = grade(row, judge_client, judge_model, pricing)
                    record = {"condition": condition, "id": qid, "correct": correct,
                              "method": method, "judge_model": judge_model, "grade_key": grade_key(row, config),
                              "grading_config": config, "pricing": pricing, "judge_usage": usage}
                    grades[(condition, qid)] = record
                    new_grades.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
    summary = build_summary(runs, ids, grades, manifest.get("seed"))
    summary["selected_grades_usage"] = summary.pop("grading_usage")
    historical_grades = [json.loads(line) for line in grades_path.read_text(encoding="utf-8").splitlines()
                         if line.strip()] if grades_path.exists() else []
    summary["grading_usage"] = grading_usage(historical_grades)
    summary["current_grading_invocation_usage"] = grading_usage(new_grades)
    summary["grading_config"] = config
    path = folder / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("Summary:", path)
    for condition, values in summary["conditions"].items():
        print(condition, "accuracy:", values["accuracy"], "tokens:", values["total_tokens"], "cost:", values["total_cost_usd"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--offline", action="store_true", help="Summarize matching cached grades only")
    for name in ("input", "cached", "output"):
        parser.add_argument(f"--judge-{name}-per-million", type=float, default=None)
    args = parser.parse_args()
    pricing = {"input_per_million": args.judge_input_per_million,
               "cached_per_million": args.judge_cached_per_million,
               "output_per_million": args.judge_output_per_million}
    evaluate(Path(args.run_dir), args.judge_model, args.conditions, args.offline, pricing=pricing)


if __name__ == "__main__":
    main()
