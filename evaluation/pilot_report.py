"""Review compatible development runs using saved records and cached grades."""

import argparse
import json
from pathlib import Path

from evaluate import (CONDITIONS, build_summary, cached_grades,
                      grading_configuration, grading_usage)
from .experiment import read_records, validate_resume
from .metrics import complete_sum, token_reduction


COUNTERS = {
    "A": (),
    "B": ("evictions", "memory_llm_calls", "recall_calls", "evicted_observations"),
    "C": ("compress_calls", "cache_hits", "compress_seconds", "tokens_before", "tokens_after"),
    "D": ("compactions", "observations_compacted", "tokens_removed",
          "duplicates_merged", "conflicts_flagged"),
}


def review_runs(folders, selected_ids, main_ids, judge_model):
    """Validate disjoint run selections and summarize their compatible union.

    Incomplete research produces a progress report. Completed, ungraded answers
    retain unknown correctness. Source episodes, manifests and grades stay intact.
    """
    selected_ids = list(selected_ids)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Expected development IDs must be nonempty and unique")
    if set(selected_ids) & set(main_ids):
        raise ValueError("Development IDs overlap the main evaluation")
    runs = {condition: {} for condition in CONDITIONS}
    grades, historical_grades, sources, missing = {}, [], [], {}
    selection = set()
    reference = None
    config = grading_configuration(judge_model)
    for folder in map(Path, folders):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        ids = manifest["selected_ids"]
        if manifest.get("selected_count") != len(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid question selection in {folder}")
        if selection & set(ids):
            raise ValueError("Run folders contain overlapping question IDs")
        selection.update(ids)
        identity = {key: manifest.get(key) for key in
                    ("dataset_sha256", "research_config", "code_revision", "seed")}
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("Run manifests have incompatible dataset/configuration/version metadata")
        rows = {condition: read_records(folder / f"{condition}.jsonl")
                for condition in CONDITIONS}
        expected = {key: manifest.get(key) for key in
                    ("dataset_sha256", "seed", "selected_ids", "excluded_ids", "research_config")}
        validate_resume(manifest, expected, rows)
        # Reuse evaluator validation even when only part of a condition is saved.
        for condition, records in rows.items():
            if records:
                build_summary({condition: records}, list(records), {}, manifest["seed"])
            missing[f"{folder}/{condition}"] = [qid for qid in ids if qid not in records]
            runs[condition].update(records)
        grade_path = folder / "grades_judge.jsonl"
        grades.update(cached_grades(grade_path, rows, config))
        if grade_path.exists():
            historical_grades.extend(json.loads(line) for line in
                                     grade_path.read_text(encoding="utf-8").splitlines() if line.strip())
        sources.append({"run_dir": str(folder), "selected_ids": ids,
                        "pricing": manifest.get("pricing")})
    if selection != set(selected_ids):
        raise ValueError("Run selections differ from the expected development question set")
    report = {"status": "incomplete" if any(missing.values()) else "complete",
              "selected_ids": selected_ids, "sources": sources,
              "saved_episodes": sum(map(len, runs.values())),
              "missing_episodes": {key: value for key, value in missing.items() if value},
              "grading_config": config}
    if report["status"] == "incomplete":
        report["recorded_research_cost_usd"] = complete_sum(
            (row.get("usage") or {}).get("estimated_cost_usd")
            for records in runs.values() for row in records.values())
        return report
    summary = build_summary(runs, selected_ids, grades, reference["seed"])
    summary["selected_grades_usage"] = summary.pop("grading_usage")
    summary["grading_usage"] = grading_usage(historical_grades)
    for condition, values in summary["conditions"].items():
        rows = [runs[condition][qid] for qid in selected_ids]
        stats = [row.get("manager_stats") or {} for row in rows]
        values["manager_operation_totals"] = {
            key: complete_sum(item.get(key) for item in stats) for key in COUNTERS[condition]}
        if condition == "C":
            totals = values["manager_operation_totals"]
            totals["document_token_reduction"] = token_reduction(
                totals["tokens_after"], totals["tokens_before"])
        condition_grades = [g for g in historical_grades if g["condition"] == condition]
        values["grading_usage"] = grading_usage(condition_grades)
        values["episodes"] = []
        for row in rows:
            cached = grades.get((condition, str(row["id"])), {})
            combined_time = complete_sum((row.get("total_seconds"), row.get("initialization_seconds")))
            values["episodes"].append({
                "question_id": str(row["id"]), "stop_reason": row.get("stop_reason"),
                "correct": cached.get("correct") if row.get("stop_reason") == "submitted" else False,
                "grading_method": cached.get("method"), "usage": row.get("usage"),
                "episode_seconds": row.get("total_seconds"),
                "initialization_seconds": row.get("initialization_seconds"),
                "episode_plus_initialization_seconds": combined_time,
                "manager_stats": row.get("manager_stats"),
                "manager_runtime": row.get("manager_runtime"),
            })
        times = [row["episode_plus_initialization_seconds"] for row in values["episodes"]]
        total = complete_sum(times)
        values["mean_episode_plus_initialization_seconds"] = total / len(times) if total is not None else None
    report["summary"] = summary
    report["grading_complete"] = all(v["ungraded_count"] == 0 for v in summary["conditions"].values())
    return report


def main():
    """Print an offline report without importing providers or writing source runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--selection-manifest", required=True,
                        help="Manifest used only to specify the expected question IDs")
    parser.add_argument("--main-ids", required=True)
    parser.add_argument("--judge-model", required=True,
                        help="Judge identifier used to select matching cached grades")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    selection = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))["selected_ids"]
    main_ids = Path(args.main_ids).read_text(encoding="utf-8").split()
    report = review_runs(args.runs, selection, main_ids, args.judge_model)
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.require_complete and report["status"] != "complete":
        parser.exit(1, "Research episodes remain incomplete; review the missing IDs before grading.\n")


if __name__ == "__main__":
    main()
