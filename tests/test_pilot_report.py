"""Check compatible pooling, incomplete runs and cached-grade handling offline."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import grade_key, grading_configuration
from evaluation.experiment import fingerprint
from evaluation.pilot_report import main, review_runs


class PilotReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"common": {"model": "synthetic-model"},
                       "methods": {c: {} for c in "ABCD"}}
        self.judge = "synthetic-judge"

    def make_run(self, name, qid, dataset="synthetic-dataset", complete=True, correct="absent"):
        folder = self.root / name
        folder.mkdir()
        manifest = {"schema_version": 2, "selected_ids": [qid], "selected_count": 1,
                    "dataset_sha256": dataset, "research_config": self.config,
                    "code_revision": "synthetic-revision", "seed": 123, "excluded_ids": []}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        grades = []
        for c in "ABCD":
            row = {"schema_version": 2, "id": qid, "episode_id": f"{c}:{qid}",
                   "condition": c, "question": qid, "answer": "synthetic answer",
                   "reference_answer": "synthetic reference", "requested_model": "synthetic-model",
                   "stop_reason": "submitted", "steps": 1, "total_seconds": 2,
                   "initialization_seconds": 0.5, "pricing": {}, "research_config": self.config,
                   "config_id": fingerprint({"common": self.config["common"], "method": {}}),
                   "usage": {"estimated_cost_usd": 0.1}, "manager_stats": {},
                   "events": [{"kind": "llm_call", "call_id": f"{c}:{qid}:0",
                               "requested_model": "synthetic-model", "input_tokens": 100,
                               "output_tokens": 20, "cached_input_tokens": 40,
                               "estimated_cost_usd": 0.1, "seconds": 1}]}
            if complete:
                (folder / f"{c}.jsonl").write_text(json.dumps(row) + "\n")
            if correct != "absent":
                grades.append({"condition": c, "id": qid, "correct": correct,
                               "grade_key": grade_key(row, grading_configuration(self.judge)),
                               "judge_usage": {"estimated_cost_usd": 0.001, "api_calls": 1}})
        if grades:
            (folder / "grades_judge.jsonl").write_text("\n".join(map(json.dumps, grades)) + "\n")
        return folder

    def test_compatible_union_preserves_ungraded_answers(self):
        folders = [self.make_run("first", "q1"), self.make_run("second", "q2")]
        report = review_runs(folders, ["q1", "q2"], [], self.judge)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["grading_complete"])
        a = report["summary"]["conditions"]["A"]
        self.assertEqual(a["total_tokens"], 240)
        self.assertIsNone(a["accuracy"])
        self.assertEqual(a["ungraded_count"], 2)
        self.assertEqual(a["mean_episode_plus_initialization_seconds"], 2.5)

    def test_valid_cached_grades_are_pooled(self):
        folders = [self.make_run("first", "q1", correct=True),
                   self.make_run("second", "q2", correct=False)]
        report = review_runs(folders, ["q1", "q2"], [], self.judge)
        self.assertTrue(report["grading_complete"])
        self.assertEqual(report["summary"]["conditions"]["A"]["accuracy"], 0.5)
        self.assertEqual(report["summary"]["grading_usage"]["api_calls"], 8)

    def test_null_cached_grade_stays_unknown(self):
        folder = self.make_run("first", "q1", correct=None)
        report = review_runs([folder], ["q1"], [], self.judge)
        self.assertIsNone(report["summary"]["conditions"]["A"]["accuracy"])
        self.assertFalse(report["grading_complete"])

    def test_mixed_datasets_rejected(self):
        folders = [self.make_run("first", "q1"), self.make_run("second", "q2", dataset="changed")]
        with self.assertRaises(ValueError):
            review_runs(folders, ["q1", "q2"], [], self.judge)

    def test_mixed_model_settings_rejected(self):
        first = self.make_run("first", "q1")
        self.config["common"]["model"] = "different-model"
        second = self.make_run("second", "q2")
        with self.assertRaises(ValueError):
            review_runs([first, second], ["q1", "q2"], [], self.judge)

    def test_changed_judge_excludes_cached_labels_but_preserves_spend(self):
        folder = self.make_run("first", "q1", correct=True)
        report = review_runs([folder], ["q1"], [], "different-judge")
        a = report["summary"]["conditions"]["A"]
        self.assertIsNone(a["accuracy"])
        self.assertEqual(a["graded_count"], 0)
        self.assertEqual(report["summary"]["grading_usage"]["api_calls"], 4)

    def test_legacy_manifest_rejected(self):
        folder = self.make_run("first", "q1")
        path = folder / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["schema_version"]
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            review_runs([folder], ["q1"], [], self.judge)

    def test_overlapping_selections_rejected(self):
        folders = [self.make_run("first", "q1"), self.make_run("second", "q1")]
        with self.assertRaises(ValueError):
            review_runs(folders, ["q1"], [], self.judge)

    def test_main_selection_overlap_rejected(self):
        folder = self.make_run("first", "q1")
        with self.assertRaises(ValueError):
            review_runs([folder], ["q1"], ["q1"], self.judge)

    def test_incomplete_research_reports_missing_episodes(self):
        folders = [self.make_run("first", "q1"), self.make_run("second", "q2", complete=False)]
        report = review_runs(folders, ["q1", "q2"], [], self.judge)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["saved_episodes"], 4)
        self.assertEqual(len(report["missing_episodes"]), 4)
        self.assertNotIn("summary", report)

    def test_cli_requires_explicit_judge_selection(self):
        argv = ["pilot_report", "--runs", "synthetic-run", "--selection-manifest",
                "synthetic-selection.json", "--main-ids", "synthetic-main.txt"]
        with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
