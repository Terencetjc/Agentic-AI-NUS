"""Verify record validation and JSONL storage using synthetic inputs."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from evaluation import (EpisodeRecord, LLMCallRecord, append_call, append_episode,
                        read_calls, read_episodes)


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "records.jsonl"
        self.call = LLMCallRecord(
            call_id="c1", episode_id="e1", model_id="synthetic-model",
            purpose="research", status="success", config_id="synthetic-config-v1",
            step_index=0, input_tokens=100, output_tokens=20,
            cached_input_tokens=40, context_size_tokens=90,
            api_cost_usd=0.02, latency_seconds=0.5,
            metadata={"synthetic": True, "notes": [None, "研究", 2, 0.5]},
        )

    def test_calls_round_trip_and_append(self):
        second = replace(self.call, call_id="c2", purpose="memory", status="error",
                         error_message="synthetic error", api_cost_usd=None)
        append_call(str(self.path), self.call)
        append_call(self.path, second)
        self.assertEqual(read_calls(self.path), [self.call, second])
        self.assertEqual(len(self.path.read_text().splitlines()), 2)

    def test_episode_round_trip_preserves_answer_and_counts(self):
        episode = EpisodeRecord(
            "e1", "q1", "baseline", "completed", final_answer="答案",
            correct=None, latency_seconds=3.5, agent_steps=2, tool_calls=3,
            searches=1, documents_opened=2, metadata={"synthetic": True},
        )
        failed = EpisodeRecord("e2", "q2", "baseline", "timeout")
        append_episode(self.path, episode)
        append_episode(self.path, failed)
        self.assertEqual(read_episodes(self.path), [episode, failed])

    def test_blank_lines_ignored_and_empty_file(self):
        self.path.write_text("\n  \n", encoding="utf-8")
        self.assertEqual(read_calls(self.path), [])
        append_call(self.path, self.call)
        self.assertEqual(read_calls(self.path), [self.call])

    def test_append_after_record_without_trailing_newline(self):
        self.path.write_text(json.dumps(asdict(self.call), ensure_ascii=False), encoding="utf-8")
        second = replace(self.call, call_id="c2")
        append_call(self.path, second)
        self.assertEqual(read_calls(self.path), [self.call, second])

    def test_invalid_lines_include_path_and_number(self):
        data = asdict(self.call)
        for invalid in ("{bad", "[]", json.dumps({**data, "unknown": 1}),
                        json.dumps({**data, "input_tokens": True}),
                        json.dumps({**data, "api_cost_usd": float("nan")}), "{}"):
            with self.subTest(invalid=invalid):
                self.path.write_text("\n" + invalid + "\n", encoding="utf-8")
                with self.assertRaises(ValueError) as error:
                    read_calls(self.path)
                self.assertIn(f"{self.path}:2:", str(error.exception))

    def test_wrong_record_type_rejected(self):
        with self.assertRaises(TypeError):
            append_call(self.path, EpisodeRecord("e", "q", "a", "error"))

    def test_unknown_episode_field_rejected(self):
        data = asdict(EpisodeRecord("e", "q", "a", "error"))
        self.path.write_text(json.dumps({**data, "unknown": 1}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, ":1:"):
            read_episodes(self.path)

    def test_call_validation(self):
        changes = [
            {"call_id": " "}, {"episode_id": 4}, {"model_id": ""},
            {"config_id": ""}, {"purpose": "grading"}, {"status": "retry"},
            {"input_tokens": -1}, {"output_tokens": 1.5},
            {"cached_input_tokens": 101}, {"context_size_tokens": True},
            {"step_index": False}, {"api_cost_usd": True},
            {"api_cost_usd": -0.1}, {"latency_seconds": float("inf")},
            {"error_message": 123}, {"metadata": []},
            {"metadata": {1: "not a string key"}},
            {"metadata": {"not_json": (1, 2)}},
            {"metadata": {"not_finite": float("nan")}},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.call, **change)

    def test_episode_validation_and_effective_outcomes(self):
        episode = EpisodeRecord("e", "q", "a", "completed", "answer")
        self.assertIsNone(episode.outcome)
        changes = [
            {"final_answer": None}, {"final_answer": 123}, {"correct": 1},
            {"condition": ""}, {"question_id": ""}, {"status": "invalid"},
            {"agent_steps": True}, {"tool_calls": -1}, {"searches": 1.2},
            {"documents_opened": False}, {"latency_seconds": float("nan")},
            {"status": "timeout", "correct": True},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(episode, **change)
        for status in ("step_limit", "context_limit", "timeout", "error"):
            self.assertIs(EpisodeRecord("e", "q", "a", status).outcome, False)

    def test_mutated_metadata_revalidated_before_writing(self):
        self.call.metadata["bad"] = object()
        with self.assertRaises(ValueError):
            append_call(self.path, self.call)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
