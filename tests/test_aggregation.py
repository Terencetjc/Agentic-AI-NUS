"""End-to-end accounting with synthetic records and known totals."""

import json
import unittest
from dataclasses import asdict, replace

from evaluation import (EpisodeRecord, LLMCallRecord, aggregate_conditions,
                        aggregate_episodes, compare_conditions)


def episode(episode_id="e1", question_id="q1", condition="baseline", **kwargs):
    values = dict(status="completed", final_answer="synthetic answer", correct=True,
                  latency_seconds=10, agent_steps=2, tool_calls=3,
                  searches=1, documents_opened=2)
    values.update(kwargs)
    return EpisodeRecord(episode_id, question_id, condition, **values)


def call(call_id="c1", episode_id="e1", **kwargs):
    values = dict(model_id="synthetic-model", purpose="research", status="success",
                  config_id="synthetic-config", input_tokens=80, output_tokens=20,
                  cached_input_tokens=30, api_cost_usd=0.1, latency_seconds=1)
    values.update(kwargs)
    return LLMCallRecord(call_id, episode_id, **values)


def conditions(calls, episodes):
    return {item.condition: item for item in
            aggregate_conditions(aggregate_episodes(calls, episodes))}


class AggregationTests(unittest.TestCase):
    def test_research_memory_and_error_attempts_all_count(self):
        calls = [call(input_tokens=100, output_tokens=20, cached_input_tokens=40),
                 call("c2", purpose="memory", input_tokens=20, output_tokens=10,
                      cached_input_tokens=0, latency_seconds=2),
                 call("c3", status="error", input_tokens=5, output_tokens=0,
                      cached_input_tokens=0, latency_seconds=0.5)]
        result = aggregate_episodes(calls, [episode()])[0]
        self.assertEqual((result.input_tokens, result.output_tokens, result.total_tokens),
                         (125, 30, 155))
        self.assertEqual(result.cached_input_tokens, 40)
        self.assertEqual((result.call_count, result.error_call_count), (3, 1))
        self.assertAlmostEqual(result.api_cost_usd, 0.3)
        self.assertEqual(result.model_call_latency_seconds, 3.5)
        self.assertEqual(result.episode.latency_seconds, 10)

    def test_failed_episodes_keep_cost_and_count_as_incorrect(self):
        records = [episode(), episode("e2", "q2", status="timeout",
                                     final_answer=None, correct=None)]
        result = conditions([call(), call("c2", "e2", api_cost_usd=0.3)], records)["baseline"]
        self.assertEqual(result.accuracy, 0.5)
        self.assertAlmostEqual(result.api_cost_usd, 0.4)
        self.assertAlmostEqual(result.cost_per_successful_task_usd, 0.4)
        self.assertEqual((result.graded_count, result.ungraded_count), (2, 0))
        self.assertEqual((result.agent_steps, result.tool_calls, result.searches,
                          result.documents_opened), (4, 6, 2, 4))

    def test_ungraded_accuracy_and_dependent_metrics_unknown(self):
        records = [episode(), episode("e2", "q2", correct=None)]
        result = conditions([call(), call("c2", "e2")], records)["baseline"]
        self.assertEqual((result.correct_count, result.graded_count, result.ungraded_count),
                         (1, 1, 1))
        self.assertIsNone(result.accuracy)
        self.assertIsNone(result.cost_per_successful_task_usd)
        self.assertEqual(result.total_tokens, 200)
        self.assertIsNone(compare_conditions(result, result).accuracy_retention)

    def test_paired_example_matches_proposal_formulas(self):
        records = [episode("b1", "q1"), episode("b2", "q2"),
                   episode("m1", "q1", "method"),
                   episode("m2", "q2", "method", correct=False)]
        calls = [call("b1", "b1"), call("b2", "b2"),
                 call("m1", "m1", input_tokens=40, output_tokens=10),
                 call("m2", "m2", input_tokens=40, output_tokens=10)]
        results = conditions(calls, records)
        comparison = compare_conditions(results["method"], results["baseline"])
        self.assertEqual(results["baseline"].total_tokens, 200)
        self.assertEqual(results["method"].total_tokens, 100)
        self.assertEqual(comparison.accuracy_retention, 0.5)
        self.assertEqual(comparison.token_reduction, 0.5)
        self.assertFalse(comparison.token_reduction >= 0.5 and
                         comparison.accuracy_retention >= 0.9)
        json.dumps(asdict(results["method"]), allow_nan=False)

    def test_missing_values_propagate_only_to_dependent_metrics(self):
        calls = [call(), call("c2", "e2", input_tokens=None,
                              api_cost_usd=None, latency_seconds=None)]
        records = [episode(), episode("e2", "q2", searches=None, latency_seconds=None)]
        result = conditions(calls, records)["baseline"]
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.total_tokens)
        self.assertIsNone(result.api_cost_usd)
        self.assertIsNone(result.cost_per_successful_task_usd)
        self.assertIsNone(result.model_call_latency_seconds)
        self.assertIsNone(result.model_call_time_per_episode.mean)
        self.assertIsNone(result.episode_latency.mean)
        self.assertIsNone(result.searches)
        self.assertEqual(result.output_tokens, 40)
        self.assertEqual(result.cached_input_tokens, 60)
        self.assertEqual(result.accuracy, 1)
        self.assertEqual(result.tool_calls, 6)

    def test_unknown_cached_usage_does_not_hide_known_total_usage(self):
        result = conditions([call(cached_input_tokens=None)], [episode()])["baseline"]
        self.assertIsNone(result.cached_input_tokens)
        self.assertEqual(result.total_tokens, 100)

    def test_latency_scopes_are_separate(self):
        result = conditions([call(), call("c2", "e2", latency_seconds=3)],
                            [episode(), episode("e2", "q2", latency_seconds=20)])["baseline"]
        self.assertEqual(result.episode_latency.mean, 15)
        self.assertEqual(result.model_call_time_per_episode.mean, 2)
        self.assertEqual(result.model_call_latency_seconds, 4)

    def test_model_and_config_identifiers_preserved(self):
        result = conditions([call(), call("c2", model_id="another-model", config_id=None)],
                            [episode()])["baseline"]
        self.assertEqual(result.model_ids, ["another-model", "synthetic-model"])
        self.assertEqual(result.config_ids, [None, "synthetic-config"])

    def test_duplicate_and_orphan_inputs_rejected(self):
        cases = [([call(), call()], [episode()]),
                 ([call()], [episode(), episode()]),
                 ([call(episode_id="missing")], [episode()]),
                 ([], [episode(), episode("e2")])]
        for calls, records in cases:
            with self.subTest(calls=calls, records=records), self.assertRaises(ValueError):
                aggregate_episodes(calls, records)

    def test_condition_aggregation_rejects_repeated_summaries(self):
        summary = aggregate_episodes([call()], [episode()])[0]
        with self.assertRaises(ValueError):
            aggregate_conditions([summary, summary])
        other = replace(summary, episode=replace(summary.episode, episode_id="other"))
        with self.assertRaises(ValueError):
            aggregate_conditions([summary, other])

    def test_question_mismatch_rejected(self):
        results = conditions([], [episode(), episode("e2", "q2", "method")])
        with self.assertRaisesRegex(ValueError, "identical question sets"):
            compare_conditions(results["method"], results["baseline"])

    def test_empty_calls_and_zero_denominators(self):
        result = conditions([], [episode(correct=False)])["baseline"]
        self.assertEqual(result.total_tokens, 0)
        self.assertEqual(result.api_cost_usd, 0)
        self.assertEqual(result.model_ids, [])
        self.assertIsNone(result.cost_per_successful_task_usd)
        comparison = compare_conditions(result, result)
        self.assertIsNone(comparison.token_reduction)
        self.assertIsNone(comparison.accuracy_retention)
        self.assertEqual(aggregate_episodes([], []), [])
        self.assertEqual(aggregate_conditions([]), [])


if __name__ == "__main__":
    unittest.main()
