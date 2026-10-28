"""Verify shared-agent correctness with synthetic records and local fakes."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
import benchmark
import evaluate
from evaluation.adapters import normalize_run
from evaluation.diagnostics import DiagnosticStore
from evaluation.experiment import (estimate_cost, fingerprint, manager_settings, pricing_from_args,
                                   read_records, research_configuration, resolve_model, validate_resume)

ROOT = Path(__file__).resolve().parents[1]


def arguments(**changes):
    values = dict(model="synthetic-model", model_source="test", cm_args="{}", condition_cm_args="{}",
                  max_steps=40, timeout=900, top_k=5, snippet_chars=400, document_chars=12000,
                  api_retries=3, max_output_tokens=2048, save_history=False,
                  input_per_million=0.4, cached_per_million=0.1, output_per_million=1.6,
                  index_dir="synthetic-missing-index")
    values.update(changes)
    return SimpleNamespace(**values)


def tool(name, args, identifier="tool-1"):
    return SimpleNamespace(id=identifier, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class Message:
    def __init__(self, calls=(), content=None):
        self.tool_calls, self.content = list(calls), content

    def model_dump(self, **kwargs):
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.function.name, "arguments": c.function.arguments}} for c in self.tool_calls]
        return data


class Clock:
    def __init__(self):
        self.value = 0
        self.waits = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += seconds


class Client:
    max_retries = 2  # Mirrors the observable SDK configuration, not network attempts.

    def __init__(self, replies, clock, usage=True, cached=40):
        self.replies = iter(replies)
        self.requests = []
        self.clock, self.usage, self.cached = clock, usage, cached
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **params):
        self.requests.append(copy.deepcopy(params))
        self.clock.value += 0.5
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached)) if self.usage else None
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)], usage=usage,
                               model="synthetic-returned-model", id="synthetic-response-id")


class Manager:
    def __init__(self, memory=False, clock=None):
        self.memory, self.clock = memory, clock
        self.written = False

    def extra_tools(self):
        return []

    def build_messages(self, history):
        if self.memory and not self.written:
            self.written = True
            self.llm([{"role": "user", "content": "synthetic memory"}], "memory_write")
        return [dict(m, content="managed input") if m["role"] == "user" else m for m in history]

    def handle_tool(self, name, args):
        return None

    def on_observation(self, name, args, text):
        return text

    def stats(self):
        return {"memory_llm_calls": int(self.written)}


class Searcher:
    def __init__(self, text="synthetic document"):
        self.text, self.searches, self.opened = text, [], []

    def search(self, query, k):
        self.searches.append((query, k))
        return [SimpleNamespace(docid="123")]

    def doc(self, identifier):
        self.opened.append(identifier)
        if self.text is None:
            return None
        return SimpleNamespace(raw=lambda: json.dumps({"contents": self.text}))


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_agent(self, replies, args=None, manager=None, searcher=None, diagnostics=False, usage=True, cached=40):
        clock = Clock()
        client = Client(replies, clock, usage, cached)
        manager = manager or Manager()
        def factory(condition, llm, **kwargs):
            manager.llm = llm
            return manager
        searcher = searcher or Searcher()
        store = DiagnosticStore(self.directory / ".diagnostics" if diagnostics else None)
        instance = agent.ResearchAgent(searcher, client, "A", args or arguments(), factory,
                                       token_counter=lambda messages: 123, clock=clock, diagnostics=store)
        row = instance.run({"id": "synthetic-q", "question": "synthetic question", "reference_answer": "a"})
        return row, client, searcher, clock

    def test_submit_stops_remaining_tools(self):
        message = Message([tool("submit_answer", {"answer": "valid"}),
                           tool("search", {"query": "must be skipped"}, "tool-2")])
        row, client, searcher, _ = self.run_agent([message])
        self.assertEqual(row["answer"], "valid")
        self.assertEqual(row["stop_reason"], "submitted")
        self.assertEqual(searcher.searches, [])
        self.assertEqual(row["events"][-1]["status"], "skipped")
        self.assertEqual(row["steps"], 1)
        _, episode = normalize_run(row)
        self.assertEqual(episode.tool_calls, 1)

    def test_invalid_answers_keep_existing_step_accounting(self):
        for answer in (None, "", "  \n", 123, [], False):
            with self.subTest(answer=answer):
                row, _, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": answer})])],
                                               arguments(max_steps=1))
                self.assertIsNone(row["answer"])
                self.assertEqual(row["stop_reason"], "step_limit")
                self.assertEqual(row["steps"], 1)
                self.assertTrue(row["events"][-1]["tool_error"])

    def test_invalid_answer_can_be_followed_by_valid_submission(self):
        row, _, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": None})]),
                                      Message([tool("submit_answer", {"answer": "ok"})])])
        self.assertEqual(row["answer"], "ok")
        self.assertEqual(row["steps"], 2)

    def test_document_prefix_and_diagnostics_preserve_content(self):
        text = "x" * 12000 + "hidden tail"
        row, client, _, _ = self.run_agent([
            Message([tool("open_document", {"doc_id": "123"})]),
            Message([tool("submit_answer", {"answer": "ok"})])], searcher=Searcher(text), diagnostics=True)
        visible = [m for m in client.requests[1]["messages"] if m["role"] == "tool"][0]
        self.assertEqual(visible["content"], text[:12000])
        event = next(e for e in row["events"] if e.get("tool") == "open_document")
        self.assertEqual(event["original_chars"], len(text))
        self.assertEqual((event["returned_start"], event["returned_end"], event["truncated"]), (0, 12000, True))
        self.assertEqual(event["raw_result_ref"], event["model_result_ref"])

    def test_missing_document_is_logged_as_error(self):
        row, client, _, _ = self.run_agent([
            Message([tool("open_document", {"doc_id": "missing"})]),
            Message([tool("submit_answer", {"answer": "ok"})])], searcher=Searcher(None))
        event = next(e for e in row["events"] if e.get("tool") == "open_document")
        self.assertEqual(event["document_status"], "unavailable")
        self.assertTrue(event["tool_error"])
        self.assertIsNone(event["returned_end"])
        self.assertEqual(client.requests[1]["messages"][-1]["content"], "Document not found or text unavailable.")

    def test_retrieval_exception_and_malformed_tool_are_logged(self):
        searcher = Searcher()
        def broken(identifier):
            raise RuntimeError("synthetic retrieval failure")
        searcher.doc = broken
        row, _, _, _ = self.run_agent([Message([tool("open_document", {"doc_id": "123"})]),
                                      Message([tool("submit_answer", {"answer": "ok"})])], searcher=searcher)
        event = next(e for e in row["events"] if e.get("tool") == "open_document")
        self.assertEqual(event["document_status"], "error")
        self.assertEqual(event["error_type"], "RuntimeError")
        malformed = tool("search", {})
        malformed.function.arguments = "{bad"
        row, _, _, _ = self.run_agent([Message([malformed])], arguments(max_steps=1))
        self.assertTrue(row["events"][-1]["tool_error"])
        self.assertEqual(row["steps"], 1)

    def test_reference_answers_are_excluded_from_research_requests(self):
        row, client, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": "ok"})])])
        self.assertNotIn("reference_answer", json.dumps(client.requests))
        self.assertEqual(row["reference_answer"], "a")

    def test_search_ids_and_exact_managed_inputs_are_saved(self):
        row, client, _, _ = self.run_agent([
            Message([tool("search", {"query": "synthetic"})]),
            Message([tool("submit_answer", {"answer": "ok"})])], diagnostics=True)
        event = next(e for e in row["events"] if e.get("tool") == "search")
        self.assertEqual(event["returned_docids"], ["123"])
        for event, request in zip([e for e in row["events"] if e.get("kind") == "llm_call"], client.requests):
            payload = json.loads((self.directory / ".diagnostics" / (event["model_input_ref"] + ".json")).read_text())
            self.assertEqual(payload, request)
            self.assertEqual(payload["messages"][1]["content"], "managed input")

    def test_diagnostics_are_optional_and_deduplicated(self):
        self.assertIsNone(DiagnosticStore().put("text"))
        store = DiagnosticStore(self.directory)
        first = store.put("text")
        self.assertEqual(first, store.put("text"))
        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)

    def test_diagnostics_do_not_change_requests(self):
        replies = [Message([tool("open_document", {"doc_id": "123"})]),
                   Message([tool("submit_answer", {"answer": "ok"})])]
        plain, plain_client, _, _ = self.run_agent(replies, diagnostics=False)
        detailed, detailed_client, _, _ = self.run_agent(replies, diagnostics=True)
        self.assertEqual(plain_client.requests, detailed_client.requests)
        self.assertFalse(plain["diagnostics_enabled"])
        self.assertTrue(detailed["diagnostics_enabled"])

    def test_research_and_memory_accounting(self):
        row, _, _, _ = self.run_agent([Message(content="memory notes"),
                                      Message([tool("submit_answer", {"answer": "ok"})])], manager=Manager(memory=True))
        self.assertEqual(row["usage"]["input_tokens"], 200)
        self.assertEqual(row["usage"]["output_tokens"], 40)
        self.assertEqual(row["usage"]["cached_input_tokens"], 80)
        self.assertAlmostEqual(row["usage"]["estimated_cost_usd"], 0.00012)
        calls, _ = normalize_run(row)
        self.assertEqual([c.purpose for c in calls], ["memory", "research"])
        self.assertEqual(calls[0].metadata["returned_model"], "synthetic-returned-model")

    def test_missing_usage_and_pricing_stay_unknown(self):
        row, _, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": "ok"})])], usage=False)
        self.assertIsNone(row["usage"]["input_tokens"])
        self.assertIsNone(row["usage"]["estimated_cost_usd"])
        row, _, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": "ok"})])],
                                     arguments(input_per_million=None))
        self.assertIsNone(row["usage"]["estimated_cost_usd"])
        self.assertEqual(row["usage"]["input_tokens"], 100)

    def test_invalid_cached_usage_stays_unknown(self):
        row, _, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": "ok"})])], cached=101)
        self.assertIsNone(row["usage"]["cached_input_tokens"])
        self.assertIsNone(row["usage"]["estimated_cost_usd"])
        self.assertEqual(row["usage"]["input_tokens"], 100)
        self.assertTrue(row["events"][0]["usage_validation_errors"])

    def test_retry_count_and_sdk_policy_are_preserved(self):
        class RateLimitError(Exception):
            pass
        row, client, _, clock = self.run_agent([RateLimitError("synthetic"), RateLimitError("synthetic"),
                                               Message([tool("submit_answer", {"answer": "ok"})])],
                                              arguments(api_retries=2))
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(len(clock.waits), 2)
        self.assertEqual(row["usage"]["api_calls"], 3)
        self.assertEqual(row["usage"]["successful_calls"], 1)
        self.assertIsNone(row["usage"]["input_tokens"])
        self.assertIsNone(row["usage"]["estimated_cost_usd"])
        self.assertAlmostEqual(row["usage"]["api_seconds"], 1.5)
        self.assertAlmostEqual(row["usage"]["retry_wait_seconds"], sum(clock.waits))
        self.assertTrue(all("timeout" not in request for request in client.requests))
        self.assertEqual(client.max_retries, 2)

    def test_exhausted_retries_remain_failed_episodes(self):
        class APITimeoutError(Exception):
            pass
        row, client, _, _ = self.run_agent([APITimeoutError("synthetic")] * 3, arguments(api_retries=2))
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(row["stop_reason"], "api_error")
        self.assertIsNone(row["answer"])
        self.assertIs(normalize_run(row)[1].outcome, False)

    def test_timeout_stays_between_iterations(self):
        row, client, _, _ = self.run_agent([Message(content="continue")], arguments(timeout=0.1))
        self.assertEqual(row["stop_reason"], "timeout")
        self.assertEqual(len(client.requests), 1)
        self.assertGreater(row["timeout_overrun_seconds"], 0)

    def test_resolve_model_precedence_and_empty_value(self):
        self.assertEqual(resolve_model("cli", {"OPENROUTER_MODEL": "env"}), ("cli", "cli"))
        self.assertEqual(resolve_model(None, {"OPENROUTER_MODEL": "env"}), ("env", "environment"))
        self.assertEqual(resolve_model(None, {})[0], "openai/gpt-4.1-mini")
        with self.assertRaises(ValueError):
            resolve_model(None, {"OPENROUTER_MODEL": ""})

    def test_bootstrap_reads_environment_after_dotenv(self):
        fake = SimpleNamespace(load_dotenv=lambda: agent.os.environ.update(OPENROUTER_MODEL="from-file"))
        with patch.dict("sys.modules", {"dotenv": fake}), patch.object(agent.os, "environ", {}):
            args = arguments(model=None)
            self.assertEqual(agent.bootstrap_model(args), "from-file")

    def config(self, **changes):
        return research_configuration(arguments(**changes), ROOT, agent.SYSTEM, agent.BASE_TOOLS, "ABCD")

    def test_pricing_and_diagnostics_do_not_change_research_fingerprint(self):
        first = self.config()
        second = self.config(input_per_million=9, save_history=True)
        self.assertEqual(fingerprint(first), fingerprint(second))
        self.assertNotEqual(fingerprint(first), fingerprint(self.config(model="other")))
        self.assertNotEqual(fingerprint(first), fingerprint(self.config(top_k=3)))

    def test_per_condition_args_leave_other_methods_unchanged(self):
        args = arguments(condition_cm_args='{"B":{"max_tokens":16000},"C":{"rate":0.5}}')
        self.assertEqual(manager_settings(args, "A"), {})
        self.assertEqual(manager_settings(args, "B"), {"max_tokens": 16000})
        self.assertEqual(manager_settings(args, "C"), {"rate": 0.5})
        self.assertEqual(manager_settings(args, "D"), {})
        with self.assertRaises(ValueError):
            self.config(cm_args='{"max_tokens":16000}')

    def test_resume_validates_research_but_allows_pricing_changes(self):
        expected = dict(dataset_sha256="dataset", seed=1, selected_ids=["q"], excluded_ids=[], research_config=self.config())
        manifest = dict(expected, schema_version=2, pricing={"input_per_million": 1})
        validate_resume(manifest, dict(expected, pricing={"input_per_million": 2}), {})
        for field, value in (("seed", 2), ("selected_ids", ["other"]),
                             ("research_config", self.config(model="other")), ("excluded_ids", ["x"])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume(manifest, dict(expected, **{field: value}), {})
        with self.assertRaises(ValueError):
            validate_resume(dict(manifest, schema_version=1), expected, {})

    def test_resume_rejects_incompatible_saved_episode(self):
        expected = dict(dataset_sha256="dataset", seed=1, selected_ids=["q"], excluded_ids=[], research_config=self.config())
        manifest = dict(expected, schema_version=2)
        row = {"id": "q", "condition": "A", "config_id": "wrong"}
        with self.assertRaises(ValueError):
            validate_resume(manifest, expected, {"A": {"q": row}})

    def test_implementation_and_prompt_changes_invalidate_resume(self):
        expected = dict(dataset_sha256="dataset", seed=1, selected_ids=["q"], excluded_ids=[], research_config=self.config())
        manifest = dict(expected, schema_version=2)
        for key in ("system_prompt", "base_tools", "implementation", "api_retries", "timeout", "max_steps"):
            changed = copy.deepcopy(expected)
            changed["research_config"]["common"][key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume(manifest, changed, {})
        changed = copy.deepcopy(expected)
        changed["research_config"]["methods"]["B"]["supplied_settings"] = {"max_tokens": 100}
        with self.assertRaises(ValueError):
            validate_resume(manifest, changed, {})

    def test_duplicate_results_rejected(self):
        path = self.directory / "runs.jsonl"
        path.write_text('{"id":"x"}\n{"id":"x"}\n')
        with self.assertRaises(ValueError):
            read_records(path)

    def test_price_unknown_and_genuine_zero_are_distinct(self):
        zero = dict(input_per_million=0, cached_per_million=0, output_per_million=0)
        self.assertEqual(estimate_cost(100, 20, 40, zero), 0)
        self.assertIsNone(estimate_cost(100, 20, 40, dict(zero, output_per_million=None)))
        self.assertIsNone(estimate_cost(100, 20, None, dict(zero, input_per_million=1)))
        with self.assertRaises(ValueError):
            pricing_from_args(arguments(input_per_million=-1))

    def test_missing_index_fails_before_loading_search_library(self):
        with self.assertRaises(RuntimeError):
            agent.load_searcher(self.directory / "missing")

    def test_grading_key_tracks_all_relevant_inputs(self):
        row = dict(question="synthetic", answer="a", reference_answer="a", stop_reason="submitted")
        config = evaluate.grading_configuration("judge")
        key = evaluate.grade_key(row, config)
        for field in ("question", "answer", "reference_answer", "stop_reason"):
            self.assertNotEqual(key, evaluate.grade_key(dict(row, **{field: "changed"}), config))
        for field in ("model", "prompt", "temperature", "max_tokens", "parser_version"):
            self.assertNotEqual(key, evaluate.grade_key(row, dict(config, **{field: "changed"})))

    def test_stale_grades_are_excluded(self):
        row = dict(id="q", condition="A", question="synthetic", answer="a", reference_answer="a", stop_reason="submitted")
        config = evaluate.grading_configuration("judge")
        record = dict(condition="A", id="q", correct=True, grade_key=evaluate.grade_key(row, config))
        path = self.directory / "grades.jsonl"
        path.write_text(json.dumps(record) + "\n")
        self.assertEqual(len(evaluate.cached_grades(path, {"A": {"q": row}}, config)), 1)
        self.assertEqual(evaluate.cached_grades(path, {"A": {"q": dict(row, answer="new")}}, config), {})

    def test_grading_with_fake_client_reuses_cache_without_requests(self):
        row, _, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": "ok"})])])
        folder = self.directory / "synthetic_grading"
        folder.mkdir()
        (folder / "A.jsonl").write_text(json.dumps(row) + "\n")
        (folder / "manifest.json").write_text(json.dumps({"selected_ids": [row["id"]], "seed": 1}))
        client = Client([Message(content="extracted_final_answer: ok\nreasoning: equivalent\ncorrect: yes")], Clock())
        evaluate.evaluate(folder, "synthetic-judge", conditions=["A"], judge_client=client)
        self.assertEqual(len(client.requests), 1)
        record = json.loads((folder / "grades_judge.jsonl").read_text())
        self.assertIn("raw_verdict", record["judge_usage"])
        self.assertIn("grade_key", record)
        evaluate.evaluate(folder, "synthetic-judge", conditions=["A"], judge_client=client)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(len((folder / "grades_judge.jsonl").read_text().splitlines()), 1)

    def test_mismatched_research_conditions_fail_before_judging(self):
        folder = self.directory / "synthetic_mismatch"
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({"selected_ids": ["q"], "seed": 1}))
        for condition, question in (("A", "first"), ("B", "different")):
            row = {"id": "q", "condition": condition, "question": question, "answer": "a",
                   "reference_answer": "a", "stop_reason": "submitted", "events": []}
            (folder / (condition + ".jsonl")).write_text(json.dumps(row) + "\n")
        client = Client([], Clock())
        with self.assertRaises(ValueError):
            evaluate.evaluate(folder, "judge", conditions=["A", "B"], judge_client=client)
        self.assertEqual(client.requests, [])

    def test_judge_audit_and_unparseable_grade(self):
        row = dict(question="synthetic", answer="a", reference_answer="a", stop_reason="submitted")
        client = Client([Message(content="extracted_final_answer: a\nreasoning: equal\ncorrect: yes\nconfidence: 90")], Clock())
        correct, _, details = evaluate.grade(row, client, "judge")
        self.assertTrue(correct)
        self.assertEqual(details["requested_model"], "judge")
        self.assertEqual(details["returned_model"], "synthetic-returned-model")
        self.assertEqual(details["extracted_final_answer"], "a")
        self.assertIsNone(details["estimated_cost_usd"])
        client = Client([Message(content="unparseable")], Clock())
        self.assertIsNone(evaluate.grade(row, client, "judge")[0])

    def test_missing_condition_and_unknown_legacy_model(self):
        with self.assertRaises(ValueError):
            evaluate.load_results(self.directory)
        row = {"id": "q", "condition": "A", "answer": "a", "stop_reason": "submitted",
               "events": [{"input_tokens": 100, "output_tokens": 20, "estimated_cost_usd": 0}]}
        calls, _ = normalize_run(row)
        self.assertIsNone(calls[0].model_id)
        self.assertIsNone(calls[0].api_cost_usd)

    def test_legacy_error_remains_an_unknown_measurement(self):
        row = {"id": "q", "condition": "D", "answer": None, "stop_reason": "api_error",
               "usage": {"api_calls": 0}, "events": [{"step": 1, "error_type": "APIStatusError", "error": "synthetic"}]}
        calls, episode = normalize_run(row)
        self.assertEqual(calls[0].status, "error")
        self.assertIsNone(calls[0].input_tokens)
        self.assertIsNone(calls[0].api_cost_usd)
        self.assertIs(episode.outcome, False)
        with self.assertRaises(ValueError):
            normalize_run(dict(row, events=[], usage={"api_calls": 3}))

    def test_synthetic_records_reconcile_agent_and_evaluation_totals(self):
        rows = {}
        for qid in ("synthetic-first", "synthetic-second"):
            row, _, _, _ = self.run_agent(
                [Message(content="synthetic memory"),
                 Message([tool("submit_answer", {"answer": "synthetic answer"})])],
                manager=Manager(memory=True),
            )
            row.update(id=qid, episode_id="A:" + qid, question=qid)
            for index, event in enumerate(row["events"]):
                if event.get("kind") == "llm_call":
                    event["call_id"] = f"{qid}:call:{index}"
            self.assertEqual(row["usage"]["input_tokens"], 200)
            self.assertEqual(row["usage"]["output_tokens"], 40)
            rows[qid] = row
        grades = {("A", "synthetic-first"): {"correct": True},
                  ("A", "synthetic-second"): {"correct": False}}
        result = evaluate.build_summary({"A": rows}, list(rows), grades)
        summary = result["conditions"]["A"]
        self.assertEqual(summary["input_tokens"], 400)
        self.assertEqual(summary["output_tokens"], 80)
        self.assertEqual(summary["cached_input_tokens"], 160)
        self.assertEqual(summary["total_tokens"], 480)
        self.assertEqual(summary["api_calls"], 4)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["total_cost_usd"], 0.00024)
        self.assertAlmostEqual(summary["avg_wall_seconds"], 1.0)
        self.assertTrue(summary["model_identity_complete"])

    def test_pending_grade_preserves_unknown_cost_per_correct(self):
        rows = {}
        grades = {}
        for qid in ("q1", "q2"):
            row, _, _, _ = self.run_agent([Message([tool("submit_answer", {"answer": "ok"})])])
            row.update(id=qid, episode_id="A:" + qid, question=qid)
            for event in row["events"]:
                if event.get("kind") == "llm_call":
                    event["call_id"] = qid + ":call"
            rows[qid] = row
        grades[("A", "q1")] = {"correct": True}
        result = evaluate.build_summary({"A": rows}, ["q1", "q2"], grades)
        self.assertIsNone(result["conditions"]["A"]["accuracy"])
        self.assertIsNone(result["conditions"]["A"]["cost_per_correct_usd"])
        self.assertIsNotNone(result["conditions"]["A"]["total_cost_usd"])

    def test_cli_help_exits_before_runtime_initialization(self):
        import contextlib
        import io
        for module in (agent, benchmark, evaluate):
            with self.subTest(module=module.__name__), patch("sys.argv", [module.__name__, "--help"]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 0)

    def invoke_offline_benchmark(self, extra):
        import contextlib
        import io
        source = self.directory / "synthetic_questions.jsonl"
        if not source.exists():
            source.write_text(json.dumps({"id": "synthetic-q", "question": "synthetic", "answer": "a"}) + "\n")
        argv = ["benchmark.py", "--questions", str(source), "--count", "1", "--model", "synthetic-model",
                "--out-dir", str(self.directory / "experiment"), *extra]
        def bootstrap(args):
            args.model, args.model_source = resolve_model(args.model, {})
        with patch("sys.argv", argv), patch.object(benchmark, "bootstrap_model", bootstrap), \
                patch.object(benchmark, "load_searcher", side_effect=AssertionError("runtime index initialization")), \
                patch.object(benchmark, "create_client", side_effect=AssertionError("runtime client initialization")), \
                contextlib.redirect_stdout(io.StringIO()):
            benchmark.main()

    def test_preflight_has_no_writes_or_runtime_initialization(self):
        self.invoke_offline_benchmark(["--preflight", "--conditions", "A"])
        self.assertFalse((self.directory / "experiment").exists())

    def test_dry_run_can_resume_subsets_and_changed_reporting_prices(self):
        self.invoke_offline_benchmark(["--dry-run", "--conditions", "A"])
        manifest_path = self.directory / "experiment/manifest.json"
        first = manifest_path.read_bytes()
        manifest = json.loads(first)
        self.assertEqual(set(manifest["research_config"]["methods"]), set("ABCD"))
        self.invoke_offline_benchmark(["--dry-run", "--conditions", "B", "C", "D", "--input-per-million", "2"])
        self.assertEqual(manifest_path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
