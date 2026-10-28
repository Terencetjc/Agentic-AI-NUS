"""Aggregate normalized records, retaining failures and missing measurements."""

from collections import defaultdict
from typing import Iterable, Optional

from . import metrics
from .schemas import (ComparisonSummary, ConditionSummary, EpisodeRecord,
                      EpisodeSummary, LLMCallRecord)


def _configs(values: Iterable[Optional[str]]) -> list[Optional[str]]:
    """Return distinct config identifiers, placing the unknown value first."""
    return sorted(set(values), key=lambda value: (value is not None, value or ""))


def aggregate_episodes(
    calls: Iterable[LLMCallRecord], episodes: Iterable[EpisodeRecord]
) -> list[EpisodeSummary]:
    """Join API attempts to episodes and return summaries in episode order.

    Include research, memory, and error attempts in accounting totals.
    Raise ValueError for duplicate IDs, orphan calls, or repeated questions
    within a condition. Callers supply the complete set of observed attempts.
    """
    records = list(episodes)
    episode_ids = set()
    pairs = set()
    for episode in records:
        if episode.episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        pair = (episode.condition, episode.question_id)
        if pair in pairs:
            raise ValueError(f"duplicate question within condition: {pair}")
        episode_ids.add(episode.episode_id)
        pairs.add(pair)

    grouped = defaultdict(list)
    call_ids = set()
    for call in calls:
        if call.call_id in call_ids:
            raise ValueError(f"duplicate call_id: {call.call_id}")
        if call.episode_id not in episode_ids:
            raise ValueError(f"unknown episode_id: {call.episode_id}")
        call_ids.add(call.call_id)
        grouped[call.episode_id].append(call)

    summaries = []
    for episode in records:
        items = grouped[episode.episode_id]
        inputs = metrics.complete_sum(call.input_tokens for call in items)
        outputs = metrics.complete_sum(call.output_tokens for call in items)
        summaries.append(EpisodeSummary(
            episode=episode,
            call_count=len(items),
            error_call_count=sum(call.status == "error" for call in items),
            model_ids=sorted({call.model_id for call in items if call.model_id is not None}),
            config_ids=_configs(call.config_id for call in items),
            input_tokens=inputs,
            output_tokens=outputs,
            cached_input_tokens=metrics.complete_sum(call.cached_input_tokens for call in items),
            total_tokens=metrics.total_tokens(inputs, outputs),
            api_cost_usd=metrics.total_api_cost(call.api_cost_usd for call in items),
            model_call_latency_seconds=metrics.complete_sum(call.latency_seconds for call in items),
            model_identity_complete=all(call.model_id is not None for call in items),
        ))
    return summaries


def aggregate_conditions(summaries: Iterable[EpisodeSummary]) -> list[ConditionSummary]:
    """Group episode summaries and return results in condition-name order.

    Missing measurements propagate to their dependent totals. Pending grades
    make accuracy and cost per successful task unavailable. Raise ValueError
    for duplicate episode IDs or repeated questions within a condition.
    """
    grouped = defaultdict(list)
    episode_ids = set()
    pairs = set()
    for summary in summaries:
        episode = summary.episode
        if episode.episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        pair = (episode.condition, episode.question_id)
        if pair in pairs:
            raise ValueError(f"duplicate question within condition: {pair}")
        episode_ids.add(episode.episode_id)
        pairs.add(pair)
        grouped[episode.condition].append(summary)

    results = []
    for condition, items in sorted(grouped.items()):
        outcomes = [item.episode.outcome for item in items]
        ungraded = sum(outcome is None for outcome in outcomes)
        correct = sum(outcome is True for outcome in outcomes)
        cost = metrics.total_api_cost(item.api_cost_usd for item in items)
        results.append(ConditionSummary(
            condition=condition,
            model_identity_complete=all(item.model_identity_complete for item in items),
            question_ids=sorted(item.episode.question_id for item in items),
            episode_count=len(items),
            graded_count=len(items) - ungraded,
            ungraded_count=ungraded,
            correct_count=correct,
            accuracy=metrics.accuracy(outcomes),
            call_count=sum(item.call_count for item in items),
            error_call_count=sum(item.error_call_count for item in items),
            model_ids=sorted({model for item in items for model in item.model_ids}),
            config_ids=_configs(config for item in items for config in item.config_ids),
            input_tokens=metrics.complete_sum(item.input_tokens for item in items),
            output_tokens=metrics.complete_sum(item.output_tokens for item in items),
            cached_input_tokens=metrics.complete_sum(item.cached_input_tokens for item in items),
            total_tokens=metrics.complete_sum(item.total_tokens for item in items),
            api_cost_usd=cost,
            cost_per_successful_task_usd=metrics.cost_per_successful_task(
                cost, None if ungraded else correct),
            model_call_latency_seconds=metrics.complete_sum(
                item.model_call_latency_seconds for item in items),
            episode_latency=metrics.latency_summary(item.episode.latency_seconds for item in items),
            model_call_time_per_episode=metrics.latency_summary(
                item.model_call_latency_seconds for item in items),
            **{name: metrics.complete_sum(getattr(item.episode, name) for item in items)
               for name in ("agent_steps", "tool_calls", "searches", "documents_opened")},
        ))
    return results


def compare_conditions(method: ConditionSummary, baseline: ConditionSummary) -> ComparisonSummary:
    """Compare aggregate accuracy and token totals with an explicit baseline.

    Raise ValueError if question sets differ. Callers establish comparability
    of models, settings, tools, and data before interpreting the ratios.
    """
    if set(method.question_ids) != set(baseline.question_ids):
        raise ValueError("baseline comparisons require identical question sets")
    return ComparisonSummary(
        condition=method.condition,
        baseline_condition=baseline.condition,
        question_ids=sorted(method.question_ids),
        accuracy_retention=metrics.accuracy_retention(method.accuracy, baseline.accuracy),
        token_reduction=metrics.token_reduction(method.total_tokens, baseline.total_tokens),
    )
