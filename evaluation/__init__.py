"""Independent, standard-library evaluation records, storage and summaries."""

from .aggregation import aggregate_conditions, aggregate_episodes, compare_conditions
from .logging import append_call, append_episode, read_calls, read_episodes
from .schemas import (ComparisonSummary, ConditionSummary, EpisodeRecord,
                      EpisodeSummary, LatencySummary, LLMCallRecord)

__all__ = [
    "LLMCallRecord", "EpisodeRecord", "EpisodeSummary", "ConditionSummary",
    "ComparisonSummary", "LatencySummary", "append_call", "append_episode",
    "read_calls", "read_episodes", "aggregate_episodes", "aggregate_conditions",
    "compare_conditions",
]
