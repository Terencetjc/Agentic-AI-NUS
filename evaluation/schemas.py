"""Provider-independent input records and derived evaluation summaries."""

import json
import math
from dataclasses import dataclass
from typing import Any, Optional


def _identifier(name: str, value: str) -> None:
    """Require a string identifier containing at least one visible character."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _count(name: str, value: Optional[int]) -> None:
    """Validate an optional nonnegative integer, rejecting booleans."""
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")


def _measurement(name: str, value: Optional[float]) -> None:
    """Validate an optional finite, nonnegative numeric measurement."""
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative, or None")


def _metadata(value: Optional[dict[str, Any]]) -> None:
    """Validate an optional JSON object with finite numbers and string keys."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(isinstance(key, str) for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("metadata must contain only JSON-compatible values")

    if value is not None:
        if type(value) is not dict:
            raise ValueError("metadata must be an object or None")
        try:
            check(value)
            json.dumps(value, allow_nan=False)
        except (RecursionError, TypeError, ValueError) as exc:
            raise ValueError("metadata must be a finite JSON-compatible object") from exc


@dataclass(frozen=True)
class LLMCallRecord:
    """Record one API attempt, using None for unavailable measurements.

    Input tokens include the cached subset. Cost is supplied in USD and latency
    in seconds. The config identifier references externally stored settings.
    """

    call_id: str
    episode_id: str
    model_id: Optional[str]
    purpose: str
    status: str
    config_id: Optional[str] = None
    step_index: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    context_size_tokens: Optional[int] = None
    api_cost_usd: Optional[float] = None
    latency_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Raise ValueError for invalid identifiers, statuses, or measurements."""
        for name in ("call_id", "episode_id"):
            _identifier(name, getattr(self, name))
        if self.model_id is not None:
            _identifier("model_id", self.model_id)
        if self.config_id is not None:
            _identifier("config_id", self.config_id)
        if self.purpose not in ("research", "memory"):
            raise ValueError("purpose must be research or memory")
        if self.status not in ("success", "error"):
            raise ValueError("status must be success or error")
        for name in ("step_index", "input_tokens", "output_tokens",
                     "cached_input_tokens", "context_size_tokens"):
            _count(name, getattr(self, name))
        if (self.cached_input_tokens is not None and self.input_tokens is not None
                and self.cached_input_tokens > self.input_tokens):
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        for name in ("api_cost_usd", "latency_seconds"):
            _measurement(name, getattr(self, name))
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise ValueError("error_message must be a string or None")
        _metadata(self.metadata)


@dataclass(frozen=True)
class EpisodeRecord:
    """Store a finished episode's answer, outcome, timing, and optional counts.

    Completed episodes require an answer and may await external grading.
    Other termination statuses have an effective incorrect outcome.
    """

    episode_id: str
    question_id: str
    condition: str
    status: str
    final_answer: Optional[str] = None
    correct: Optional[bool] = None
    latency_seconds: Optional[float] = None
    agent_steps: Optional[int] = None
    tool_calls: Optional[int] = None
    searches: Optional[int] = None
    documents_opened: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate the episode and reject conflicting status and grade values."""
        for name in ("episode_id", "question_id", "condition"):
            _identifier(name, getattr(self, name))
        if self.status not in ("completed", "step_limit", "context_limit", "timeout", "error"):
            raise ValueError("invalid episode status")
        if self.final_answer is not None and not isinstance(self.final_answer, str):
            raise ValueError("final_answer must be a string or None")
        if self.status == "completed" and self.final_answer is None:
            raise ValueError("completed episodes require final_answer")
        if self.correct is not None and type(self.correct) is not bool:
            raise ValueError("correct must be a boolean or None")
        if self.status != "completed" and self.correct is True:
            raise ValueError("non-completed episodes cannot be correct")
        _measurement("latency_seconds", self.latency_seconds)
        for name in ("agent_steps", "tool_calls", "searches", "documents_opened"):
            _count(name, getattr(self, name))
        _metadata(self.metadata)

    @property
    def outcome(self) -> Optional[bool]:
        """Return the external grade for completed episodes, otherwise False."""
        return self.correct if self.status == "completed" else False


@dataclass(frozen=True)
class LatencySummary:
    """Store measurement coverage and statistics for one latency scope."""

    count: int
    measured_count: int
    mean: Optional[float]
    median: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]


@dataclass(frozen=True)
class EpisodeSummary:
    """Combine an episode record with totals from its observed API attempts."""

    episode: EpisodeRecord
    call_count: int
    error_call_count: int
    model_ids: list[str]
    config_ids: list[Optional[str]]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    total_tokens: Optional[int]
    api_cost_usd: Optional[float]
    model_call_latency_seconds: Optional[float]
    model_identity_complete: bool = True


@dataclass(frozen=True)
class ConditionSummary:
    """Store usage, outcomes, behaviour counts, and latency for one condition.

    Graded count includes automatic incorrect outcomes for failed episodes.
    Correct count includes known correct answers while other grades are pending.
    """

    condition: str
    question_ids: list[str]
    episode_count: int
    graded_count: int
    ungraded_count: int
    correct_count: int
    accuracy: Optional[float]
    call_count: int
    error_call_count: int
    model_ids: list[str]
    config_ids: list[Optional[str]]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    total_tokens: Optional[int]
    api_cost_usd: Optional[float]
    cost_per_successful_task_usd: Optional[float]
    model_call_latency_seconds: Optional[float]
    episode_latency: LatencySummary
    model_call_time_per_episode: LatencySummary
    agent_steps: Optional[int]
    tool_calls: Optional[int]
    searches: Optional[int]
    documents_opened: Optional[int]
    model_identity_complete: bool = True


@dataclass(frozen=True)
class ComparisonSummary:
    """Store accuracy retention and token reduction for matched question sets."""

    condition: str
    baseline_condition: str
    question_ids: list[str]
    accuracy_retention: Optional[float]
    token_reduction: Optional[float]
