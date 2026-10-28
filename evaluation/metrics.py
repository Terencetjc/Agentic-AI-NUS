"""Small metric functions; None means unavailable or mathematically undefined."""

from statistics import mean, median
from typing import Iterable, Optional, TypeVar

from .schemas import LatencySummary

Number = TypeVar("Number", int, float)


def complete_sum(values: Iterable[Optional[Number]]) -> Optional[Number]:
    """Sum only when every measurement is known. An empty collection sums to 0."""
    items = list(values)
    if any(value is None for value in items):
        return None
    return sum(items)


def accuracy(outcomes: Iterable[Optional[bool]]) -> Optional[float]:
    """Return the correct fraction, or None for empty or ungraded inputs."""
    items = list(outcomes)
    if any(value is not None and type(value) is not bool for value in items):
        raise ValueError("outcomes must contain booleans or None")
    if not items or any(value is None for value in items):
        return None
    return sum(items) / len(items)


def accuracy_retention(method: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Divide method accuracy by baseline; return None for missing or zero baseline."""
    if method is None or baseline is None or baseline == 0:
        return None
    return method / baseline


def total_tokens(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[int]:
    """Sum input and output usage; input already includes cached tokens."""
    return complete_sum((input_tokens, output_tokens))


def token_reduction(method: Optional[int], baseline: Optional[int]) -> Optional[float]:
    """Return 1 - method / baseline, or None for missing or zero baseline usage."""
    if method is None or baseline is None or baseline == 0:
        return None
    return 1 - method / baseline


def total_api_cost(costs: Iterable[Optional[float]]) -> Optional[float]:
    """Sum supplied USD costs, returning None if any cost is unavailable."""
    return complete_sum(costs)


def cost_per_successful_task(
    total_cost: Optional[float], correct_count: Optional[int]
) -> Optional[float]:
    """Divide all episode costs by correct tasks; return None if undefined."""
    if total_cost is None or correct_count is None or correct_count == 0:
        return None
    return total_cost / correct_count


def latency_summary(values: Iterable[Optional[float]]) -> LatencySummary:
    """Summarize durations in seconds and expose measurement coverage.

    Empty inputs or missing durations yield None for all four statistics.
    """
    items = list(values)
    measured_count = sum(value is not None for value in items)
    if not items or measured_count != len(items):
        return LatencySummary(len(items), measured_count, None, None, None, None)
    return LatencySummary(len(items), measured_count, mean(items), median(items),
                          min(items), max(items))
