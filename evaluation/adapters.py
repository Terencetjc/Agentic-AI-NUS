"""Normalize shared-agent outputs for the existing evaluation summaries."""

from .aggregation import aggregate_conditions, aggregate_episodes
from .schemas import EpisodeRecord, LLMCallRecord

STATUSES = {"submitted": "completed", "step_limit": "step_limit", "timeout": "timeout",
            "context_overflow": "context_limit", "api_error": "error"}


def normalize_run(row, correct=None):
    """Convert one run, preserving unknown legacy model/configuration provenance."""
    episode_id = str(row.get("episode_id") or f"{row['condition']}:{row['id']}")
    model_id = row.get("requested_model")
    calls = []
    if "events" not in row:
        raise ValueError("Run has no call-event records; measurement completeness is unknown")
    events = row["events"]
    has_legacy_attempt_errors = any("error_type" in e and "purpose" in e for e in events)
    for i, event in enumerate(events):
        legacy_error = (row.get("schema_version", 1) < 2 and "error_type" in event and "tool" not in event
                        and ("purpose" in event or not has_legacy_attempt_errors))
        if event.get("kind") != "llm_call" and "input_tokens" not in event and not legacy_error:
            continue
        calls.append(LLMCallRecord(
            call_id=event.get("call_id") or f"{episode_id}:legacy:{i}", episode_id=episode_id,
            model_id=event.get("requested_model") or model_id, config_id=row.get("config_id"),
            purpose="memory" if event.get("purpose") in ("memory", "memory_write") else "research",
            status=event.get("status", "error" if legacy_error else "success"), step_index=event.get("step"),
            input_tokens=event.get("input_tokens"), output_tokens=event.get("output_tokens"),
            cached_input_tokens=event.get("cached_input_tokens"),
            context_size_tokens=event.get("context_estimated_tokens"),
            api_cost_usd=(event.get("estimated_cost_usd") if row.get("pricing") is not None else None),
            latency_seconds=event.get("seconds"), error_message=event.get("error"),
            metadata={"returned_model": event.get("returned_model"),
                      "model_identity_known": bool(event.get("requested_model") or row.get("requested_model")),
                      "measurement_scope": event.get("measurement_scope", "legacy logical call")},
        ))
    if not calls and (row.get("usage") or {}).get("api_calls", 0):
        raise ValueError("Run has usage totals but missing call-event measurements")
    tools = [event for event in row.get("events", []) if "tool" in event and event.get("status") != "skipped"]
    episode = EpisodeRecord(
        episode_id=episode_id, question_id=str(row["id"]), condition=row["condition"],
        status=STATUSES.get(row.get("stop_reason"), "error"), final_answer=row.get("answer"),
        correct=correct, latency_seconds=row.get("total_seconds"), agent_steps=row.get("steps"),
        tool_calls=len(tools) if "events" in row else None,
        searches=sum(e["tool"] == "search" for e in tools) if "events" in row else None,
        documents_opened=sum(e["tool"] == "open_document" for e in tools) if "events" in row else None,
        metadata={"manager_stats": row.get("manager_stats"), "stop_reason": row.get("stop_reason"),
                  "configuration_known": row.get("config_id") is not None},
    )
    return calls, episode


def summarize_runs(rows, grades):
    """Aggregate existing run and grade records through the shared evaluator."""
    calls, episodes = [], []
    for row in rows:
        converted, episode = normalize_run(row, grades.get((row["condition"], str(row["id"]))))
        calls.extend(converted)
        episodes.append(episode)
    return aggregate_conditions(aggregate_episodes(calls, episodes))
