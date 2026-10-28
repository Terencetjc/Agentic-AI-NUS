"""Resolve settings and identify research configurations using local files."""

import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

DEFAULT_MODEL = "openai/gpt-4.1-mini"
METHOD_FILES = {"A": "full_context.py", "B": "memgpt.py",
                "C": "llmlingua_compression.py", "D": "srcm.py"}
COMMON_FIELDS = ("max_steps", "timeout", "top_k", "snippet_chars", "document_chars",
                 "api_retries", "max_output_tokens")


def fingerprint(value):
    """Hash a JSON-compatible value with stable field ordering."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_model(cli_model, environment):
    """Resolve CLI, then OPENROUTER_MODEL, then the documented default."""
    source = "cli" if cli_model is not None else (
        "environment" if "OPENROUTER_MODEL" in environment else "default")
    model = cli_model if cli_model is not None else environment.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Research model must be a nonempty identifier")
    return model.strip(), source


def manager_settings(args, condition):
    """Resolve legacy shared arguments and optional per-condition overrides."""
    shared = json.loads(args.cm_args)
    overrides = json.loads(getattr(args, "condition_cm_args", "{}"))
    if not isinstance(shared, dict) or not isinstance(overrides, dict):
        raise ValueError("Context-manager arguments must be JSON objects")
    if any(key not in METHOD_FILES or not isinstance(value, dict)
           for key, value in overrides.items()):
        raise ValueError("Condition arguments must map A/B/C/D to JSON objects")
    return {**shared, **overrides.get(condition, {})}


def manager_defaults(root, condition):
    """Read literal constructor defaults without importing a context manager."""
    import ast
    source = root / "context_managers" / METHOD_FILES[condition]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                    names = method.args.args[-len(method.args.defaults):]
                    return {name.arg: ast.literal_eval(value)
                            for name, value in zip(names, method.args.defaults)}
    return {}


def research_configuration(args, root, system_prompt, tools, conditions):
    """Describe research-affecting inputs; pricing and diagnostics stay separate."""
    root = Path(root)
    def file_hash(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()
    common = {name: getattr(args, name) for name in COMMON_FIELDS}
    common.update(model=args.model, temperature=0, system_prompt=system_prompt,
                  base_tools=tools, index_identity=index_identity(args.index_dir),
                  retry_policy="outer api_retries plus unchanged SDK default retries",
                  implementation={path: file_hash(path) for path in
                                  ("agent.py", "context_managers/base.py", "context_managers/__init__.py")})
    try:
        common["openai_sdk_version"] = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        common["openai_sdk_version"] = None
    methods = {}
    for condition in sorted(conditions):
        path = "context_managers/" + METHOD_FILES[condition]
        defaults = manager_defaults(root, condition)
        supplied = manager_settings(args, condition)
        if condition == "A" and supplied:
            raise ValueError("Condition A accepts no context-manager settings")
        if condition != "A" and any(key not in defaults for key in supplied):
            raise ValueError(f"Unknown settings for condition {condition}: {sorted(set(supplied) - set(defaults))}")
        methods[condition] = {"supplied_settings": supplied, "resolved_defaults": defaults,
                              "effective_settings": {**defaults, **supplied},
                              "implementation_sha256": file_hash(path)}
    return {"common": common, "methods": methods}


def index_identity(directory):
    """Describe the existing local index without loading or downloading it."""
    root = Path(directory)
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []
    return {"files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                       "mtime_ns": p.stat().st_mtime_ns} for p in files],
            "identity_kind": "local file metadata; not a corpus content hash"}


def code_revision(root):
    """Return the local Git revision for provenance."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pricing_from_args(args):
    """Read optional USD-per-million rates, retaining explicit zeros."""
    pricing = {key: getattr(args, key, None) for key in
               ("input_per_million", "cached_per_million", "output_per_million")}
    for value in pricing.values():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Pricing rates must be finite and nonnegative")
    return pricing


def estimate_cost(input_tokens, output_tokens, cached_tokens, pricing):
    """Calculate USD cost only when usage and all required rates are known."""
    if input_tokens is None or output_tokens is None or any(v is None for v in pricing.values()):
        return None
    if cached_tokens is None:
        if pricing["input_per_million"] != pricing["cached_per_million"]:
            return None
        cached_tokens = 0
    if cached_tokens > input_tokens:
        raise ValueError("Cached usage exceeds total input usage")
    return ((input_tokens - cached_tokens) * pricing["input_per_million"]
            + cached_tokens * pricing["cached_per_million"]
            + output_tokens * pricing["output_per_million"]) / 1_000_000


def read_records(path):
    """Read unique episode IDs from JSONL, failing on malformed or duplicate data."""
    records = {}
    path = Path(path)
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                key = str(row["id"])
                if key in records:
                    raise ValueError(f"Duplicate ID {key} in {path}:{number}")
                records[key] = row
    return records


def validate_resume(manifest, expected, records_by_condition):
    """Require matching research and selection metadata before appending results."""
    if manifest.get("schema_version") != 2:
        raise ValueError("Legacy manifest has unknown configuration; use a new output directory")
    if len(expected["selected_ids"]) != len(set(expected["selected_ids"])):
        raise ValueError("Question selection contains duplicate IDs")
    for key in ("dataset_sha256", "seed", "selected_ids", "excluded_ids", "research_config"):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"Resume configuration differs: {key}")
    selected = set(expected["selected_ids"])
    for condition, records in records_by_condition.items():
        config_id = fingerprint({"common": expected["research_config"]["common"],
                                 "method": expected["research_config"]["methods"][condition]})
        for qid, row in records.items():
            if qid not in selected or row.get("condition") != condition or row.get("config_id") != config_id:
                raise ValueError(f"Incompatible saved episode: {condition}/{qid}")
