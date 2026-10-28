import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from agent import (BASE_TOOLS, SYSTEM, ResearchAgent, bootstrap_model, create_client,
                   load_searcher, load_questions)
from evaluate import evaluate
from evaluation.diagnostics import DiagnosticStore
from evaluation.experiment import (code_revision, fingerprint, pricing_from_args, read_records,
                                   research_configuration, validate_resume)

MODES = [('A', 'baseline'), ('B', 'memgpt'), ('C', 'llmlingua'), ('D', 'srcm')]

def read_completed(path):
    """Read completed episode IDs while rejecting duplicate records."""
    return read_records(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions', default='data/questions_830.jsonl')
    p.add_argument('--count', type=int, default=120)
    p.add_argument('--exclude-ids', default=None, help='Optional file containing pilot IDs to exclude')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='experiments/main_seed42')
    p.add_argument('--conditions', nargs='+', choices=list('ABCD'), default=list('ABCD'))
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--snippet-chars', type=int, default=400)
    p.add_argument('--document-chars', type=int, default=12000)
    p.add_argument('--api-retries', type=int, default=3)
    p.add_argument('--max-output-tokens', type=int, default=2048)
    p.add_argument('--cm-args', default='{}')
    p.add_argument('--condition-cm-args', default='{}')
    p.add_argument('--model', default=None)
    p.add_argument('--index-dir', default='indexes/bm25')
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--ids', default=None, help='Use an existing fixed ID list instead of sampling')
    p.add_argument('--preflight', action='store_true', help='Inspect configuration and local resources without writes or requests')
    p.add_argument('--input-per-million', type=float, default=None)
    p.add_argument('--cached-per-million', type=float, default=None)
    p.add_argument('--output-per-million', type=float, default=None)
    p.add_argument('--max-budget-usd', type=float, default=None, help='Stop between questions when reported estimated spending reaches this amount')
    p.add_argument('--judge-model', default=None, help='Optional paid judge; omitted = retain answers for later grading')
    p.add_argument('--save-history', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Only select and save questions, no API calls')
    a = p.parse_args()
    bootstrap_model(a)
    pricing = pricing_from_args(a)
    if a.count < 1:
        p.error('--count must be positive')
    if a.max_budget_usd is not None and (a.max_budget_usd <= 0 or any(v is None for v in pricing.values())):
        p.error('--max-budget-usd requires a positive cap and all three pricing rates')
    source = Path(a.questions)
    if not source.exists():
        p.error(f'Missing {source}. First run python prepare_questions.py, or supply a question JSONL file.')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    excluded = set()
    if a.exclude_ids:
        excluded = {line.strip() for line in Path(a.exclude_ids).read_text(encoding='utf-8').splitlines() if line.strip()}
    questions = load_questions(str(source), None, exclude_ids=excluded)
    ids = [q['id'] for q in questions]
    if len(ids) != len(set(ids)):
        p.error('Dataset has duplicate question IDs')
    if a.count > len(questions):
        p.error(f'Asked for {a.count} questions but only {len(questions)} available')
    out = Path(a.out_dir)
    manifest_path = out / 'manifest.json'
    root = Path(__file__).parent
    # Freeze all method configurations so condition subsets can share one run directory.
    research_config = research_configuration(a, root, SYSTEM, BASE_TOOLS, [c for c, _ in MODES])
    a.research_config = research_config
    a.code_revision = code_revision(root)
    fixed_ids = Path(a.ids).read_text(encoding='utf-8').splitlines() if a.ids else None
    if fixed_ids is not None:
        fixed_ids = [qid.strip() for qid in fixed_ids if qid.strip()]
        if len(fixed_ids) != a.count or len(set(fixed_ids)) != len(fixed_ids) or not set(fixed_ids) <= set(ids):
            p.error('--ids must contain exactly --count unique, available, non-excluded IDs')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        chosen = manifest['selected_ids']
        if len(chosen) != a.count or (fixed_ids is not None and chosen != fixed_ids):
            p.error('Existing question selection differs; select a new output directory')
    else:
        shuffled = list(ids)
        random.Random(a.seed).shuffle(shuffled)
        chosen = fixed_ids if fixed_ids is not None else shuffled[:a.count]
        manifest = {'schema_version': 2, 'dataset_sha256': digest, 'seed': a.seed, 'selected_count': a.count,
                    'selected_ids': chosen, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'source': str(source), 'excluded_ids': sorted(excluded),
                    'research_config': research_config, 'code_revision': a.code_revision,
                    'pricing': pricing}
    expected = {'dataset_sha256': digest, 'seed': a.seed, 'selected_ids': chosen,
                'excluded_ids': sorted(excluded), 'research_config': research_config}
    existing = {c: read_completed(out / f'{c}.jsonl') for c, _ in MODES}
    if not manifest_path.exists() and any(existing.values()):
        p.error('Existing episodes lack a manifest; select a new output directory')
    if manifest_path.exists():
        validate_resume(manifest, expected, existing)
    by_id = {q['id']: q for q in questions}
    selected = [by_id[i] for i in chosen]
    if a.preflight:
        print(json.dumps({'model': a.model, 'model_source': a.model_source,
                          'questions': chosen, 'configuration': research_config, 'pricing': pricing,
                          'index_present': any(Path(a.index_dir).rglob('segments_*'))}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not (out / 'selected_ids.txt').exists():
        (out / 'selected_ids.txt').write_text('\n'.join(chosen) + '\n', encoding='utf-8')
    print(f'Selected {len(chosen)} of {len(questions)} questions, seed={a.seed}. IDs saved to {out}/selected_ids.txt')
    if a.dry_run:
        print('Dry run: no model requests made.')
        return
    import os
    if not os.getenv('OPENROUTER_API_KEY'):
        p.error('Missing OPENROUTER_API_KEY')
    searcher = load_searcher(a.index_dir)
    client = create_client()
    a.seed = None
    a.resume = True
    previous = [r for records in existing.values() for r in records.values()]
    if a.max_budget_usd is not None and any((r.get('usage') or {}).get('estimated_cost_usd') is None for r in previous):
        p.error('Existing spending is unknown; a spending-capped resume requires complete costs')
    total_cost = sum((r.get('usage') or {}).get('estimated_cost_usd') or 0 for r in previous)
    for condition, mode in MODES:
        if condition not in a.conditions:
            continue
        a.mode = mode
        a.config_id = fingerprint({'common': research_config['common'], 'method': research_config['methods'][condition]})
        diagnostics = DiagnosticStore(out / '.diagnostics' if a.diagnostics or a.save_history else None)
        agent = ResearchAgent(searcher, client, condition, a, diagnostics=diagnostics)
        path = out / f'{condition}.jsonl'
        completed = read_completed(path)

        with path.open('a', encoding='utf-8') as f:
            for i, task in enumerate(selected, 1):
                if task['id'] in completed:
                    continue
                if a.max_budget_usd is not None and total_cost >= a.max_budget_usd:
                    print('Estimated spending cap reached; stopping before next question.')
                    return
                print(f'[{condition} {i}/{len(selected)}] {task["id"]}', flush=True)
                result = agent.run(task)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                cost = result['usage']['estimated_cost_usd']
                if cost is not None:
                    total_cost += cost
                print(' ', result['stop_reason'], 'usage:', result['usage'])
                if a.max_budget_usd is not None and cost is None:
                    print('Spending measurement is unknown; stopped after preserving the episode.')
                    return
    if all(set(read_completed(out / f'{c}.jsonl')) == set(chosen) for c in a.conditions):
        if a.judge_model:
            evaluate(out, a.judge_model, a.conditions)
    else:
        print('Some conditions incomplete. Re-run with same --out-dir to resume.')

if __name__ == '__main__':
    main()
