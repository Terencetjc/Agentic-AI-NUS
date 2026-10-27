import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from agent import ResearchAgent, load_searcher, load_questions
from evaluate import evaluate

MODES = [('A', 'baseline'), ('B', 'memgpt'), ('C', 'llmlingua'), ('D', 'srcm')]

def read_completed(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                records[str(row['id'])] = row
    return records

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
    p.add_argument('--input-per-million', type=float, default=0.0)
    p.add_argument('--cached-per-million', type=float, default=0.0)
    p.add_argument('--output-per-million', type=float, default=0.0)
    p.add_argument('--max-budget-usd', type=float, default=None, help='Stop between questions when reported estimated spending reaches this amount')
    p.add_argument('--judge-model', default=None, help='Optional paid judge; omitted = strict exact-match proxy')
    p.add_argument('--save-history', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Only select and save questions, no API calls')
    a = p.parse_args()
    if a.count < 1:
        p.error('--count must be positive')
    if a.max_budget_usd is not None and not any((a.input_per_million, a.cached_per_million, a.output_per_million)):
        p.error('--max-budget-usd requires explicit nonzero pricing flags')
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
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest['dataset_sha256'] != digest or manifest['seed'] != a.seed or
                manifest['selected_count'] != a.count or
                manifest.get('excluded_ids', []) != sorted(excluded)):
            p.error('Existing manifest differs in dataset/seed/count. Choose a new --out-dir.')
        chosen = manifest['selected_ids']
    else:
        shuffled = list(ids)
        random.Random(a.seed).shuffle(shuffled)
        chosen = shuffled[:a.count]
        manifest = {'dataset_sha256': digest, 'seed': a.seed, 'selected_count': a.count,
                    'selected_ids': chosen, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'source': str(source), 'excluded_ids': sorted(excluded)}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    by_id = {q['id']: q for q in questions}
    selected = [by_id[i] for i in chosen]
    (out / 'selected_ids.txt').write_text('\n'.join(chosen) + '\n', encoding='utf-8')
    print(f'Selected {len(chosen)} of {len(questions)} questions, seed={a.seed}. IDs saved to {out}/selected_ids.txt')
    if a.dry_run:
        print('Dry run: no model requests made.')
        return
    from dotenv import load_dotenv
    from openai import OpenAI
    import os
    load_dotenv()
    if not os.getenv('OPENROUTER_API_KEY'):
        p.error('Missing OPENROUTER_API_KEY')
    searcher = load_searcher()
    client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'])
    a.seed = None
    a.resume = True
    total_cost = 0.0
    for condition, mode in MODES:
        if condition not in a.conditions:
            continue
        a.mode = mode
        agent = ResearchAgent(searcher, client, condition, a)
        path = out / f'{condition}.jsonl'
        completed = read_completed(path)

        total_cost += sum((r.get('usage') or {}).get('estimated_cost_usd') or 0 for r in completed.values())
        with path.open('a', encoding='utf-8') as f:
            for i, task in enumerate(selected, 1):
                if task['id'] in completed:
                    continue
                if a.max_budget_usd is not None and total_cost >= a.max_budget_usd:
                    print('Estimated spending cap reached; stopping before next question.')
                    return
                print(f'[{condition} {i}/{len(selected)}] {task["id"]}', flush=True)
                result = agent.run(task)
                if not any((a.input_per_million, a.cached_per_million, a.output_per_million)):
                    result['usage']['estimated_cost_usd'] = None
                else:
                    total_cost += result['usage']['estimated_cost_usd'] or 0
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                print(' ', result['stop_reason'], 'tokens:', (result['usage']['input_tokens'] + result['usage']['output_tokens']))
    if all(set(read_completed(out / f'{c}.jsonl')) == set(chosen) for c in a.conditions):
        if a.judge_model:
            evaluate(out, a.judge_model)
    else:
        print('Some conditions incomplete. Re-run with same --out-dir to resume.')

if __name__ == '__main__':
    main()
