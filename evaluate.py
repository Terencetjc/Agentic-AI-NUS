import argparse
import json
import os
import re
import statistics
from pathlib import Path

CONDITIONS = ('A', 'B', 'C', 'D')
GRADER_PROMPT = '''Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: {response}
[correct_answer]: {correct_answer}
Your judgement must be in the format and criteria specified below:
extracted_final_answer: Extract the final answer from [response], or None if absent.
reasoning: Compare only the extracted answer to [correct_answer]. Do not independently solve the question.
correct: Answer yes if the extracted answer matches the correct answer, including semantically equivalent wording or a small margin of error for numerical problems; otherwise no.
confidence: Your confidence as a percentage.'''

def load_results(folder):
    runs = {}
    for c in CONDITIONS:
        path = folder / f'{c}.jsonl'
        if not path.exists():
            continue
        records = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                key = str(row['id'])
                if key in records:
                    raise ValueError(f'Duplicate ID {key} in {path}')
                records[key] = row
        runs[c] = records
    return runs

def grade(row, judge_client=None, judge_model=None):
    if row.get('stop_reason') != 'submitted' or not row.get('answer'):
        return False, 'not_submitted', 0.0
    reference = row.get('reference_answer')
    if reference is None or not str(reference).strip():
        return None, 'no_reference', 0.0
    response = judge_client.chat.completions.create(
        model=judge_model,
        messages=[{'role': 'user', 'content': GRADER_PROMPT.format(question=row['question'], response=row['answer'], correct_answer=reference)}],
        temperature=0, max_tokens=500)
    verdict = response.choices[0].message.content or ''
    m = re.search(r'(?im)^correct\s*:\s*(yes|no)\b', verdict)
    if not m:
        return None, 'judge_unparseable', 0.0
    return m.group(1).lower() == 'yes', 'llm_judge_nonofficial_settings', 0.0

def evaluate(folder, judge_model):
    runs = load_results(folder)
    if not runs:
        raise ValueError('No A.jsonl/B.jsonl/C.jsonl/D.jsonl files found')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    ids = manifest['selected_ids']
    expected = set(ids)
    for c, rows in runs.items():
        if set(rows) != expected:
            missing = sorted(expected - set(rows))
            extra = sorted(set(rows) - expected)
            raise ValueError(f'{c} has mismatched IDs; missing={missing[:5]}, extra={extra[:5]}')
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError('Set OPENROUTER_API_KEY in .env to use semantic grading')
    judge_client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'])

    grades_path = folder / 'grades_judge.jsonl'
    grades = {}
    if grades_path.exists():
        for line in grades_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                g = json.loads(line)
                if g.get('judge_model') == judge_model:
                    grades[(g['condition'], g['id'])] = g
    with grades_path.open('a', encoding='utf-8') as gf:
        for c, rows in runs.items():
            for qid in ids:
                if (c, qid) in grades:
                    continue
                result, method, _ = grade(rows[qid], judge_client, judge_model)
                g = {'condition': c, 'id': qid, 'correct': result, 'method': method, 'judge_model': judge_model}
                grades[(c, qid)] = g
                gf.write(json.dumps(g) + '\n')
                gf.flush()
    summary = {'seed': manifest['seed'], 'selected_count': len(ids), 'grading': 'llm_judge_nonofficial_settings', 'conditions': {}}
    for c, rows in runs.items():
        items = [rows[i] for i in ids]
        vals = [grades[(c, i)]['correct'] for i in ids]
        graded = [v for v in vals if v is not None]
        usages = [r.get('usage') or {} for r in items]
        totals = [int(u.get('input_tokens') or 0) + int(u.get('output_tokens') or 0) for u in usages]
        costs = [u.get('estimated_cost_usd') for u in usages]
        n_correct = sum(v is True for v in vals)
        n = len(ids)
        summary['conditions'][c] = {
            'questions': n, 'graded': len(graded), 'correct': n_correct,
            'accuracy': (n_correct / n if len(graded) == n else None),
            'total_tokens': sum(totals), 'avg_tokens': statistics.mean(totals),
            'input_tokens': sum(int(u.get('input_tokens') or 0) for u in usages),
            'output_tokens': sum(int(u.get('output_tokens') or 0) for u in usages),
            'cached_input_tokens': sum(int(u.get('cached_input_tokens') or 0) for u in usages),
            'api_calls': sum(int(u.get('api_calls') or 0) for u in usages),
            'avg_wall_seconds': statistics.mean(float(r.get('total_seconds') or 0) for r in items),
            'total_cost_usd': (sum(costs) if all(x is not None for x in costs) else None),
            'cost_per_correct_usd': (sum(costs) / n_correct if n_correct and all(x is not None for x in costs) else None),
            'stop_reasons': {k: sum(r.get('stop_reason') == k for r in items) for k in sorted({r.get('stop_reason') for r in items})},
        }
    baseline = summary['conditions'].get('A')
    if baseline:
        for c, s in summary['conditions'].items():
            s['token_reduction_vs_A'] = (1 - s['total_tokens'] / baseline['total_tokens']) if baseline['total_tokens'] else None
            s['accuracy_retention_vs_A'] = (s['accuracy'] / baseline['accuracy']) if s['accuracy'] is not None and baseline['accuracy'] else None
    out = folder / 'summary.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Summary:', out)
    for c, s in summary['conditions'].items():
        accuracy = 'N/A' if s['accuracy'] is None else f"{s['accuracy']:.1%}"
        print(f"{c}: accuracy={accuracy}, correct={s['correct']}/{s['questions']}, tokens={s['total_tokens']:,}, cost={s['total_cost_usd']}")
    print('WARNING: Semantic grading uses an LLM judge; official benchmark settings have not been verified.')
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--judge-model', required=True, help='OpenRouter model used for semantic grading')
    a = p.parse_args()
    evaluate(Path(a.run_dir), a.judge_model)

if __name__ == '__main__':
    main()
