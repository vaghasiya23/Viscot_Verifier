"""Offline structural audit and reproducible review sample; no API/GPU calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from generation.skeleton_gen import build_scaffold
from generation.trace_gen import extract_code_block
from filters.coverage_filter import check

ROOT = Path(__file__).resolve().parent


def audit_run(run_dir, sample_size=10, seed=42):
    run = Path(run_dir)
    rows = json.loads((run / 'golden/golden_traces.json').read_text())
    samples = json.loads((ROOT / 'data/raw/gqa_300_samples.json').read_text())
    by_key = {(s['image'], s['question']): s for s in samples}
    results = []
    for i, row in enumerate(rows):
        sample = by_key.get((row['image'], row['question']))
        reasons = []
        if sample is None:
            reasons.append('Source sample missing')
        else:
            result = check(extract_code_block(row['golden_target']), build_scaffold(sample['reasoning'], sample))
            if not result['passed']:
                reasons.append(result['reason'])
        if row.get('validation_version') != 'strict-v2' or not row.get('evidence'):
            reasons.append('Missing strict runtime evidence; regenerate, do not relabel as verified')
        results.append({'index': i, 'image': row['image'], 'question': row['question'], 'issues': reasons})
    chosen = sorted(random.Random(seed).sample(range(len(rows)), min(sample_size, len(rows))))
    report = {'golden_count': len(rows), 'flagged_count': sum(bool(x['issues']) for x in results),
              'seed': seed, 'structural_audit': results,
              'review_sample': [dict(results[i], target=rows[i]['golden_target'],
                                     evidence=rows[i].get('evidence', []),
                                     image_path=str(ROOT / 'data/raw/images' / rows[i]['image'])) for i in chosen],
              'review_instructions': 'Inspect each sampled image, claim direction, box binding, tool result and final answer. Record errors/reviewed as an estimated error rate; this report does not certify visual truth. If an error is found, investigate that pattern across the batch before SFT export.'}
    for bucket in ('trash', 'ambiguous'):
        path = run / bucket / f'{bucket}_traces.json'
        entries = json.loads(path.read_text()) if path.exists() else []
        report[bucket + '_reasons'] = dict(Counter(x.get('filter', 'execution_uncertain') for x in entries))
    target = run / 'audit_report.json'
    target.write_text(json.dumps(report, indent=2))
    print(f'Audit: {report["flagged_count"]}/{len(rows)} structurally flagged; {len(chosen)} selected for visual review. {target}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error('sample_size must be nonnegative')
    audit_run(args.run_dir, args.sample_size, args.seed)
