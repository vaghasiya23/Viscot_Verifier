"""Exercise scaffold + real vision tools without spending Gemini quota."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from generation.skeleton_gen import build_scaffold, canonical_code

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('num_samples must be positive')
    from filters import execution_filter, outcome_filter
    samples = json.loads((ROOT/'data/raw/gqa_300_samples.json').read_text())[:args.num_samples]
    path = ROOT/'data/processed'/('diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    results = []
    for i,sample in enumerate(samples):
        scaffold = build_scaffold(sample['reasoning'],sample)
        print(f"CHECK {i+1}/{len(samples)}: {sample['question']}",flush=True)
        if any(s['needs_review'] for s in scaffold):
            row = {'ok':False,'error':'Unsupported scaffold'}
        else:
            result = execution_filter.run(canonical_code(scaffold), str(ROOT/'data/raw/images'/sample['image']))
            verdict = outcome_filter.check(result,sample,'VALID')
            row = {'ok':verdict['status']=='valid','error':result.get('error') or verdict['reason'],
                   'evidence':result.get('evidence',[]),'answer':result['local_scope'].get('final_answer')}
        row.update(image=sample['image'],question=sample['question'],target=sample['answer'])
        results.append(row)
        print('PASS' if row['ok'] else 'DEFER: '+row['error'],flush=True)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(results,indent=2))
    print(f"Real-tool diagnostic: {sum(x['ok'] for x in results)}/{len(results)} passed. {path}")


if __name__ == '__main__':
    main()
