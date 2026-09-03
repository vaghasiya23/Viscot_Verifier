"""
pipeline.py — Stage A orchestrator.

Loads N samples, runs best-of-K trace generation for each, passes every
candidate through all 4 filters, saves results into golden/trash/ambiguous.

Usage:
    python pipeline.py                          # runs on first 5 samples (stage A default)
    python pipeline.py --num_samples 3          # run on first 3 samples
    python pipeline.py --num_candidates 3       # 3 attempts per sample instead of 5
"""
import json
import os
import sys
import argparse
import yaml

# Make sure the project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from generation.skeleton_gen import build_scaffold, build_generation_prompt
from generation.trace_gen import CoderModel, generate_candidate, extract_code_block, extract_verdict
from filters import execution_filter, outcome_filter, coverage_filter, arg_validity_filter


def load_config():
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_samples(config, num_samples):
    data_path = os.path.join(ROOT, "data", "raw", "gqa_300_samples.json")
    with open(data_path, "r") as f:
        samples = json.load(f)
    return samples[:num_samples]


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  Saved {len(data)} entries to {filepath}")


def run_pipeline(num_samples=5, num_candidates=5):
    config = load_config()
    samples = load_samples(config, num_samples)

    print(f"\n{'='*60}")
    print(f"Stage A Pipeline — {len(samples)} samples, {num_candidates} candidates each")
    print(f"{'='*60}\n")

    # Load the coder model once
    coder = CoderModel(model_name=config["models"]["coder"])

    # Output buckets
    golden = []
    trash = []
    ambiguous = []

    img_dir = os.path.join(ROOT, "data", "raw", "images")

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)}: {sample['image']} ---")
        print(f"  Q: {sample['question']}")
        print(f"  A: {sample.get('answer', '?')}")

        # Check if reasoning field exists (needed for scaffold)
        if "reasoning" not in sample or not sample["reasoning"]:
            print("  SKIP: No 'reasoning' field in this sample.")
            trash.append({
                "image": sample["image"],
                "reason": "No reasoning field",
            })
            continue

        # Build the deterministic scaffold once per sample
        scaffold = build_scaffold(sample["reasoning"])
        print(f"  Scaffold: {len(scaffold)} steps")

        # Resolve full image path
        img_path = os.path.join(img_dir, sample["image"])
        if not os.path.exists(img_path):
            print(f"  SKIP: Image not found at {img_path}")
            trash.append({
                "image": sample["image"],
                "reason": f"Image not found: {img_path}",
            })
            continue

        valid_candidates = []

        for attempt in range(num_candidates):
            print(f"  Attempt {attempt+1}/{num_candidates}...")

            # --- GENERATE ---
            candidate = generate_candidate(sample, coder,
                                           temperature=config["generation"]["temperature"])
            code = candidate["code"]
            verdict_text = candidate["verdict_text"]

            if not code:
                print(f"    => FAIL: No python code block generated")
                trash.append({
                    "image": sample["image"],
                    "attempt": attempt+1,
                    "trace": candidate["raw_trace"],
                    "filter": "no_code",
                    "error": "No python code block found in trace",
                })
                continue

            # --- FILTER 1: Arg Validity (cheap, no execution needed) ---
            arg_result = arg_validity_filter.check(code)
            if not arg_result["passed"]:
                print(f"    => FAIL [arg_validity]: {arg_result['reason']}")
                for v in arg_result["violations"]:
                    print(f"       - {v}")
                trash.append({
                    "image": sample["image"],
                    "attempt": attempt+1,
                    "trace": candidate["raw_trace"],
                    "filter": "arg_validity",
                    "error": arg_result["reason"],
                    "violations": arg_result["violations"],
                })
                continue

            # --- FILTER 2: Execution (runs the code for real) ---
            exec_result = execution_filter.run(code, img_path)
            if not exec_result["ok"]:
                error = exec_result["error"]
                print(f"    => FAIL [execution]: {error}")

                # Check if it's an ambiguity assertion
                if exec_result.get("is_assertion") and ("ambiguous" in error.lower() or "multiple" in error.lower()):
                    ambiguous.append({
                        "image": sample["image"],
                        "attempt": attempt+1,
                        "trace": candidate["raw_trace"],
                        "error": error,
                    })
                else:
                    trash.append({
                        "image": sample["image"],
                        "attempt": attempt+1,
                        "trace": candidate["raw_trace"],
                        "filter": "execution",
                        "error": error,
                    })
                continue

            # --- FILTER 3: Coverage (did the code touch all scaffold steps?) ---
            cov_result = coverage_filter.check(code, scaffold)
            if not cov_result["passed"]:
                print(f"    => FAIL [coverage]: {cov_result['reason']}")
                trash.append({
                    "image": sample["image"],
                    "attempt": attempt+1,
                    "trace": candidate["raw_trace"],
                    "filter": "coverage",
                    "error": cov_result["reason"],
                })
                continue

            # --- FILTER 4: Outcome (does derived answer match ground truth?) ---
            out_result = outcome_filter.check(exec_result, sample, verdict_text)
            if not out_result["passed"]:
                print(f"    => FAIL [outcome]: {out_result['reason']}")
                trash.append({
                    "image": sample["image"],
                    "attempt": attempt+1,
                    "trace": candidate["raw_trace"],
                    "filter": "outcome",
                    "error": out_result["reason"],
                })
                continue

            # --- ALL FILTERS PASSED ---
            print(f"    => PASS (all 4 filters)")
            valid_candidates.append({
                "trace": candidate["raw_trace"],
                "code": code,
                "verdict": verdict_text,
                "coverage": cov_result["coverage"],
                "code_len": len(code),
            })

            # Early stop: if we have 2 valid candidates, no need to keep trying
            if len(valid_candidates) >= 2:
                break

        # --- SELECT BEST ---
        if valid_candidates:
            # Sort by: highest coverage first, then shortest code as tiebreak
            best = sorted(valid_candidates, key=lambda x: (-x["coverage"], x["code_len"]))[0]
            golden.append({
                "image": sample["image"],
                "question": sample["question"],
                "answer": sample.get("answer", ""),
                "input_cot": sample["thought"],
                "golden_target": best["trace"],
                "verdict": best["verdict"],
                "num_valid_of_total": f"{len(valid_candidates)}/{num_candidates}",
            })
            print(f"  => SAVED TO GOLDEN ({len(valid_candidates)} valid out of {num_candidates} attempts)")
        else:
            print(f"  => EXHAUSTED ALL {num_candidates} ATTEMPTS. No golden trace.")

    # --- SAVE ---
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Golden: {len(golden)}")
    print(f"  Trash:  {len(trash)}")
    print(f"  Ambiguous: {len(ambiguous)}")
    print(f"{'='*60}")

    out_dir = os.path.join(ROOT, "data", "processed")
    save_json(golden, os.path.join(out_dir, "golden", "golden_traces.json"))
    save_json(trash, os.path.join(out_dir, "trash", "trash_traces.json"))
    save_json(ambiguous, os.path.join(out_dir, "ambiguous", "ambiguous_traces.json"))

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage A Pipeline")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process")
    parser.add_argument("--num_candidates", type=int, default=5, help="Attempts per sample")
    args = parser.parse_args()

    run_pipeline(num_samples=args.num_samples, num_candidates=args.num_candidates)