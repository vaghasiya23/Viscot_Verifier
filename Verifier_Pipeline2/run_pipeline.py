"""
run_pipeline.py - Orchestrator for Process Verifier SFT trace generation & execution.
"""
import os
import sys
import json
import argparse
import yaml
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from generator import GeminiTraceGenerator
from sandbox import execute_verifier


def load_config():
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def append_jsonl(record: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def run(num_samples: int = 5, start_idx: int = 0, model_name: str = None, api_key: str = None, mock: bool = False):
    config = load_config()
    model = model_name or config.get("models", {}).get("generator", "gemini-2.5-flash")
    
    # Initialize Generator
    try:
        generator = GeminiTraceGenerator(model_name=model, api_key=api_key, mock=mock)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("Please export GEMINI_API_KEY in your terminal, pass --api_key YOUR_KEY, or use --mock")
        sys.exit(1)

    # Load samples
    data_path = os.path.join(ROOT, config["paths"]["raw_data"])
    with open(data_path, "r") as f:
        samples = json.load(f)

    selected_samples = samples[start_idx: start_idx + num_samples]
    images_dir = os.path.join(ROOT, config["paths"]["images_dir"])
    golden_path = os.path.join(ROOT, config["paths"]["sft_golden"])
    rejected_path = os.path.join(ROOT, config["paths"]["sft_rejected"])

    print(f"\n{'='*65}")
    print(f"🚀 Process Verifier Pipeline 2.0: Starting Generation")
    print(f"Total samples to process: {len(selected_samples)} (start={start_idx})")
    print(f"Generator Model: {model}")
    print(f"Golden output: {golden_path}")
    print(f"{'='*65}\n")

    stats = {"total": len(selected_samples), "golden": 0, "rejected": 0, "error": 0}

    for i, sample in enumerate(selected_samples):
        idx = start_idx + i
        question = sample.get("question", "")
        thought = sample.get("thought", "")
        target_answer = sample.get("answer", "")
        image_name = sample.get("image", "")
        image_path = os.path.join(images_dir, image_name)

        print(f"[{i+1}/{len(selected_samples)}] Sample #{idx} | Image: {image_name}")
        print(f"  Q: {question}")
        print(f"  A: {target_answer}")

        if not os.path.exists(image_path):
            print(f"  ⚠️ Image not found: {image_path}. Skipping.")
            continue

        # 1. Generate Verification Code
        print("  Generating verification code...")
        try:
            code = generator.generate(question=question, thought=thought, answer=target_answer)
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            stats["error"] += 1
            continue

        # 2. Execute Code in Sandbox
        print("  Executing code in sandbox...")
        exec_result = execute_verifier(code, image_path)
        verdict = exec_result.get("verdict", "ERROR")
        ratio = exec_result.get("completion_ratio", 0.0)
        print(f"  Verdict: {verdict} | Process Completion Ratio: {ratio*100:.1f}%")

        record = {
            "sample_index": idx,
            "question": question,
            "thought": thought,
            "answer": target_answer,
            "image": image_name,
            "verification_code": code,
            "execution_result": exec_result,
            "timestamp": datetime.now().isoformat(),
        }

        # 3. Quality Gating
        if verdict == "VALID":
            append_jsonl(record, golden_path)
            stats["golden"] += 1
            print("  ✅ Saved to GOLDEN traces.")
        else:
            append_jsonl(record, rejected_path)
            stats["rejected"] += 1
            print(f"  ⚠️ Saved to REJECTED traces (Reason: {exec_result.get('error', 'Step failure')})")

        print("-" * 60)

    print(f"\n{'='*65}")
    print(f"🎉 Pipeline Run Completed!")
    print(f"Stats: Total={stats['total']} | Golden={stats['golden']} | Rejected={stats['rejected']} | Errors={stats['error']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Process Verifier Trace Generation Pipeline")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting sample index")
    parser.add_argument("--model", type=str, default=None, help="Gemini model name")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API Key")
    parser.add_argument("--mock", action="store_true", help="Run with mock generator for offline/dry-run testing")
    args = parser.parse_args()

    run(
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        model_name=args.model,
        api_key=args.api_key,
        mock=args.mock,
    )
