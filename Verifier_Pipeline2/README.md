# Process Verifier Pipeline 2.0 (`Verifier_Pipeline2`)

A clean, object-oriented framework for generating and auditing executable Python process verification traces for Multimodal Chain-of-Thought (CoT) reasoning.

---

## 🏗️ Core Architecture & Philosophy

1. **Process Verification (No Answer Leakage):**
   * The generated Python verification code inspects visual grounding step-by-step (`img.find()`, `patch.verify_property()`, spatial math).
   * It **never** asserts `proposed_answer == ground_truth` inside the verification code. The ground-truth answer is only used externally by the pipeline as a quality gate to ensure the SFT training dataset contains valid, trustworthy traces.

2. **Unified `ImagePatch` API (ViperGPT-inspired):**
   * Every crop and image is an `ImagePatch` object with:
     * Detection: `patch.find("chair")`
     * Probing: `patch.verify_property("chair", "blue")`, `patch.query("What color?")`
     * OCR: `patch.read_text()`
     * Spatial: `patch.horizontal_center`, `patch.is_to_right_of(other)`, `patch.distance(other)`

3. **Structured Verdicts:**
   * Execution returns a structured result dictionary:
     * `verdict`: `"VALID"` or `"INVALID"`
     * `failed_step`: `int` or `None`
     * `passed_count`: Number of verified visual claims
     * `completion_ratio`: Dense process reward signal ($\in [0.0, 1.0]$) for GRPO.

---

## 📁 Directory Structure

```
Verifier_Pipeline2/
├── config.yaml                     # Pipeline configuration
├── run_pipeline.py                 # Main batch generation and execution orchestrator
├── tools/                          # Core vision tool wrappers
│   ├── image_patch.py              # Unified ImagePatch class
│   ├── detector.py                 # OWLv2 object detector (lazy-loaded)
│   ├── probe.py                    # Qwen2-VL-2B visual probe (lazy-loaded)
│   ├── ocr.py                      # EasyOCR text reader (lazy-loaded)
│   └── spatial.py                  # Pure Python geometry & bounding box math
├── sandbox/
│   └── executor.py                 # Safe Python execution runner with error catching
├── generator/
│   ├── prompts.py                  # System prompts, API specs, and few-shot examples
│   └── trace_generator.py          # Gemini API code generator client
├── tests/
│   ├── test_tools.py               # Unit tests for OWLv2 and spatial math
│   └── test_executor.py            # Unit tests for sandbox verdict execution
└── data/
    ├── raw/                        # GQA benchmark dataset and images (symlinked)
    └── sft/
        ├── golden_traces.jsonl     # High-quality verified traces for SFT training
        └── rejected_traces.jsonl   # Failed/hallucinated traces (negatives for PRM/DPO)
```

---

## 🚀 Quickstart

### 1. Run Unit Tests
Verify that OWLv2, spatial geometry, and the sandbox executor function properly on your GPU:

```bash
python tests/test_tools.py
python tests/test_executor.py
```

### 2. Run Trace Generation Pipeline
Set your Google AI Studio key and run:

```bash
export GEMINI_API_KEY="your-api-key-here"
python run_pipeline.py --num_samples 5
```

Or pass it directly via CLI:
```bash
python run_pipeline.py --num_samples 5 --api_key "your-api-key-here"
```
