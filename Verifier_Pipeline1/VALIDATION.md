The Stage A pipeline now uses the `strict-v2` acceptance contract.

Gemini must copy the scaffold's executable statements; comments and formatting
may differ. Python AST comparison runs BEFORE execution. This deliberately
rejects equivalent rewrites as well as weaker predicates, skipped checks,
extra assignments, tool overrides and detection-only fallbacks. Gemini's raw
response is retained as `source_trace`; the accepted SFT target uses the
scaffold's claims and the runtime's verdict, not invented execution narration.

GQA relation role `s` makes the new entity the subject; `o` makes it the object.
Semantic relations receive both detected boxes, marked red (subject) and blue
(object) in the full image. `on`, `in`, and depth are visually probed rather
than approximated by overlap. The answer comes from an open-ended visual query
without the target answer in that query. Only exact normalized answers and a
small explicit synonym list match. Broad categories and substrings do not.

Multiple boxes are deduplicated by confidence-ordered IoU suppression. Relations
are evaluated against all reference candidates rather than requiring exactly
one detection. Multiple answer boxes are accepted only if their visual answers
agree under the explicit answer-matching rules. Self-pairs cannot verify a
semantic relation.

For VisCoT GQA samples with `bboxs`, the target entity uses those annotated
regions as explicit SFT localization supervision. The executable target contains
the coordinates in an `annotated_regions` call, whose evidence record identifies
their source. It validates coordinates, not object identity. Relations and the
open-ended answer still require visual evidence. Accepted rows carry
`localization_source: viscot_annotation`. These are annotation-assisted training
targets, not autonomous detection benchmarks; a deployed model must predict its
own localization when annotations are unavailable.

For unnamed (`_`) and abstract intermediate entities without annotation support, a full-image visual question
proposes a concrete detection label without access to the target answer. The
proposed object is then detected and its relation independently checked. This
proposal can still be wrong or incomplete; it does not certify uniqueness.
Horizontal/vertical position filters use image halves. Attribute filters,
including negation, receive explicit visual checks.

Unsupported operations, missing dependencies, conflicting answers, uncertain
visual responses and failed relations are deferred. At implementation time
290/297 source programs were supported; this is compile coverage, NOT accuracy
or the number of golden outputs. The same deterministic execution failure is
not retried through Gemini. Incorrect code/format can still receive another
candidate attempt. A generated INVALID verdict is not a verified negative.

Run in the existing ai environment with GEMINI_API_KEY exported:

```bash
python -u pipeline.py --num_samples 10 --num_candidates 2 2>&1 | tee strict_run.log
python -B -m unittest discover -s tests -v
```

Each accepted row includes `validation_version`, generator model and recorded
tool calls/arguments/results. Each completed run writes `audit_report.json`,
including a reproducible random sample of up to 10 accepted traces, local image
paths and evidence. For an existing run or a larger review sample:

```bash
python audit_batch.py data/processed/YOUR_RUN_DIRECTORY --sample_size 15 --seed 42
```

Inspect the selected images alongside claim direction, referent boxes and tool
answers. Record errors / reviewed as a sample error rate; investigate any
systematic issue across the batch before training. Ten examples are a small
spot check, not statistical certification. Old outputs are never upgraded by
an offline audit; regenerate them under the new checks.

Limits: AST conformance establishes adherence to the program, not visual truth.
OWLv2 and Qwen2-VL can still be wrong. Broad-category detections may fail; multiple
boxes with conflicting answers are deferred. Original
CoT is retained as input but its free-text claims are not independently parsed;
the verified target covers the dataset reasoning program. The audit JSON is for
human review, not an automatic visual correctness score. Real-image diagnostics can be run without Gemini quota using:

```bash
python -u validate_samples.py --num_samples 10
```

These diagnostics exercise the canonical scaffold and real vision backends;
they do not generate SFT targets or measure visual truth against human review.

The saved real-tool diagnostic `data/processed/diagnostic_20260907_042233.json`
completed all 10 initial samples: 7 passed, 3 disagreed with the target answer.
This used the real GPU vision backends and canonical programs, with no Gemini
calls. It measures execution/answer agreement, not human-verified accuracy.
