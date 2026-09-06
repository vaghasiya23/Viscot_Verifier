"""
Curated golden exemplars for few-shot in-context learning.
Provides concrete code and reasoning patterns for spatial, semantic, and attribute queries.
"""

FEWSHOT_EXAMPLES = [
    {
        "description": "Spatial Relation Verification (right of)",
        "full_trace": '''<thought>
The scaffold requires detecting a chair (e0), detecting candidate sofas and verifying that a sofa is to the right of the chair (e1), and stating the final answer (e2).
</thought>
<extract_claims>
1. There is a chair in the image.
2. There is a sofa located to the right of the chair.
3. The name of the furniture to the right of the chair is "sofa".
</extract_claims>
<tool_verification>
```python
# Step 0: Detect reference object (chair)
e0 = detect_and_crop(img_path, "chair")
assert len(e0) > 0, "No chair detected"
chair_box = e0[0]

# Step 1: Detect target object candidates (sofa) and verify spatial relation
e1_candidates = detect_and_crop(img_path, "sofa")
assert len(e1_candidates) > 0, "No sofa detected"

e1 = [s for s in e1_candidates if check_spatial_relation(s, chair_box, "right")]
assert len(e1) > 0, "No sofa found to the right of the chair"

# Step 2: Query name of the furniture
final_answer = "sofa"
```
</tool_verification>
<final_verdict>
VALID. All claims verified: chair detected, sofa detected, and sofa is geometrically to the right of the chair. Derived answer "sofa" matches ground-truth answer "sofa".
</final_verdict>'''
    },
    {
        "description": "Semantic Attribute Verification (wearing a shirt)",
        "full_trace": '''<thought>
The scaffold requires detecting a person (e0), detecting a shirt (e1), verifying the semantic relation "wearing" using visual probe (e2), and determining the person's identity.
</thought>
<extract_claims>
1. There is a person in the image.
2. There is a shirt in the image.
3. The person is wearing the shirt.
4. The identity of the person wearing the shirt is "girl".
</extract_claims>
<tool_verification>
```python
# Step 0: Detect person candidates
e0 = detect_and_crop(img_path, "person")
assert len(e0) > 0, "No person detected"

# Step 1: Detect shirts
e1 = detect_and_crop(img_path, "shirt")
assert len(e1) > 0, "No shirt detected"

# Step 2: Verify semantic relation 'wearing' using vlm_probe on person crops
wearing_shirt = []
for p in e0:
    if vlm_probe(img_path, p, "Is this person wearing a shirt?"):
        wearing_shirt.append(p)
assert len(wearing_shirt) > 0, "No person found wearing a shirt"

# Step 3: Verify identity of the person
person_box = wearing_shirt[0]
is_girl = vlm_probe(img_path, person_box, "Is this person a girl?")
assert is_girl, "Person wearing shirt is not a girl"

final_answer = "girl"
```
</tool_verification>
<final_verdict>
VALID. All claims verified: person and shirt detected, wearing relation confirmed via VLM, and person verified as a girl. Derived answer "girl" matches ground-truth answer "girl".
</final_verdict>'''
    },
    {
        "description": "Filtering by Horizontal Position and Relational Query",
        "full_trace": '''<thought>
The scaffold requires detecting benches (e0), filtering for the bench on the right side of the image (e1), detecting an animal (bird) sitting on that bench (e2), and deriving the answer (e3).
</thought>
<extract_claims>
1. There is a bench in the image.
2. The bench is on the right side of the scene.
3. An animal (bird) is sitting on the right-side bench.
4. The name of the animal answers the question as "bird".
</extract_claims>
<tool_verification>
```python
# Step 0: Detect benches
e0 = detect_and_crop(img_path, "bench")
assert len(e0) > 0, "No bench detected"

# Step 1: Filter bench on the right side using horizontal position
# Center x coordinate is (xmin + xmax) / 2
right_bench = max(e0, key=lambda b: (b[0] + b[2]) / 2)
e1 = [right_bench]

# Step 2: Detect animal candidates (bird) and verify sitting relation
e2_candidates = detect_and_crop(img_path, "bird")
assert len(e2_candidates) > 0, "No bird detected"

sitting_bird = [
    b for b in e2_candidates
    if check_spatial_relation(b, right_bench, "overlapping") or check_spatial_relation(b, right_bench, "near")
]
assert len(sitting_bird) > 0, "No bird found sitting on the right bench"
e2 = sitting_bird

# Step 3: Name of the animal
final_answer = "bird"
```
</tool_verification>
<final_verdict>
VALID. All claims verified: bench detected, right-side bench localized, bird detected and verified sitting on the bench. Derived answer "bird" matches ground-truth answer "bird".
</final_verdict>'''
    }
]


def format_fewshot_block(examples: list = None) -> str:
    """Formats exemplars for injection into the coder prompt. Returns '' if none."""
    examples = examples if examples is not None else FEWSHOT_EXAMPLES
    if not examples:
        return ""
    blocks = [
        f"### FEW-SHOT EXEMPLAR {i+1} ({ex['description']}):\n{ex['full_trace']}"
        for i, ex in enumerate(examples)
    ]
    return "\n\n".join(blocks)
