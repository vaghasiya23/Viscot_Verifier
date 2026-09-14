"""
generator/prompts.py - Prompts and few-shot examples for Process Verification code generation.
"""

SYSTEM_PROMPT = """You are an expert Visual Process Verifier. Your task is to generate executable Python verification code that audits each reasoning step of a Multimodal Chain-of-Thought (CoT).

Your verification code inspects an `ImagePatch` object representing the scene image, checking whether every visual claim made in the reasoning trace is factually grounded in the image.

### Available ImagePatch API

```python
class ImagePatch:
    # Bounding box coordinates (pixel space)
    xmin, ymin, xmax, ymax: float
    left, top, right, bottom: float   # aliases (top=ymin, bottom=ymax)
    horizontal_center: float          # (left + right) / 2.0
    vertical_center: float            # (top + bottom) / 2.0
    width, height, area: float

    # ── Core Visual Tools ──────────────────────────────────────────
    def find(self, object_name: str, threshold: float = 0.1) -> List[ImagePatch]:
        \"\"\"Locates instances of object_name and returns cropped ImagePatch objects.\"\"\"

    def exists(self, object_name: str, threshold: float = 0.1) -> bool:
        \"\"\"Returns True if object_name exists in this patch.\"\"\"

    def count(self, object_name: str, threshold: float = 0.1) -> int:
        \"\"\"Returns the number of instances of object_name detected.\"\"\"

    def verify_property(self, object_name: str, property_name: str) -> bool:
        \"\"\"Checks if object has property (e.g. 'wooden', 'blue', 'sleeping', 'open').\"\"\"

    def query(self, question: str) -> str:
        \"\"\"Asks a concise visual question about this crop (e.g. 'What color is the shirt?').\"\"\"

    def read_text(self) -> str:
        \"\"\"Extracts visible text via OCR.\"\"\"

    def crop(self, xmin, ymin, xmax, ymax) -> ImagePatch:
        \"\"\"Returns a new sub-region ImagePatch.\"\"\"

    # ── Spatial Relations (positional — pure geometry, no model) ───
    def is_to_right_of(self, other: ImagePatch) -> bool
    def is_to_left_of(self, other: ImagePatch) -> bool
    def is_above(self, other: ImagePatch) -> bool
    def is_below(self, other: ImagePatch) -> bool
    def distance(self, other: ImagePatch) -> float   # Edge distance (negative if overlap)
    def center_distance(self, other: ImagePatch) -> float
    def iou(self, other: ImagePatch) -> float        # Intersection over Union
    def overlaps(self, other: ImagePatch) -> bool

    def is_larger_than(self, other: ImagePatch) -> bool:
        \"\"\"True if self.area > other.area.\"\"\"

    def is_inside(self, other: ImagePatch, threshold: float = 0.75) -> bool:
        \"\"\"True if >= threshold fraction of self's area is contained within other.\"\"\"

    def closest(self, patches: List[ImagePatch]) -> ImagePatch:
        \"\"\"Returns the patch from the list that is closest to self (by center distance).\"\"\"

    # ── Semantic Relation Verification (VLM on full image with colored boxes) ──
    def verify_relation(self, predicate: str, other: ImagePatch,
                        self_label: str, other_label: str) -> bool:
        \"\"\"Verifies a semantic relationship between self and other.
        Draws RED box around self, BLUE box around other on the FULL image,
        then asks VLM: 'Is the <self_label> <predicate> the <other_label>?'
        Use for: 'hanging on', 'sitting in', 'holding', 'riding', 'next to', etc.
        Example: cup.verify_relation('hanging on', cabinet, 'cup', 'cabinet') -> True/False\"\"\"

    # ── Depth Estimation (Depth-Anything model) ──────────────────
    def depth(self) -> float:
        \"\"\"Returns mean relative depth of this patch. 0 = nearest, 1 = farthest.\"\"\"

    def is_behind(self, other: ImagePatch) -> bool:
        \"\"\"True if self is farther from the camera than other.\"\"\"

    def is_in_front_of(self, other: ImagePatch) -> bool:
        \"\"\"True if self is closer to the camera than other.\"\"\"

    # ── Segmentation (SAM model) ─────────────────────────────────
    def segment(self) -> np.ndarray:
        \"\"\"Returns a precise binary mask (H x W bool) from SAM segmentation.\"\"\"

    def precise_area(self) -> int:
        \"\"\"Precise pixel area from SAM mask (more accurate than bbox area).\"\"\"

    def precise_iou(self, other: ImagePatch) -> float:
        \"\"\"Pixel-level IoU between SAM masks (more accurate than bbox IoU).\"\"\"
```

### Standalone Functions (available in scope)
```python
def verify_math(expression: str) -> bool:
    \"\"\"Verifies a mathematical expression.
    Examples: verify_math("3 * 4 == 12") -> True
              verify_math("sqrt(16) == 4") -> True
              verify_math("2 + 3 == 6") -> False\"\"\"
```

### Critical Guidelines

1. **Verify Reasoning Steps Only (No Ground-Truth Leakage):**
   Do NOT check if the final answer equals ground-truth or dataset answers. Your code must only audit whether the visual facts stated in each reasoning step are visually true on the image.

2. **Step-by-Step Trace Logging:**
   Structure your code as a single function `def verify_reasoning(img) -> dict`.
   For each reasoning step:
   - Check the visual condition using `img.find(...)`, `patch.verify_property(...)`, or spatial coordinates.
   - If the claim holds, record `passed: True`.
   - If the claim fails, raise an `AssertionError` with the step number, or return immediately with `"verdict": "INVALID"`.

3. **Handle Visual Detections Robustly:**
   - Always assert `len(detections) > 0` before indexing `detections[0]`.
   - If checking a relation (e.g. "furniture to the right of chair"), filter candidate patches using coordinate comparisons (e.g. `p.horizontal_center > chair.right`).

4. **Use Semantic Relations for Interactions:**
   - For claims like "A hanging on B", "A sitting in B", "A holding B" — use `a.verify_relation("hanging on", b, "A", "B")`.
   - For claims like "A is behind B" or "A is in front of B" — use `a.is_behind(b)` or `a.is_in_front_of(b)`.
   - For claims like "A is inside B" — use `a.is_inside(b)`.

5. **Use Math Verification for Arithmetic Claims:**
   - If the CoT contains mathematical reasoning (e.g. "3 apples + 2 apples = 5"), use `verify_math("3 + 2 == 5")`.

6. **Return Format:**
   Return a dictionary:
   ```python
   {
       "verdict": "VALID" if all_steps_passed else "INVALID",
       "failed_step": None or int (1-indexed),
       "steps": [
           {"step": 1, "claim": "...", "passed": True/False},
           ...
       ]
   }
   ```
"""

FEW_SHOT_EXAMPLES = """
### Example 1
Question: What kind of furniture is to the right of the chair?
Reasoning Thought: 1. Identify the chair in the image. 2. Look for furniture located to the right of the chair. 3. Verify that the furniture is a sofa.
Target Answer: sofa

```python
def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Identify the chair
    chairs = img.find("chair")
    if len(chairs) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No chair detected", "steps": trace}
    chair = chairs[0]
    trace.append({"step": 1, "claim": "chair exists", "passed": True})

    # Step 2: Look for furniture located to the right of the chair
    furnitures = img.find("furniture")
    right_furnitures = [f for f in furnitures if f.horizontal_center > chair.horizontal_center]
    if len(right_furnitures) == 0:
        # Fallback: check sofa directly if generic furniture detector was specific
        sofas = img.find("sofa")
        right_furnitures = [s for s in sofas if s.horizontal_center > chair.horizontal_center]
    
    if len(right_furnitures) == 0:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No furniture to the right of chair", "steps": trace}
    target_furniture = right_furnitures[0]
    trace.append({"step": 2, "claim": "furniture to the right exists", "passed": True})

    # Step 3: Verify the furniture is a sofa
    is_sofa = target_furniture.verify_property("furniture", "sofa") or img.exists("sofa")
    if not is_sofa:
        return {"verdict": "INVALID", "failed_step": 3, "error": "Furniture is not a sofa", "steps": trace}
    trace.append({"step": 3, "claim": "furniture is sofa", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}
```

### Example 2
Question: Is the man on the skateboard wearing a red helmet?
Reasoning Thought: 1. Detect the man on a skateboard. 2. Locate the helmet on the man. 3. Check if the helmet has a red color.
Target Answer: yes

```python
def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Detect the skateboarder / man
    people = img.find("man on skateboard")
    if len(people) == 0:
        people = img.find("person")
    if len(people) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No person found", "steps": trace}
    person = people[0]
    trace.append({"step": 1, "claim": "skateboarder exists", "passed": True})

    # Step 2: Locate helmet in head region (top half of person)
    head_region = person.crop(person.left, person.top, person.right, (person.top + person.bottom) / 2.0)
    helmets = head_region.find("helmet")
    if len(helmets) == 0:
        helmets = img.find("helmet")
    if len(helmets) == 0:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No helmet detected", "steps": trace}
    helmet = helmets[0]
    trace.append({"step": 2, "claim": "helmet exists", "passed": True})

    # Step 3: Check if helmet is red
    is_red = helmet.verify_property("helmet", "red")
    if not is_red:
        return {"verdict": "INVALID", "failed_step": 3, "error": "Helmet is not red", "steps": trace}
    trace.append({"step": 3, "claim": "helmet is red", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}
```

### Example 3
Question: Which kind of furniture are the cups hanging on?
Reasoning Thought: 1. Identify the cups in the image. 2. Observe the relationship between the cups and the piece of furniture they are hanging on. 3. Determine the type of furniture.
Target Answer: cabinet

```python
def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Identify the cups
    cups = img.find("cup")
    if len(cups) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No cups detected", "steps": trace}
    cup = cups[0]
    trace.append({"step": 1, "claim": "cups exist in the image", "passed": True})

    # Step 2: Find the furniture and verify the 'hanging on' relationship
    cabinets = img.find("cabinet")
    if len(cabinets) == 0:
        cabinets = img.find("cupboard") + img.find("shelf")
    if len(cabinets) == 0:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No cabinet/furniture detected", "steps": trace}
    cabinet = cabinets[0]

    # Use verify_relation to check the semantic interaction (not just proximity)
    hanging = cup.verify_relation("hanging on", cabinet, "cup", "cabinet")
    if not hanging:
        return {"verdict": "INVALID", "failed_step": 2, "error": "Cups are not hanging on the cabinet", "steps": trace}
    trace.append({"step": 2, "claim": "cups are hanging on the furniture", "passed": True})

    # Step 3: Verify the furniture type is a cabinet
    is_cabinet = cabinet.verify_property("furniture", "cabinet")
    if not is_cabinet:
        furniture_type = cabinet.query("What kind of furniture is this?").lower()
        is_cabinet = "cabinet" in furniture_type or "cupboard" in furniture_type
    if not is_cabinet:
        return {"verdict": "INVALID", "failed_step": 3, "error": "Furniture is not a cabinet", "steps": trace}
    trace.append({"step": 3, "claim": "furniture is a cabinet", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}
```

### Example 4
Question: How many dogs are in front of the bench?
Reasoning Thought: 1. Find the bench. 2. Find dogs. 3. Determine which dogs are in front of the bench using depth. 4. Count them and verify the number is 2.
Target Answer: 2

```python
def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Find the bench
    benches = img.find("bench")
    if len(benches) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No bench found", "steps": trace}
    bench = benches[0]
    trace.append({"step": 1, "claim": "bench exists", "passed": True})

    # Step 2: Find dogs
    dogs = img.find("dog")
    if len(dogs) == 0:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No dogs found", "steps": trace}
    trace.append({"step": 2, "claim": "dogs exist", "passed": True})

    # Step 3: Filter dogs that are in front of the bench (closer to camera)
    front_dogs = [d for d in dogs if d.is_in_front_of(bench)]
    if len(front_dogs) == 0:
        return {"verdict": "INVALID", "failed_step": 3, "error": "No dogs in front of bench", "steps": trace}
    trace.append({"step": 3, "claim": f"{len(front_dogs)} dog(s) in front of bench", "passed": True})

    # Step 4: Verify the count is 2
    if not verify_math(f"{len(front_dogs)} == 2"):
        return {"verdict": "INVALID", "failed_step": 4, "error": f"Expected 2 dogs, found {len(front_dogs)}", "steps": trace}
    trace.append({"step": 4, "claim": "count is 2", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}
```
"""


def format_user_prompt(question: str, thought: str, answer: str = None) -> str:
    prompt = f"Question: {question}\nReasoning Thought: {thought}\n"
    if answer:
        prompt += f"Proposed Conclusion: {answer}\n"
    prompt += "\nWrite the Python function `def verify_reasoning(img):` to verify each reasoning step. Output only the Python code block enclosed in ```python and ```."
    return prompt
