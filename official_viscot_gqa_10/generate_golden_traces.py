import json
import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = "/data1/aminur/om/GRPO/official_viscot_gqa_10"
input_file = os.path.join(base_dir, "gqa_10_samples.json")
output_file = os.path.join(base_dir, "generated_3_samples.json")

import sys
sys.path.append(base_dir)
from tools.sandbox import detect_and_crop, check_spatial_relation
from tools.color_tool import extract_dominant_color
from tools.ocr_tool import read_text_ocr

print("Loading Qwen2.5-Coder-7B-Instruct (from local cache)...")
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

SYSTEM_PROMPT = """You are an expert Visual Verification Programmer. Your task is to verify a multimodal reasoning step.
You are provided with an image path and a reasoning text.

You MUST use this EXACT Python API:
- detect_and_crop(img_path: str, obj: str) -> list
- check_spatial_relation(box1: list, box2: list, relation: str) -> bool
- extract_dominant_color(img_path: str, bbox: list) -> list
- read_text_ocr(img_path: str, bbox: list=None) -> str

INSTRUCTIONS:
1. Write a <thought> block planning your verification.
2. Write an <extract_claims> block breaking the text into atomic facts.
3. Write a <tool_verification> block containing ONLY executable Python code enclosed in ```python ... ```.
   - Assign tool outputs to variables.
   - Always check if a list is empty before accessing elements (e.g., `if len(boxes) > 0:`).
   - Use `assert` statements. If a claim is false, the assert must fail.
   - The spatial relation MUST ONLY be 'overlapping', 'left', 'right', or 'inside'.
   - Use the variable `img_path` which is already provided to you. Do NOT import cv2 or numpy.

EXAMPLE OF A PERFECT TRACE:
<thought>
I need to verify that there is a person, a shirt, and the person is wearing the shirt.
</thought>
<extract_claims>
1. There is a person.
2. There is a shirt.
3. The person is wearing the shirt.
</extract_claims>
<tool_verification>
```python
# Claim 1: Person exists
person_boxes = detect_and_crop(img_path, "person")
assert len(person_boxes) > 0, "No person found"
person_box = person_boxes[0]

# Claim 2: Shirt exists
shirt_boxes = detect_and_crop(img_path, "shirt")
assert len(shirt_boxes) > 0, "No shirt found"

# Claim 3: Person is wearing shirt
is_wearing = check_spatial_relation(person_box, shirt_boxes[0], "overlapping") 
assert is_wearing == True, "Person is not wearing a shirt"
```
</tool_verification>
"""

def generate_trace(sample):
    img_filename = sample['image']
    img_path = os.path.join(base_dir, "images", img_filename)
    user_content = f"Image: {img_path}\nReasoning to verify: {sample['thought']}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print(f"\nGenerating trace for {img_filename}...")
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response, img_path

with open(input_file, 'r') as f:
    samples = json.load(f)[:3]

generated_dataset = []

for sample in samples:
    trace, img_path = generate_trace(sample)
    code_match = re.search(r"```python(.*?)```", trace, re.DOTALL)
    execution_success = False
    error_msg = ""
    
    if code_match:
        python_code = code_match.group(1).strip()
        print("\n--- Generated Code ---")
        print(python_code)
        
        local_scope = {
            'img_path': img_path,
            'detect_and_crop': detect_and_crop,
            'check_spatial_relation': check_spatial_relation,
            'extract_dominant_color': extract_dominant_color,
            'read_text_ocr': read_text_ocr
        }
        
        try:
            print(f"Executing Sandbox...")
            exec(python_code, globals(), local_scope)
            print("=> EXECUTION SUCCESS")
            execution_success = True
        except Exception as e:
            print(f"=> EXECUTION FAILED: {e}")
            error_msg = str(e)
            
    generated_dataset.append({
        "image": sample['image'],
        "input_cot": sample['thought'],
        "teacher_output": trace,
        "execution_passed": execution_success,
        "error_if_any": error_msg
    })

with open(output_file, 'w') as f:
    json.dump(generated_dataset, f, indent=4)
