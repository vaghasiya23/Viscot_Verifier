import json
import re
import os
from PIL import Image, ImageDraw, ImageFont
from filters.execution_filter import run

os.makedirs("data/processed/visualizations", exist_ok=True)

with open("data/processed/golden/golden_traces.json") as f:
    traces = json.load(f)

for i, t in enumerate(traces):
    print(f"\n==================================================")
    print(f"SAMPLE {i+1}: {t['image']}")
    print(f"Question: {t['question']}")
    print(f"Ground-truth Answer: {t['answer']}")
    print(f"==================================================")

    # Extract python code
    code_match = re.search(r"```python(.*?)```", t["golden_target"], re.DOTALL)
    if not code_match:
        continue
    code = code_match.group(1).strip()
    img_path = os.path.join("data/raw/images", t["image"])

    print("\n--- EXECUTING CODE IN SANDBOX ---")
    res = run(code, img_path)
    print(f"Execution Succeeded: {res['ok']}")

    scope = res.get("local_scope", {})
    print("\n--- RUNTIME VARIABLE OUTPUTS ---")
    tool_names = {'detect_and_crop', 'check_spatial_relation', 'vlm_probe', 'read_text_ocr', 
                  'get_color', 'compare_size', 'resolve_left_right', 'estimate_depth_order', 'img_path'}
    
    # Let's collect boxes to draw on the image
    boxes_to_draw = []
    
    for k, v in scope.items():
        if not k.startswith("_") and k not in tool_names:
            print(f"  {k} = {v}")
            # If it's a bounding box or list of boxes, save for visualization
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], (int, float)) and len(v) == 4:
                    boxes_to_draw.append((k, v))
                elif isinstance(v[0], list) and len(v[0]) == 4:
                    for idx, b in enumerate(v):
                        boxes_to_draw.append((f"{k}[{idx}]", b))

    # Draw boxes on the image and save
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
        
        for c_idx, (label, box) in enumerate(boxes_to_draw):
            color = colors[c_idx % len(colors)]
            xmin, ymin, xmax, ymax = box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            draw.text((xmin + 4, ymin + 4), label, fill=color)
        
        save_path = os.path.join("data/processed/visualizations", f"sample_{i+1}_{t['image']}")
        img.save(save_path)
        print(f"\nSaved visual annotated image to: {save_path}")

print("\nAll golden sample executions and visual overlays completed!")
