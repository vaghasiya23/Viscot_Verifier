"""
vlm_probe: general-purpose visual QA fallback for anything geometry/OCR/color
can't resolve -- semantic relations (wearing, holding, riding, eating),
state/attribute filters (old, open, little), naming fallback, A-vs-B comparisons.

Backend: Qwen2-VL-2B-Instruct (small enough to co-exist with the 7B coder
model + OWLv2 + EasyOCR on a single A6000 48GB GPU).
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

MOCK_MODE = False  # Real backend is wired in now

print("[probe.py] Loading Qwen2-VL-2B-Instruct for VLM probing...")
_vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
_vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("[probe.py] VLM probe ready.")


def vlm_probe(image_path: str, bbox: list, question: str) -> bool:
    """
    Crops the image to bbox = [xmin, ymin, xmax, ymax], asks the VLM a
    yes/no question about that crop, returns True/False.

    Use this for:
    - Semantic relations: "Is this person wearing a shirt?"
    - Attributes/states: "Is this door open?"
    - Identity verification: "Is this a girl?"
    - Anything check_spatial_relation/get_color/read_text_ocr can't handle.
    """
    # Crop the image to the bounding box region
    image = Image.open(image_path).convert("RGB")
    xmin, ymin, xmax, ymax = bbox
    crop = image.crop((xmin, ymin, xmax, ymax))

    # Save temp crop (Qwen2-VL needs a file path or PIL image)
    import tempfile, os
    tmp_path = os.path.join(tempfile.gettempdir(), "vlm_probe_crop.jpg")
    crop.save(tmp_path)

    # Build the prompt: force a yes/no answer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": tmp_path},
                {"type": "text", "text": f"Answer with ONLY 'yes' or 'no'. {question}"},
            ],
        }
    ]

    text = _vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_vlm_model.device)

    generated_ids = _vlm_model.generate(**inputs, max_new_tokens=10)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    answer = _vlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()

    print(f"      [VLM Probe] Q: '{question}' -> A: '{answer}'")

    # Parse yes/no
    return answer.startswith("yes")