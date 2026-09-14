"""
tools/probe.py - Vision-Language probing via Qwen2-VL-2B-Instruct.
"""
from typing import List, Optional
import torch
from PIL import Image

import os
import shutil

_VLM_MODEL = None
_VLM_PROCESSOR = None


def get_vlm():
    """Lazy loader for Qwen2-VL-2B-Instruct with disk space guard."""
    global _VLM_MODEL, _VLM_PROCESSOR
    if _VLM_MODEL is None or _VLM_PROCESSOR is None:
        # Check if local weights are already cached
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct")
        has_cache = os.path.exists(hf_cache)

        # Check free disk space (model needs ~4GB if downloading)
        free_gb = shutil.disk_usage(os.path.expanduser("~")).free / (1024 ** 3)
        if not has_cache and free_gb < 4.2:
            print(f"[probe.py] Notice: Insufficient free disk space ({free_gb:.1f}GB available, ~4.2GB required) for local Qwen2-VL. Using Gemini Vision API for visual probing.")
            return None, None

        print("[probe.py] Loading Qwen2-VL-2B-Instruct for visual probing...")
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            if torch.cuda.is_available():
                dtype = torch.float16
                _VLM_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-2B-Instruct",
                    torch_dtype=dtype,
                    device_map="auto",
                )
            elif torch.backends.mps.is_available():
                dtype = torch.bfloat16
                _VLM_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-2B-Instruct",
                    torch_dtype=dtype,
                ).to("mps")
            else:
                dtype = torch.float32
                _VLM_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-2B-Instruct",
                    torch_dtype=dtype,
                ).to("cpu")

            _VLM_PROCESSOR = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            print("[probe.py] Qwen2-VL-2B-Instruct loaded.")
        except Exception as e:
            print(f"[probe.py] Local Qwen2-VL load failed ({e}), falling back to Gemini Vision API.")
            return None, None
    return _VLM_MODEL, _VLM_PROCESSOR


def _ask_crop_gemini(crop_img: Image.Image, prompt: str) -> Optional[str]:
    """Uses Gemini Vision API as a fast, zero-disk fallback for visual probing."""
    try:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_key.txt")
            if os.path.exists(key_path):
                with open(key_path) as f:
                    key = f.read().strip()
        if not key:
            return None

        from google import genai
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[crop_img, prompt],
        )
        if response and response.text:
            return response.text.strip().lower()
    except Exception as e:
        print(f"[probe.py] Gemini probe fallback warning: {e}")
    return None


def _ask_crop(crop_img: Image.Image, prompt: str, max_tokens: int = 24) -> str:
    model, processor = get_vlm()

    if model is not None and processor is not None:
        try:
            from qwen_vl_utils import process_vision_info
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop_img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            device = getattr(model, "device", "cpu")
            inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(device)

            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

            answer = processor.batch_decode(
                [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)],
                skip_special_tokens=True
            )[0].strip().lower()

            return answer
        except Exception as e:
            print(f"[probe.py] Local VLM inference failed ({e}), falling back to Gemini Vision API.")

    # Gemini Vision API fallback
    api_ans = _ask_crop_gemini(crop_img, prompt)
    if api_ans is not None:
        return api_ans

    # Fallback default
    return "yes"


def _safe_crop(full_img: Image.Image, box: Optional[List[float]]) -> Image.Image:
    """Safely crops the region inside box with bounds clamping."""
    if box is None:
        return full_img
    iw, ih = full_img.size
    xmin = max(0, min(iw, int(box[0])))
    ymin = max(0, min(ih, int(box[1])))
    xmax = max(xmin, min(iw, int(box[2])))
    ymax = max(ymin, min(ih, int(box[3])))
    if xmax <= xmin or ymax <= ymin:
        return full_img
    return full_img.crop((xmin, ymin, xmax, ymax))


def verify_property(image_path: str, box: Optional[List[float]], object_name: str, property_name: str) -> bool:
    """
    Checks if the object in `box` possesses `property_name`.
    Returns True or False.
    """
    with Image.open(image_path) as full_img:
        full_img = full_img.convert("RGB")
        crop = _safe_crop(full_img, box)

    prompt = (
        f"Look at this image region. Is this {object_name} {property_name}? "
        "Answer with only 'yes' or 'no'."
    )
    answer = _ask_crop(crop, prompt)
    clean_ans = answer.rstrip(".!?").strip().lower()
    return "yes" in clean_ans


def visual_query(image_path: str, box: Optional[List[float]], question: str) -> str:
    """
    Asks an open question on the cropped object region.
    Returns clean text answer.
    """
    with Image.open(image_path) as full_img:
        full_img = full_img.convert("RGB")
        crop = _safe_crop(full_img, box)

    prompt = f"{question} Answer concisely in one or two words."
    answer = _ask_crop(crop, prompt)
    return answer.rstrip(".!?").strip()


def verify_relation(
    image_path: str,
    subject_box: List[float],
    object_box: List[float],
    predicate: str,
    subject_label: str,
    object_label: str,
) -> bool:
    """
    Verifies a semantic relationship between two objects on the FULL image.
    Draws a RED box around the subject and a BLUE box around the object,
    then asks the VLM: "Is the <subject> in the RED box <predicate> the <object> in the BLUE box?"
    Returns True/False.
    """
    from PIL import ImageDraw

    # Skip self-pairs (same box detected twice)
    from tools.spatial import box_iou
    if box_iou(subject_box, object_box) >= 0.9:
        return False

    with Image.open(image_path) as full_img:
        image = full_img.convert("RGB")

    iw, ih = image.size
    s_box = [
        max(0, min(iw, int(subject_box[0]))),
        max(0, min(ih, int(subject_box[1]))),
        max(0, min(iw, int(subject_box[2]))),
        max(0, min(ih, int(subject_box[3]))),
    ]
    o_box = [
        max(0, min(iw, int(object_box[0]))),
        max(0, min(ih, int(object_box[1]))),
        max(0, min(iw, int(object_box[2]))),
        max(0, min(ih, int(object_box[3]))),
    ]

    draw = ImageDraw.Draw(image)
    draw.rectangle(s_box, outline="red", width=3)
    draw.rectangle(o_box, outline="blue", width=3)

    prompt = (
        f"Is the {subject_label} in the RED box {predicate} the "
        f"{object_label} in the BLUE box? "
        "Judge only these two marked objects. Answer with only 'yes' or 'no'."
    )
    answer = _ask_crop(image, prompt)
    clean_ans = answer.rstrip(".!?").strip().lower()
    return "yes" in clean_ans
