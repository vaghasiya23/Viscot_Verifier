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


class UncertainVisualEvidence(RuntimeError):
    pass


def _ask(image, question, max_tokens=24):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = _vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = _vlm_processor(text=[text], images=images, videos=videos,
                            padding=True, return_tensors="pt").to(_vlm_model.device)
    with torch.inference_mode():
        ids = _vlm_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    answer = _vlm_processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)],
        skip_special_tokens=True)[0].strip().lower()
    print(f"      [VLM Probe] Q: {question!r} -> A: {answer!r}")
    return answer


def _yes_no(image, question):
    answer = _ask(image, "Answer only yes, no, or unknown if unclear. " + question)
    answer = answer.rstrip(".!?")
    if answer not in {"yes", "no"}:
        raise UncertainVisualEvidence("Visual probe did not give a definite yes/no answer")
    return answer == "yes"


def vlm_probe(image_path, bbox, question):
    with Image.open(image_path) as source:
        crop = source.convert("RGB").crop(tuple(bbox))
    return _yes_no(crop, question)


def vlm_relation(image_path, subject_box, object_box, predicate, subject_label, object_label):
    # The full image preserves context; marked boxes bind the two referents.
    from verification_utils import box_iou
    if box_iou(subject_box, object_box) >= 0.9:
        return False  # Skip self-pairs; other distinct pairs may establish the relation.
    from PIL import ImageDraw
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(subject_box), outline="red", width=3)
    draw.rectangle(tuple(object_box), outline="blue", width=3)
    question = (f"Is the {subject_label} in the RED box {predicate} the "
                f"{object_label} in the BLUE box? Judge only these two marked objects.")
    return _yes_no(image, question)


def vlm_query(image_path, bbox, attribute):
    if attribute not in {"name", "type", "color", "material", "shape"}:
        raise ValueError("Unsupported visual query attribute")
    with Image.open(image_path) as source:
        crop = source.convert("RGB").crop(tuple(bbox))
    question = ("Name the object in this image in one or two words." if attribute in {"name", "type"}
                else f"What is the {attribute} of the object?")
    answer = _ask(crop, question + " Give only the answer.")
    if not answer or answer.rstrip(".!?") in {"unknown", "unclear", "none", "n/a"}:
        raise UncertainVisualEvidence("Visual answer is uncertain")
    return answer


def vlm_related_name(image_path, references, predicate, role):
    """Propose an unnamed entity without seeing the dataset answer.

    This is only a proposal: detection and a bound pair relation check follow.
    """
    if not references or role not in {"s", "o"}:
        raise ValueError("Missing reference boxes or invalid relation role")
    from PIL import ImageDraw
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in references:
        draw.rectangle(tuple(box), outline="red", width=3)
    question = (f"What object is {predicate} the objects outlined in red?" if role == "s" else
                f"What are the objects outlined in red {predicate}?")
    answer = _ask(image, question + " Answer with only the related object name, in one or two words.")
    if not answer or answer.rstrip(".!?") in {"unknown", "unclear", "none", "n/a"} or len(answer.split()) > 6:
        raise UncertainVisualEvidence("Cannot propose an unambiguous related object label")
    return answer.rstrip(".!?")
