"""
detect_and_crop: open-vocab object detection via OWLv2.
Handles both `select` (plain object query) and `filter` (compound attribute
query, e.g. "wooden table", "small chair") -- OWLv2 takes free-text labels,
so filter-by-attribute is just a richer query string, not a separate tool.

Box format returned everywhere in this project: [xmin, ymin, xmax, ymax] (flat list, floats).
"""
import torch
from transformers import pipeline
from PIL import Image
from verification_utils import deduplicate_boxes

print("[detection.py] Loading OWLv2 (google/owlv2-base-patch16-ensemble)...")
_detector = pipeline(
    model="google/owlv2-base-patch16-ensemble",
    task="zero-shot-object-detection",
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def detect_and_crop(image_path: str, object_query: str, threshold: float = 0.1) -> list:
    """
    Finds instances of `object_query` in the image.

    object_query can be a plain class ("person") or a compound attribute
    query ("wooden table", "small red chair") -- OWLv2 handles free text,
    so this single call covers both `select` and `filter` GQA ops.

    Returns: list of [xmin, ymin, xmax, ymax] boxes (possibly empty).
    NOTE: empty list is a valid, expected result -- callers MUST check
    len(boxes) before indexing. Do not assume detection always succeeds.
    """
    image = Image.open(image_path).convert("RGB")
    predictions = _detector(image, candidate_labels=[object_query])
    boxes = sorted((p for p in predictions if p["score"] > threshold), key=lambda p: p["score"], reverse=True)
    raw_boxes = [[b["box"]["xmin"], b["box"]["ymin"], b["box"]["xmax"], b["box"]["ymax"]] for b in boxes]
    return deduplicate_boxes(raw_boxes)