"""
tools/detector.py - Open-vocabulary object detection via OWLv2.
"""
from typing import List
from PIL import Image
import torch
from tools.spatial import deduplicate_boxes

_DETECTOR = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_detector():
    """Lazy loader for OWLv2 pipeline to preserve GPU memory until needed."""
    global _DETECTOR
    if _DETECTOR is None:
        device = get_device()
        print(f"[detector.py] Loading OWLv2 (google/owlv2-base-patch16-ensemble) on {device}...")
        from transformers import pipeline
        try:
            _DETECTOR = pipeline(
                model="google/owlv2-base-patch16-ensemble",
                task="zero-shot-object-detection",
                device=device,
            )
        except Exception as e:
            if device != "cpu":
                print(f"[detector.py] Loading on {device} failed ({e}), falling back to CPU...")
                _DETECTOR = pipeline(
                    model="google/owlv2-base-patch16-ensemble",
                    task="zero-shot-object-detection",
                    device="cpu",
                )
            else:
                raise
    return _DETECTOR


def detect(image_path: str, object_query: str, threshold: float = 0.1, crop_box: List[float] = None) -> List[List[float]]:
    """
    Finds instances of `object_query` in the image (or inside crop_box).
    Returns list of [xmin, ymin, xmax, ymax] absolute pixel boxes.
    """
    detector = get_detector()
    with Image.open(image_path) as full_img:
        full_img = full_img.convert("RGB")
        offset_x, offset_y = 0.0, 0.0
        if crop_box is not None:
            # Crop to specified region with safe boundary clamping
            iw, ih = full_img.size
            xmin = max(0, min(iw, int(crop_box[0])))
            ymin = max(0, min(ih, int(crop_box[1])))
            xmax = max(xmin, min(iw, int(crop_box[2])))
            ymax = max(ymin, min(ih, int(crop_box[3])))
            if xmax <= xmin or ymax <= ymin:
                return []
            target_img = full_img.crop((xmin, ymin, xmax, ymax))
            offset_x, offset_y = float(xmin), float(ymin)
        else:
            target_img = full_img

        try:
            predictions = detector(target_img, candidate_labels=[object_query])
        except Exception as e:
            # Fallback to CPU pipeline if MPS encounters an unhandled operation
            if getattr(detector, "device", None) is not None and str(detector.device) != "cpu":
                print(f"[detector.py] Prediction failed on {detector.device}, retrying on CPU: {e}")
                from transformers import pipeline
                cpu_detector = pipeline(
                    model="google/owlv2-base-patch16-ensemble",
                    task="zero-shot-object-detection",
                    device="cpu",
                )
                predictions = cpu_detector(target_img, candidate_labels=[object_query])
            else:
                raise

    boxes = sorted((p for p in predictions if p["score"] > threshold), key=lambda p: p["score"], reverse=True)
    raw_boxes = [
        [
            b["box"]["xmin"] + offset_x,
            b["box"]["ymin"] + offset_y,
            b["box"]["xmax"] + offset_x,
            b["box"]["ymax"] + offset_y
        ]
        for b in boxes
    ]
    return deduplicate_boxes(raw_boxes)
