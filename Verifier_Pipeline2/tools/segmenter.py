"""
tools/segmenter.py - SAM-based segmentation for precise object masks.
Uses facebook/sam-vit-base, lazy-loaded on first call.
"""
import numpy as np
import torch
from PIL import Image
from typing import List

_SAM_MODEL = None
_SAM_PROCESSOR = None


def get_sam():
    """Lazy loader for SAM model."""
    global _SAM_MODEL, _SAM_PROCESSOR
    if _SAM_MODEL is None or _SAM_PROCESSOR is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[segmenter.py] Loading SAM (facebook/sam-vit-base) on {device}...")
        from transformers import SamModel, SamProcessor

        try:
            _SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        except Exception as e:
            if device != "cpu":
                print(f"[segmenter.py] Loading on {device} failed ({e}), falling back to CPU...")
                _SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-base").to("cpu")
            else:
                raise
        _SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-base")
        print("[segmenter.py] SAM loaded.")
    return _SAM_MODEL, _SAM_PROCESSOR


def segment_box(image_path: str, box: List[float]) -> np.ndarray:
    """
    Returns a binary mask (H x W bool array) for the object within the
    given bounding box [xmin, ymin, xmax, ymax].

    Uses SAM with the box as a prompt to get a precise segmentation.
    """
    model, processor = get_sam()
    image = Image.open(image_path).convert("RGB")
    iw, ih = image.size

    # Clamp box coordinates to image boundaries
    clamped_box = [
        max(0, min(iw, int(box[0]))),
        max(0, min(ih, int(box[1]))),
        max(0, min(iw, int(box[2]))),
        max(0, min(ih, int(box[3]))),
    ]

    # SAM expects input_boxes as [[[x1, y1, x2, y2]]]
    input_box = [clamped_box]
    inputs = processor(image, input_boxes=[input_box], return_tensors="pt").to(
        model.device
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    # Post-process masks to original image size
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    # masks[0] = first image, shape: (1, num_masks, H, W)
    # Pick the mask with the highest IoU score
    scores = outputs.iou_scores[0][0]  # (num_masks,)
    best_idx = scores.argmax().item()

    mask = masks[0][0][best_idx].numpy().astype(bool)
    return mask


def mask_pixel_area(mask: np.ndarray) -> int:
    """Returns the number of True pixels in a binary mask."""
    return int(mask.sum())


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Computes pixel-level IoU between two binary masks of the same shape."""
    if mask_a.shape != mask_b.shape:
        # Resize mask_b to match mask_a
        from PIL import Image as PILImage

        mb = PILImage.fromarray(mask_b.astype(np.uint8) * 255)
        mb = mb.resize((mask_a.shape[1], mask_a.shape[0]), PILImage.NEAREST)
        mask_b = np.array(mb) > 127

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0
