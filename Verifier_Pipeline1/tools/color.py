"""get_color: dominant color of a cropped region. Pure image processing, no model call."""
from PIL import Image
import numpy as np

# Coarse named-color buckets in RGB. Extend if traces need finer distinctions.
_COLOR_NAMES = {
    "red": (220, 20, 60), "orange": (255, 140, 0), "yellow": (240, 220, 60),
    "green": (34, 139, 34), "blue": (30, 60, 200), "purple": (128, 0, 200),
    "pink": (255, 105, 180), "brown": (139, 90, 43), "black": (20, 20, 20),
    "white": (245, 245, 245), "gray": (128, 128, 128),
}


def get_color(image_path: str, bbox: list) -> str:
    """
    Returns the closest named color for the dominant pixel color in bbox
    = [xmin, ymin, xmax, ymax]. Deterministic, no model call.
    NOTE: this is a coarse heuristic (mean color + nearest named bucket) --
    good enough for "what color is X" claims, not a substitute for a real
    color-classification model if traces demand finer granularity later.
    """
    image = Image.open(image_path).convert("RGB")
    xmin, ymin, xmax, ymax = bbox
    crop = image.crop((xmin, ymin, xmax, ymax))
    arr = np.array(crop).reshape(-1, 3)
    if arr.size == 0:
        return "unknown"
    mean_color = arr.mean(axis=0)

    best_name, best_dist = "unknown", float("inf")
    for name, rgb in _COLOR_NAMES.items():
        dist = sum((a - b) ** 2 for a, b in zip(mean_color, rgb))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name