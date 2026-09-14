"""
tools/depth.py - Monocular depth estimation using Depth-Anything-V2-Small.
Lazy-loaded on first call. Caches depth maps per image path.
"""
import numpy as np
import torch
from PIL import Image
from typing import List

_DEPTH_PIPE = None
_depth_cache: dict = {}


def get_depth_pipeline():
    """Lazy loader for depth estimation pipeline."""
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1

        print(f"[depth.py] Loading Depth-Anything-V2-Small on {device}...")
        from transformers import pipeline

        try:
            _DEPTH_PIPE = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=device,
            )
        except Exception as e:
            if device != -1:
                print(f"[depth.py] Loading on {device} failed ({e}), falling back to CPU...")
                _DEPTH_PIPE = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=-1,
                )
            else:
                raise
        print("[depth.py] Depth-Anything loaded.")
    return _DEPTH_PIPE


def get_depth_map(image_path: str) -> np.ndarray:
    """
    Returns a normalised depth map (H x W float32, values in [0, 1]).
    0 = nearest, 1 = farthest.
    Result is cached per image_path.
    """
    if image_path not in _depth_cache:
        pipe = get_depth_pipeline()
        image = Image.open(image_path).convert("RGB")
        result = pipe(image)
        # result["depth"] is a PIL Image in newer transformers
        depth_pil = result["depth"]
        depth_map = np.array(depth_pil, dtype=np.float32)

        # Normalise to [0, 1]
        dmin, dmax = depth_map.min(), depth_map.max()
        if dmax > dmin:
            depth_map = (depth_map - dmin) / (dmax - dmin)
        else:
            depth_map = np.zeros_like(depth_map)

        _depth_cache[image_path] = depth_map
    return _depth_cache[image_path]


def estimate_depth(image_path: str, box: List[float]) -> float:
    """
    Returns the mean relative depth of the region inside `box` [xmin, ymin, xmax, ymax].
    Value in [0, 1]: 0 = nearest to camera, 1 = farthest.
    """
    depth_map = get_depth_map(image_path)
    dh, dw = depth_map.shape

    # Scale box coords to depth map dimensions (they might differ from original image)
    with Image.open(image_path) as img:
        iw, ih = img.size
    scale_x = dw / iw
    scale_y = dh / ih

    x1 = max(0, int(box[0] * scale_x))
    y1 = max(0, int(box[1] * scale_y))
    x2 = min(dw, int(box[2] * scale_x))
    y2 = min(dh, int(box[3] * scale_y))

    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 0.5  # fallback: unknown depth
    return float(region.mean())
