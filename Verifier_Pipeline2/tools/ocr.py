"""
tools/ocr.py - Text extraction via EasyOCR.
"""
from typing import List, Optional
from PIL import Image
import numpy as np

_OCR_READER = None


def get_ocr():
    """Lazy loader for EasyOCR reader."""
    global _OCR_READER
    if _OCR_READER is None:
        print("[ocr.py] Loading EasyOCR...")
        import easyocr
        import torch
        gpu_available = torch.cuda.is_available()
        _OCR_READER = easyocr.Reader(['en'], gpu=gpu_available)
    return _OCR_READER


def read_text(image_path: str, box: Optional[List[float]] = None) -> str:
    """Extracts text from image or crop box. Returns empty string if none found."""
    reader = get_ocr()
    with Image.open(image_path) as full_img:
        img = full_img.convert("RGB")
        if box is not None:
            xmin, ymin, xmax, ymax = [int(v) for v in box]
            img = img.crop((xmin, ymin, xmax, ymax))

    results = reader.readtext(np.array(img), detail=0)
    return " ".join(results).strip() if results else ""
