"""read_text_ocr: text grounding in natural scenes, via EasyOCR."""
import easyocr
from PIL import Image
import numpy as np

print("[ocr.py] Loading EasyOCR...")
_reader = easyocr.Reader(['en'], gpu=True)


def read_text_ocr(image_path: str, bbox: list = None) -> str:
    """
    Extracts text from an image, optionally cropped to bbox = [xmin, ymin, xmax, ymax].
    Returns "" if no text found -- callers should treat empty string as a
    valid "no text here" result, not an error.
    """
    image = Image.open(image_path).convert("RGB")
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        image = image.crop((xmin, ymin, xmax, ymax))
    results = _reader.readtext(np.array(image), detail=0)
    return " ".join(results) if results else ""