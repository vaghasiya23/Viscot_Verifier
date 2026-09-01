import easyocr
from PIL import Image
import numpy as np

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)

def read_text_ocr(image_path: str, bbox: list = None) -> str:
    """
    Extracts text from an image. 
    If bbox is provided [xmin, ymin, xmax, ymax], it crops first.
    """
    image = Image.open(image_path).convert("RGB")
    
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        image = image.crop((xmin, ymin, xmax, ymax))
    
    # EasyOCR expects a numpy array
    img_np = np.array(image)
    
    # Read text
    results = reader.readtext(img_np, detail=0) # detail=0 returns just strings
    
    if not results:
        return ""
    
    return " ".join(results)

if __name__ == "__main__":
    # Test it on one of our GQA images
    img_path = "/data1/aminur/om/GRPO/official_viscot_gqa_10/images/2331819.jpg"
    print("\n[Sandbox-OCR] Running read_text_ocr on the full image...")
    text = read_text_ocr(img_path)
    print(f"   -> Extracted Text: '{text}'")
