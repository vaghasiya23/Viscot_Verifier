import torch
from transformers import pipeline
from PIL import Image

print("Loading OWLv2 Object Detector (this takes a few seconds)...")
# Load a lightweight zero-shot object detector to act as our detect_and_crop tool
detector = pipeline(
    model="google/owlv2-base-patch16-ensemble", 
    task="zero-shot-object-detection",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def detect_and_crop(image_path: str, object_name: str, threshold: float = 0.1) -> list:
    """Finds objects in the image and returns bounding boxes [xmin, ymin, xmax, ymax]."""
    image = Image.open(image_path).convert("RGB")
    predictions = detector(
        image, 
        candidate_labels=[object_name],
    )
    # Filter by threshold
    boxes = [pred["box"] for pred in predictions if pred["score"] > threshold]
    
    # Format into a simple list of lists: [[xmin, ymin, xmax, ymax]]
    formatted_boxes = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in boxes]
    return formatted_boxes

def check_spatial_relation(box1: list, box2: list, relation: str) -> bool:
    """Checks basic spatial math between two boxes [xmin, ymin, xmax, ymax]"""
    # Box format: [xmin, ymin, xmax, ymax]
    b1_xmin, b1_ymin, b1_xmax, b1_ymax = box1
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = box2
    
    if relation == "overlapping" or relation == "wearing":
        # Check for intersection
        overlap_x = (b1_xmin < b2_xmax) and (b1_xmax > b2_xmin)
        overlap_y = (b1_ymin < b2_ymax) and (b1_ymax > b2_ymin)
        return overlap_x and overlap_y
    
    return False

# ==========================================
# EXECUTING THE GOLDEN TRACE FOR 2331819.jpg
# ==========================================
if __name__ == "__main__":
    img_path = "/data1/aminur/om/GRPO/official_viscot_gqa_10/images/2331819.jpg"
    print(f"\n[Sandbox] Executing Verification Trace for {img_path}...")
    
    try:
        # Claim 1: Person exists
        print("[Sandbox] Running: detect_and_crop(img, 'person')")
        person_boxes = detect_and_crop(img_path, 'person')
        print(f"   -> Found {len(person_boxes)} person(s). Boxes: {person_boxes}")
        assert len(person_boxes) >= 1, 'No person detected.'
        
        # Claim 2: Shirt exists
        print("[Sandbox] Running: detect_and_crop(img, 'shirt')")
        shirt_boxes = detect_and_crop(img_path, 'shirt')
        print(f"   -> Found {len(shirt_boxes)} shirt(s). Boxes: {shirt_boxes}")
        assert len(shirt_boxes) >= 1, 'No shirt detected.'
        
        # Claim 3: Person is wearing shirt (overlapping)
        print("[Sandbox] Running: check_spatial_relation(person, shirt, 'overlapping')")
        is_wearing = check_spatial_relation(person_boxes[0], shirt_boxes[0], 'overlapping')
        print(f"   -> Result: {is_wearing}")
        assert is_wearing == True, 'The person is not wearing the shirt.'
        
        print("\n[VERDICT] SUCCESS: All claims verified. The Reasoning Step is VALID.")
        
    except AssertionError as e:
        print(f"\n[VERDICT] FAILED: {e}. The Reasoning Step is INVALID.")

