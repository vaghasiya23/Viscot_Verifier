import sys
import os
from PIL import Image, ImageDraw

sys.path.append("/data1/aminur/om/GRPO/official_viscot_gqa_10")
from sandbox import detect_and_crop

def draw_and_save(img_path, save_path):
    print("Running detector to get boxes...")
    person_boxes = detect_and_crop(img_path, 'person')
    shirt_boxes = detect_and_crop(img_path, 'shirt')
    
    print("Drawing boxes on image...")
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw persons in RED
    for box in person_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin), "person", fill="red")
        
    # Draw shirts in BLUE
    for box in shirt_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=3)
        draw.text((xmin, ymin), "shirt", fill="blue")
        
    img.save(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    base_dir = "/data1/aminur/om/GRPO/official_viscot_gqa_10/images"
    in_img = os.path.join(base_dir, "2331819.jpg")
    out_img = os.path.join(base_dir, "2331819_detected.jpg")
    
    draw_and_save(in_img, out_img)
