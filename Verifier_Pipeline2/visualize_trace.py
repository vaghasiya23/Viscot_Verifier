"""
visualize_trace.py - Generates visual audit folders for verification traces.

For each sample, it saves:
  1. annotated_image.jpg  - Full image with detected bounding boxes & GT box overlaid
  2. crop_*.jpg           - Individual cropped patches inspected by the verifier
  3. verification_code.py - The exact Python code generated
  4. verdict.json         - Step-by-step pass/fail results & completion ratio

Usage:
  python visualize_trace.py                       # Visualizes first 5 golden traces
  python visualize_trace.py --sample_idx 0        # Visualizes sample #0
  python visualize_trace.py --source rejected    # Visualizes rejected/failed traces
"""
import os
import sys
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tools import ImagePatch

# Bright distinct colors for different classes
PALETTE = [
    "#00E5FF",  # Cyan
    "#FF1744",  # Red
    "#76FF03",  # Bright Green
    "#D500F9",  # Purple
    "#FF9100",  # Orange
    "#FFEA00",  # Yellow
]


def draw_labeled_box(draw: ImageDraw.ImageDraw, box, label: str, color: str, width: int = 3):
    xmin, ymin, xmax, ymax = box
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)
    
    # Text background tag
    text = f" {label} "
    draw.rectangle([xmin, max(0, ymin - 16), xmin + len(text) * 8, ymin], fill=color)
    draw.text((xmin + 2, max(0, ymin - 15)), text, fill="black")


def visualize_sample(record: dict, output_base_dir: str):
    idx = record.get("sample_index", 0)
    verdict = record.get("execution_result", {}).get("verdict", "UNKNOWN")
    image_name = record.get("image", "")
    image_path = os.path.join(ROOT, "data", "raw", "images", image_name)

    if not os.path.exists(image_path):
        print(f"[-] Image not found: {image_path}")
        return

    sample_dir = os.path.join(output_base_dir, f"sample_{idx}_{verdict}")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Save Code and Verdict
    with open(os.path.join(sample_dir, "verification_code.py"), "w") as f:
        f.write(record.get("verification_code", ""))

    with open(os.path.join(sample_dir, "verdict.json"), "w") as f:
        json.dump(record, f, indent=2)

    # 2. Extract objects from question & thought to detect and visualize
    img_patch = ImagePatch(image_path)
    with Image.open(image_path).convert("RGB") as base_img:
        draw = ImageDraw.Draw(base_img)
        
        # Parse targets from thought or question
        search_terms = []
        answer = record.get("answer", "")
        if answer:
            search_terms.append(answer)

        # Extract words from steps
        for step in record.get("execution_result", {}).get("steps", []):
            claim = step.get("claim", "")
            for word in claim.split():
                clean_word = word.strip(",.!?\"'").lower()
                if clean_word in ["chair", "sofa", "cup", "cups", "cabinet", "table", "person", "dog", "cat", "car", "bench", "rock", "bird"]:
                    if clean_word not in search_terms:
                        search_terms.append(clean_word)

        crop_idx = 0
        for i, term in enumerate(search_terms):
            color = PALETTE[i % len(PALETTE)]
            detections = img_patch.find(term)
            for j, det in enumerate(detections):
                draw_labeled_box(
                    draw,
                    det.box,
                    f"{term} ({det.horizontal_center:.0f},{det.vertical_center:.0f})",
                    color=color,
                    width=3
                )

                # Save cropped patch
                try:
                    xmin, ymin, xmax, ymax = [int(v) for v in det.box]
                    if xmax > xmin and ymax > ymin:
                        with Image.open(image_path).convert("RGB") as orig:
                            crop = orig.crop((xmin, ymin, xmax, ymax))
                            crop.save(os.path.join(sample_dir, f"crop_{crop_idx}_{term}.jpg"))
                            crop_idx += 1
                except Exception:
                    pass

        # Draw Ground Truth box if present in raw sample
        try:
            with open(os.path.join(ROOT, "data", "raw", "gqa_300_samples.json")) as f:
                raw_samples = json.load(f)
            if idx < len(raw_samples) and "bboxs" in raw_samples[idx]:
                for gt_box in raw_samples[idx]["bboxs"]:
                    draw_labeled_box(draw, gt_box, "GROUND TRUTH", color="#FFFFFF", width=2)
        except Exception:
            pass

        # Save the full annotated image
        annotated_path = os.path.join(sample_dir, "annotated_image.jpg")
        base_img.save(annotated_path, quality=95)

    print(f"[+] Created visualization at: {sample_dir}")
    print(f"    - Full overlay: {annotated_path}")
    print(f"    - Crops saved:  {crop_idx}")


def main():
    parser = argparse.ArgumentParser(description="Save full visual audit folders for verifier traces")
    parser.add_argument("--source", choices=["golden", "rejected"], default="golden", help="Trace source to visualize")
    parser.add_argument("--sample_idx", type=int, default=None, help="Specific sample index to visualize")
    parser.add_argument("--max_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--out_dir", type=str, default="data/visualizations", help="Output directory")
    args = parser.parse_args()

    input_file = os.path.join(ROOT, "data", "sft", f"{args.source}_traces.jsonl")
    if not os.path.exists(input_file):
        print(f"[-] File not found: {input_file}")
        sys.exit(1)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.sample_idx is not None:
        records = [r for r in records if r.get("sample_index") == args.sample_idx]

    records = records[:args.max_samples]
    print(f"\nVisualizing {len(records)} trace(s) to {out_dir}...\n")
    for r in records:
        visualize_sample(r, out_dir)

    print(f"\nDone! Open {out_dir} to see all annotated images and crops.\n")


if __name__ == "__main__":
    main()
