"""
tests/test_tools.py - Unit test for tools and ImagePatch API.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import ImagePatch

def test_image_patch():
    img_path = os.path.join(ROOT, "data", "raw", "images", "2410353.jpg")
    assert os.path.exists(img_path), f"Image missing: {img_path}"

    print(f"Loading ImagePatch: {img_path}")
    img = ImagePatch(img_path)
    print(f"Root patch size: {img.width}x{img.height}, area={img.area}")
    assert img.width == 500 and img.height == 471

    # 1. Detection test
    print("Testing find('chair')...")
    chairs = img.find("chair")
    print(f"Found {len(chairs)} chair(s): {chairs}")
    assert len(chairs) > 0, "Expected at least one chair"

    # 2. Spatial coordinates test
    c = chairs[0]
    print(f"Chair center: ({c.horizontal_center:.1f}, {c.vertical_center:.1f})")
    assert c.horizontal_center > 0 and c.vertical_center > 0

    # 3. Detection of furniture / sofa
    print("Testing find('sofa')...")
    sofas = img.find("sofa")
    print(f"Found {len(sofas)} sofa(s): {sofas}")

    print("\n--- All tool tests passed successfully! ---")

if __name__ == "__main__":
    test_image_patch()
