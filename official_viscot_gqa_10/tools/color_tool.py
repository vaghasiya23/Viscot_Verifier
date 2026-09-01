from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

def extract_dominant_color(image_path: str, bbox: list = None) -> list:
    """
    Extracts the dominant RGB color from an image or bounding box.
    Returns the [R, G, B] value.
    """
    image = Image.open(image_path).convert("RGB")
    
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        image = image.crop((xmin, ymin, xmax, ymax))
        
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    
    # We use KMeans to find the most dominant cluster of pixels
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Find the cluster with the most pixels
    labels = list(kmeans.labels_)
    dominant_cluster = max(set(labels), key=labels.count)
    dominant_rgb = kmeans.cluster_centers_[dominant_cluster]
    
    return [int(c) for c in dominant_rgb]

if __name__ == "__main__":
    img_path = "/data1/aminur/om/GRPO/official_viscot_gqa_10/images/2331819.jpg"
    print("\n[Sandbox-Color] Extracting dominant color from full image...")
    rgb = extract_dominant_color(img_path)
    print(f"   -> Dominant RGB: {rgb}")
    
    # Let's crop to a small area where the person is
    bbox = [214, 0, 433, 374]
    print(f"\n[Sandbox-Color] Extracting dominant color from bounding box {bbox}...")
    rgb_crop = extract_dominant_color(img_path, bbox=bbox)
    print(f"   -> Dominant RGB: {rgb_crop}")
