import os
import json
import requests
import time

base_dir = "/data1/aminur/om/GRPO/official_viscot_gqa_10"
img_dir = os.path.join(base_dir, "images")
os.makedirs(img_dir, exist_ok=True)

input_file = "/data1/aminur/om/GRPO/Visual-CoT-Repo/viscot_dataset/gqa_cot_train.jsonl"

samples = []
with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        samples.append(json.loads(line.strip()))

# Try direct URLs since the VG API returned 403
for sample in samples:
    img_filename = sample["image"]
    img_path = os.path.join(img_dir, img_filename)
    
    print(f"Downloading {img_filename}...")
    # Try both VG_100K and VG_100K_2
    url1 = f"https://cs.stanford.edu/people/rak248/VG_100K/{img_filename}"
    url2 = f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_filename}"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    resp = requests.get(url1, headers=headers)
    if resp.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(resp.content)
        print("Success from VG_100K")
    else:
        resp = requests.get(url2, headers=headers)
        if resp.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(resp.content)
            print("Success from VG_100K_2")
        else:
            print(f"Failed to download {img_filename}")
            
    time.sleep(0.5)

print("\nDone downloading images!")
