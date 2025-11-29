import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

IMAGE_FOLDER = "mygo_image"
CSV_FILE = "description_AF3.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load CSV: expecting name + description columns
df = pd.read_csv(CSV_FILE)
songs = df["song name"].tolist()
descriptions = df["description"].tolist()

results = []

# Process each image
#TODO: CLIP Only takes max 77 tokens
for image_file in tqdm(os.listdir(IMAGE_FOLDER)):
    if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, image_file)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True).to(DEVICE)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    
    top_k = torch.topk(logits_per_image, 3)
    bottom_k = torch.topk(logits_per_image, 3, largest=False)
    
    results.append({
        "image": image_file,
        "top_3_songs": [songs[i] for i in top_k.indices[0].cpu()],
        "top_3_scores": [round(float(s), 4) for s in top_k.values[0].cpu()],
        "bottom_3_songs": [songs[i] for i in bottom_k.indices[0].cpu()],
        "bottom_3_scores": [round(float(s), 4) for s in bottom_k.values[0].cpu()]
    })

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv("clip_similarity_results.csv", index=False)

