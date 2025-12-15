import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, CLIPTextModelWithProjection, AutoTokenizer

IMAGE_FOLDER = "mygo_image"
CSV_FILE = "description_AF3.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100
#dtype = 

# Load Long CLIP
model_id = ("zer0int/LongCLIP-GmP-ViT-L-14")
config = CLIPConfig.from_pretrained(model_id)
config.text_config.max_position_embeddings = 248
text_emb_model = CLIPTextModelWithProjection.from_pretrained(model_id, config=config.text_config).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id, config=config).to(DEVICE)
processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=248)


# Load CSV: expecting name + description columns
df = pd.read_csv(CSV_FILE)
songs = df["song name"].tolist()
descriptions = df["description"].tolist()

results = []
text_embs = []

# Get text embeddings
for i in range(0, len(descriptions), BATCH_SIZE):
    batch_descriptions = descriptions[i:i + BATCH_SIZE]
    inputs = tokenizer(batch_descriptions, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)
    outputs = text_emb_model(**inputs)
    batch_text_embs = outputs.text_embeds
    batch_text_embs /= batch_text_embs.norm(dim=-1, keepdim=True)
    text_embs.append(batch_text_embs)
    
    
text_embs = torch.cat(text_embs, dim=0).cpu()


# Process each image
for image_file in tqdm(os.listdir(IMAGE_FOLDER)):
    if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, image_file)
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model.get_image_features(**inputs)
        outputs /= outputs.norm(dim=-1, keepdim=True)
        outputs = outputs.cpu()
        
        logits_per_image = (outputs @ text_embs.T).squeeze(0)
        
        top_k = torch.topk(logits_per_image, 3)
        bottom_k = torch.topk(logits_per_image, 3, largest=False)
    
    results.append({
        "image": image_file,
        "top_3_songs": [songs[int(i)] for i in top_k.indices.detach()],
        "top_3_scores": [round(float(s), 4) for s in top_k.values.detach()],
        "bottom_3_songs": [songs[int(i)] for i in bottom_k.indices.detach()],
        "bottom_3_scores": [round(float(s), 4) for s in bottom_k.values.detach()]
    })
    del inputs, outputs, logits_per_image, top_k, bottom_k
    torch.cuda.empty_cache()

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv("clip_similarity_results_fma.csv", index=False)

