import json
import pandas as pd
import ast
import os

clip_similarity_result = pd.read_csv("clip_similarity_results_fma.csv")
fma_description_result = pd.read_csv("alm_retrieval/description_fma_AF3.csv")

with open("vlm_retrieval/vlm_result.json", "r") as f:
    vlm_result = json.load(f)
img_to_description = {item["name"]: item["description"] for item in vlm_result}

imgs = clip_similarity_result["image"].tolist()
imgs = [os.path.splitext(name)[0] for name in imgs]
top_3_songs = clip_similarity_result["top_3_songs"].tolist()
top_1_songs = [ast.literal_eval(item)[0] for item in top_3_songs]

fma_song_name = fma_description_result["song name"].tolist()
fma_descriptions = fma_description_result["description"].tolist()

fma_name_to_desc = {key: value for key, value in zip(fma_song_name, fma_descriptions)}

data = []
for i in range(len(imgs)):
    item = {
        "img_name": imgs[i],
        "img_description": img_to_description[imgs[i]],
        "song_name": top_1_songs[i],
        "song_description": fma_name_to_desc[top_1_songs[i]]
    }
    data.append(item)

with open("finetune_LM/finetune_data.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)