import os
import glob
import json
import tqdm
import torch
import laion_clap
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--audio_dir", type=str, default='fma_small/')
parser.add_argument("--vlm_result_path", type=str, default='vlm_retrieval/vlm_result.json')
args = parser.parse_args()


AUDIO_DIR = args.audio_dir
VLM_RESULT_PATH = args.vlm_result_path
AUDIO_EXTENSIONS = ('.mp3', '.wav')

audio_paths = []
audio_names = []
for dirpath, dirnames, filenames in os.walk(AUDIO_DIR):
    for f in filenames:
        file_path = os.path.join(dirpath, f)
        if os.path.isfile(file_path) and file_path.lower().endswith(AUDIO_EXTENSIONS):
            audio_paths.append(file_path)

audio_paths = sorted(audio_paths)
audio_names = [os.path.basename(path) for path in audio_paths]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt("./model/630k-audioset-best.pt", verbose=False)
model = model.to(DEVICE)
model.eval()
print(f"clap_model device: {next(model.parameters()).device}")

with open(VLM_RESULT_PATH, "r") as f:
    vlm_result = json.load(f)

img_names = [item["name"] for item in vlm_result]
img_descriptions = [item["description"] for item in vlm_result]

with torch.no_grad():
    audio_embed = model.get_audio_embedding_from_filelist(audio_paths, use_tensor=True).to(DEVICE)
    texts_embed = model.get_text_embedding(img_descriptions, use_tensor=True).to(DEVICE)
    audio_embed = F.normalize(audio_embed, dim=1)
    texts_embed = F.normalize(texts_embed, dim=1)

print("audio_embed shape:", audio_embed.shape)
print("texts_embed shape:", texts_embed.shape)


result = []
for i in tqdm.trange(len(img_names), desc="Computing similarity"):
    embed = texts_embed[i]
    embed_repeat = embed.unsqueeze(0).repeat(len(audio_embed), 1)
    sim = F.cosine_similarity(embed_repeat, audio_embed, dim=1)

    
    top_3_values, top_3_idx = torch.topk(sim, 3)

    result_item = {
        "img_name": img_names[i],
        "top_3_songs": [audio_names[idx] for idx in top_3_idx],
        "top_3_scores": [round(val.item(), 4) for val in top_3_values],
    }
    result.append(result_item)

with open("CLAP_similarity_result_fma.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)