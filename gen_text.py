import utils
import librosa
import os
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
from tqdm import tqdm
import pandas as pd

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

AUDIO_DIR = './'
AUDIO_EXTENSIONS = ('.mp3', '.wav')

audio_files = []
for item in os.listdir(AUDIO_DIR):
    item_path = os.path.join(AUDIO_DIR, item)
    print(item_path)
    if os.path.isfile(item_path) and item_path.lower().endswith(AUDIO_EXTENSIONS):
        audio_files.append(item_path)
    elif os.path.isdir(item_path):
        for sub_item in os.listdir(item_path):
            sub_item_path = os.path.join(item_path, sub_item)
            if os.path.isfile(sub_item_path) and sub_item_path.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(sub_item_path)
                
# model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,
#                                                            trust_remote_code=True, 
#                                                            quantization_config=bnb_config,
#                                                            device_map='auto')
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)

model = AudioFlamingo3ForConditionalGeneration.from_pretrained("nvidia/audio-flamingo-3-hf" ,
                                                            quantization_config=bnb_config,
                                                            torch_dtype=torch.float16,
                                                            device_map='auto')
processor = AutoProcessor.from_pretrained("nvidia/audio-flamingo-3-hf")

prompt = ("Describe the music only through its emotional tone, atmosphere, and the imagery it evokes."
          "Focus on mood, energy, emotional color, and sensory impressions."
          "DO NOT naming genres, styles, instruments, cultural references, or any terms related to specific regions or identities."
          "Write one natural descriptive paragraph without section titles or lists.")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": ''}
        ]
    }
]

result = []
for file in tqdm(audio_files, desc='Audio files'):
    try:
        messages[0]["content"][1]["path"] = file
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device, dtype=torch.float16)   
            
        generated_ids = model.generate(**inputs, max_new_tokens=248)
        generated_ids = generated_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        result.append({'song name': os.path.basename(file), 'description': response})
    except Exception as e:
        print(f"Failed handling {file}: {e}")
df = pd.DataFrame(result)
df.to_csv(f"description_fma_AF3.csv", index=True)
print("Done")
