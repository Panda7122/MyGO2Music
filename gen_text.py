import utils
import librosa
import os
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AudioFlamingo3ForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
from tqdm import tqdm
import pandas as pd

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

AUDIO_DIR = 'audio/'

audio_files = os.listdir(AUDIO_DIR)
audio_files = [os.path.join(AUDIO_DIR, a) for a in audio_files]

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

prompt = ("Describe the music in terms of emotion, atmosphere and vibes."
          "It is preferable if the description can also describe images. "
          "Don't include any headers like '1.', 'Emotions:', etc.")

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
df.to_csv("description_AF3.csv", index=True)
print("Done")
