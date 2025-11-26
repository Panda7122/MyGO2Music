import utils
import librosa
import os
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
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
print(audio_files[0])

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,
                                                           trust_remote_code=True, 
                                                           quantization_config=bnb_config,
                                                           device_map='auto')
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)

print(model.device)
# Example usage inferencing 1 audio
prompt = "What do you hear from this music? Describe in terms of emotion and atmosphere"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio"},
            {"type": "text", "text": prompt}
        ]
    }
]

result = []
for file in tqdm(audio_files, desc='Audio files'):
    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        audio, sr = librosa.load(file, sr=processor.feature_extractor.sampling_rate)
        audio = audio[:30*sr]
        inputs = processor(text=text, audio=audio, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generated_ids = model.generate(**inputs, max_length=4096)
        generated_ids = generated_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        result.append({'song name': os.path.basename(file), 'description': response})
    except Exception as e:
        print(f"Failed handling {file}: {e}")
df = pd.DataFrame(result)
df.to_csv("audio_summaries.csv", index=True)
print("Done")