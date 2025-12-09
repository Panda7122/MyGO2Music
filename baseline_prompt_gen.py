import torch
import glob
import json
import os
import tqdm
from argparse import ArgumentParser
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

with open('vlm_retrieval/vlm_result.json', 'r') as f:
    dataset = json.load(f)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": ("You are a music generation expert. Based on the following image description, generate a detailed music prompt for a funny and humorous song. "
                         "Output only the music prompt as plain text, without any additional formatting. "
                        "The prompt should prioritize: humor and comedic tone, matching the image's funny elements, genre, BPM, mood, timbre, rhythm, melody, key/tonality, and atmosphere. "
                        "Be specific and descriptive to help generate the most matching and entertaining song. \n\nImage description:")
            },
            {
                "type": "text"
            }
        ],
    }
]
result = []
for data in tqdm.tqdm(dataset):
    name = data['name']
    description = data['description']
    conversation[0]["content"][1]['text']=description
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=[text_prompt], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(DEVICE)

    # Inference: Generation of the output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        min_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    prompt = output_texts[0]
    # Basic sanity checks: ensure non-trivial length and avoid single-word repetition
    def is_bad(text: str) -> bool:
        t = text.strip()
        if len(t.split()) < 12:
            return True
        # check repeated single token dominating output
        tokens = t.split()
        most = max({w: tokens.count(w) for w in set(tokens)}, key=lambda k: tokens.count(k))
        if tokens.count(most) / max(1, len(tokens)) > 0.6:
            return True
        return False

    # If bad, attempt one regeneration with stronger constraints
    if is_bad(prompt):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=320,
            min_new_tokens=96,
            do_sample=True,
            temperature=0.9,
            top_p=0.92,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        prompt = output_texts[0]
    tqdm.tqdm.write(f"{name}: {prompt}")
    result.append(
        {
            'name':name,
            'prompt':prompt
        }
    )
with open("prompts.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)