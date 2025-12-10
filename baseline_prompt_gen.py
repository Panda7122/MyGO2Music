import torch
import json
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

with open('vlm_retrieval/vlm_result.json', 'r') as f:
    dataset = json.load(f)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507").to(DEVICE)

system_instruction = """You are a creative AI Music Composer. 
Your task is to convert an image description into a specific AUDIO/MUSIC prompt.
Do NOT describe what is in the image. instead, describe the SOUND, GENRE, and MOOD that fits the image.
Make it funny and entertaining. Keep it under 40 words."""

# 範例 (Few-Shot)
example_1_input = "Image description: A fat orange cat sleeping on a computer keyboard."
example_1_output = "A lazy, comedic jazz tune with a slow tuba bassline. The rhythm is clumsy and sleepy, perfect for a heavy cat napping on keys. Lo-fi style."
result = []
for data in tqdm.tqdm(dataset):
    name = data['name']
    description = data['description']
    conversation = [
        {
            "role": "user",
            "content": f"""{system_instruction}

Example:
Input: {example_1_input}
Output: {example_1_output}

Task:
Input: Image description: {description}
Output:"""
        }
    ]
    text_prompt = tokenizer.apply_chat_template(conversation, 
                                                add_generation_prompt=True, 
                                                tokenize=True,
                                                return_dict=True,
                                                return_tensors="pt").to(DEVICE)

    # Inference: Generation of the output
    prompt = ""
    while len(prompt) == 0:
        output_ids = model.generate(
            **text_prompt,
            max_new_tokens=80,
            min_new_tokens=10,
            do_sample=True,    
            temperature=0.7,
            repetition_penalty=1.1
        )

        prompt = tokenizer.decode(output_ids[0][text_prompt["input_ids"].shape[-1]:], skip_special_tokens=True)
        last_punct = max(prompt.rfind('.'), prompt.rfind('!'), prompt.rfind('?'))
        if last_punct != -1:
            prompt = prompt[:last_punct+1]
        if len(prompt) > 200:
            prompt = prompt[:200]
            last_punct = max(prompt.rfind('.'), prompt.rfind('!'), prompt.rfind('?'))
            prompt = prompt[:last_punct+1]
    tqdm.tqdm.write(f"{name}: {prompt}")
    result.append(
        {
            'name':name,
            'prompt':prompt
        }
    )
with open("prompts.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)