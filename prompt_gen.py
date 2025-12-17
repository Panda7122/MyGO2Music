import json
import torch
import peft
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype="bfloat16"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = peft.PeftModel.from_pretrained(model, './lora_finetuned_model/checkpoint-237').to(DEVICE)
with open("vlm_retrieval/vlm_result.json", "r") as f:
    datas = json.load(f)

system_instruction = """You are a creative AI Music Composer. 
Your task is to convert an image description into a specific AUDIO/MUSIC prompt."""

result = []
for i in tqdm(range(len(datas))):
    name = datas[i]['name']
    description = datas[i]['description']
    item = {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Image description: {datas[i]['description']}"},
        ]
    }
    
    # Format messages for model input
    prompt = tokenizer.apply_chat_template(
        item["messages"],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        top_p=0.9
    )
    
    # Decode and display output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[(len(system_instruction)+len(f"\nsystem\nuser\nImage description: {datas[i]['description']}\nassistant\n")):]
    tqdm.write(f"{name}: {response}")
    result.append(
        {
            'name':name,
            'prompt':response
        }
    )
with open("prompts_finetune.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)