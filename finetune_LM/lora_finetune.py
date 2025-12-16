import json
from argparse import ArgumentParser
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

parser = ArgumentParser()
parser.add_argument("vlm_description", type=str)
parser.add_argument("prompts_file", type=str)
args = parser.parse_args()

# create dataset
with open(args.vlm_description, "r") as f:
    vlm_description = json.load(f)
img_names = [item["name"] for item in vlm_description]
img_descriptions = [item["description"] for item in vlm_description]

with open(args.prompts_file, "r") as f:
    prompts_file = json.load(f)
img_prompts = [item["prompt"] for item in prompts_file]


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype="bfloat16"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

model = get_peft_model(model, lora_config)

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

item = {
    "instruction": "You are a creative AI Music Composer. Your task is to convert an image description into a specific AUDIO/MUSIC prompt.",
    "input": "Image description:",
    "output": ""
}

training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()