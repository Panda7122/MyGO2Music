import torch
import glob
import json
import os
import tqdm
from argparse import ArgumentParser
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

parser = ArgumentParser()
parser.add_argument("input_dir")
args = parser.parse_args()

INPUT_DIR = args.input_dir
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

input_paths = glob.glob(os.path.join(INPUT_DIR, "*"))
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Image
input_images = [Image.open(path) for path in input_paths]
IMG_COUNT = len(input_images)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text", 
                "text": "Describe the feeling of this image."
            },
        ],
    }
]

img_descriptions = []
for i in tqdm.trange(IMG_COUNT):
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=[text_prompt], images=input_images[i], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(DEVICE)

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    img_descriptions.append(output_texts[0])

result = []
for i in range(IMG_COUNT):
    result.append({
        "name": os.path.splitext(os.path.split(input_paths[i])[-1])[0],
        "description": img_descriptions[i]
    })

with open("vlm_result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)