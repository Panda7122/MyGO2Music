# MyGO2Music: Generate Music for Anime memes
[FMA]: 

## Minimum Hardware Requirements

- **CPU:** AMD Ryzen 5 7500 or equivalent and above (6 cores, 12 threads)
- **Memory:** 16GB RAM or more  
- **Storage:** At least 32GB (for datasets and models)  
- **GPU (Recommended):** NVIDIA RTX 3090 or above, at least 24GB VRAM, CUDA support  
- **Operating System:** Arch Linux x86_64(Linux 6.16.10-zen1-1-zen)
- **Python Version:** 3.10.10  
- 
## Envirment set up

we use python version **3.10.10**
please prepare a virtual envirment for this python version
and use `pip3 install -r requirements.txt` and `pip3 install -r ./vlm_retrieval/requirements.txt` for build up the envirment
if you want to use suno as music generater, please set up `.env` file looks like

```
SUNO_API_TOKEN=<your api token>
```
## Data preparation
Get MyGO!!!!! images to play (images are saved to `mygo_image`):
```
python get_mygo_image.py
```
We use [FMA]: https://freemusicarchive.org to finetune our prompt generating LM

## stage 1
### ALM+CLIP
run `python3 ./alm_retrieval/gen_text.py` for generate alm result in `alm_retrieval/description_fma_AF3.csv`

run `python3 CLIP_cosine_sim.py` for calculate the similarity of image and audio caption, and result will save to `clip_similarity_results_fma.csv`

### VLM+CLAP
please run `sh get_clap_model.sh` for getting CLAP model

run `python3 ./vlm_retrieval/vlm_inference.py --input_dir <your image set path>` for generate vlm result

run `python3 ./vlm+clap_similarity.py --vlm_result_path <your vlm result path> --audio_dir <your audio set path>` for calculate the similarity of image caption and audio

## stage 2
for run the baseline model

please run `python3 baseline_prompt_gen.py` for generate the prompt for all image caption that generate in stage 1

then run `python3 music_generator.py --input_json prompt.json --method MusicGen --output ./music/baseline/` for generate with musicgen(if you want to use SUNO, replace "MusicGen" to SUNO)

## stage 3

for finetune the model
please run `python3 finetune_LM/get_train_data.py` to preset dataset
and run `python3 finetune_LM/lora_finetune.py` to finetune lm(it will save model to `./lora_finetuned_model/`)
please run `python3 prompt_gen.py` for generate the prompt for all image caption that generate in stage 1

then run `python3 music_generator.py --input_json prompts_finetune.json --method MusicGen --output ./music/finetune/` for generate with musicgen(if you want to use SUNO, replace "MusicGen" to SUNO)