# MyGO2Music: Generate Music for Anime memes

## Minimum Hardware Requirements

- **CPU:** AMD Ryzen 5 7500 or equivalent and above (6 cores, 12 threads)
- **Memory:** 16GB RAM or more  
- **Storage:** At least 32GB (for datasets and models)  
- **GPU (Recommended):** NVIDIA RTX 3090 or above, at least 24GB VRAM, CUDA support  
- **Operating System:** Arch Linux x86_64(Linux 6.16.10-zen1-1-zen)
- **Python Version:** 3.10.10  

## Envirment set up

Python version: **3.10.10**

We recommend using a virtual environment for managing dependencies. To install required packages, run

```sh
pip install -r requirements.txt
```

If you want to use Suno as the music generater, set up the environment variable `SUNO_API_TOKEN`. For example:

```sh
export SUNO_API_TOKEN=<your api token>
```

## Data preparation

Get MyGO!!!!! images to play (images are saved to `mygo_image`):

```python
python get_mygo_image.py
```

We use [Free Music Archive](https://freemusicarchive.org) as our dataset to finetune our prompt generating LM. Feel free to try out other music datasets.

## Code

### stage 1

#### ALM+CLIP

To generate audio captions using Audio Language Model (ALM), run `python3 alm_retrieval/gen_text.py`:

```python
python alm_retrieval/gen_text.py \
    --audio_dir /path/to/audios \
    --output audio_captions.csv
```

Then, to calculate the similarity of image and audio captions using CLIP, run `python alm_retrieval/CLIP_cosine_sim.py`:

```python
python alm_retrieval/CLIP_cosine_sim.py \
    --image_dir /path/to/images \
    --desc_file audio_captions.csv \
    --output clip_similarity_results_fma.csv
```

#### VLM+CLAP(recommended)

First, run `sh get_clap_model.sh` to download the CLAP model.

To generate images captions using Visual Language Model (ALM), run `vlm_retrieval/vlm_inference.py`(the results will be saved to `vlm_result.json`):

```python
python vlm_retrieval/vlm_inference.py \
    --input_dir /path/to/images
```

Then, to calculate the similarity of image captions and audio using CLAP, run `vlm+clap_similarity.py`(the results will be saved to `CLAP_similarity_result_fma.json`):

```python
python vlm+clap_similarity.py \
    --vlm_result_path vlm_result.json \
    --audio_dir /path/to/audios
```

### stage 2

To generate music prompts with the baseline model([model link](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)):

```python
python prompt_gen_baseline.py
```

Run `python3 prompt_gen.py` to generate the prompt for all image captions generated in stage 1:

```python
python3 prompt_gen.py \
    --model_path ./lora_finetuned_model/ \
    --input_data vlm_result.json \
    --output_json prompts_finetune.json
```

## stage 3

To finetune the model, first run `python3 finetune_LM/get_train_data.py` to preset dataset:

```python
python3 finetune_LM/get_train_data.py \
    --method ['clip_similarity', 'clap_similarity'] \
    --output finetune_data.json
```

and run `python3 finetune_LM/lora_finetune.py` to finetune lm(it will save model to `./lora_finetuned_model/` by default):

```python
python3 finetune_LM/lora_finetune.py \
    --finetune_data finetune_data.json \
    --output_dir ./lora_finetuned_model/
```

Run `python3 prompt_gen.py` to generate the prompt for all image captions generated in stage 1:

```python
python3 prompt_gen.py \
    --model_path ./lora_finetuned_model/ \
    --input_data vlm_result.json \
    --output_json prompts_finetune.json
```

then run `python3 music_generator.py` to generate music with musicgen(if you want to use SUNO, replace "MusicGen" with SUNO):

```python
python3 music_generator.py \
    --input_json /path/to/prompts.json \
    --method MusicGen \
    --output ./music/finetuned/
```
