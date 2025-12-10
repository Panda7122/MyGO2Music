import requests
import dotenv
import os
import json
import tqdm
import argparse
import scipy


def generate_SUNO(dataset):
    dotenv.load_dotenv()
    url = "https://api.acedata.cloud/suno/audios"
    SUNO_TOKEN=os.environ.get('SUNO_API_TOKEN')
    print(SUNO_TOKEN)
    headers = {
        "authorization": f"Bearer {SUNO_TOKEN}",
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    result_song = []
    for data in tqdm.tqdm(dataset):
        title = data['name']
        prompt = data['prompt']
        payload = {
            "action": "generate",
            "prompt": f"{prompt}",
            "model": "chirp-v4",
            "title": f"{title}",
            "vocal_gender": "f"
        }    


        response = requests.post(url, json=payload, headers=headers)
        print(response.status_code)
        print(response.text)
        result = response.text
        if result['success'] != True:
            tqdm.tqdm.write(f'Warning {title} generate failed')
            continue
        data = result['data'][0]
        audio_url = data['audio_url']
        lyric = data['lyric']
        filename = f"./music/baseline/{title}.mp3"
        if not os.path.exists("./music/"):
            os.makedirs("./music/")
        if not os.path.exists("./music/baseline/"):
            os.makedirs("./music/baseline/")
        response = requests.get(audio_url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"download success, save music at {filename}")
        else:
            print("download failed")
        result_song.append({
            "title": title,
            "file_location":filename,
            "lyric": lyric,
            "audio_url": audio_url
        })
    with open("song_data.json", "w") as f:
        json.dump(result_song, f, indent=2, ensure_ascii=False)
        
def generate_MusicGen(dataset):
    # MusicGen may have issues with long prompts
    model_id = "facebook/musicgen-medium"
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to("cuda")
    #model.config.max_length_seconds = 95
    if not os.path.exists("./music/"):
        os.makedirs("./music/")
    output_dir = "./music/baseline_musicgen/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data in tqdm.tqdm(dataset):
        title = data['name']
        prompt = data['prompt']
        inputs = processor(text=prompt, padding=True, return_tensors="pt").to("cuda")
        audio_values = model.generate(**inputs, do_sample=True)
        print(audio_values.shape)
        filename = os.path.join(output_dir, f"{title}.wav")
        scipy.io.wavfile.write(filename, model.config.audio_encoder.sampling_rate, audio_values[0, 0].cpu().numpy())      
        print(f"save music at {filename}")
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='prompts.json', help='input json file with music prompts')
    parser.add_argument('--method', '-m', type=str, default='SUNO', help='music generation method')
    args = parser.parse_args()
    
    with open(args.input_json, 'r') as f:
        dataset = json.load(f)
    
    if args.method == 'SUNO':
        generate_SUNO(dataset)
    elif args.method == 'MusicGen':
        generate_MusicGen(dataset)