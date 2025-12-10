import requests
import dotenv
import os
import json
import tqdm
dotenv.load_dotenv()
url = "https://api.acedata.cloud/suno/audios"
SUNO_TOKEN=os.environ.get('SUNO_API_TOKEN')
print(SUNO_TOKEN)
headers = {
    "authorization": f"Bearer {SUNO_TOKEN}",
    "accept": "application/json",
    "content-type": "application/json"
}
with open('prompts.json', 'r') as f:
    dataset = json.load(f)
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