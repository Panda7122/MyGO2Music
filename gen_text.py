import utils
import librosa
#from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

AUDIO_DIR = 'data/fma_small/'
METADATA_DIR = 'data/fma_metadata/'
GENRES = ['Pop', 'Rock'] # see fma_metadata/genres.csv for info

tracks = utils.load('data/fma_metadata/tracks.csv')
tracks = tracks[tracks['set', 'subset'] <= 'small']
tracks = tracks[tracks['track', 'genre_top'] == 'Pop']
# tracks = tracks[tracks['track', 'genre_top'].isin(GENRES)]

test_filename = utils.get_audio_path(AUDIO_DIR, tracks.index[0])
print(test_filename)
""" DONT TRY THIS AT HOME
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)

prompt = "describe the music in terms of emotions, genre and atmosphere"
audio, sr = librosa.load(test_filename, sr=processor.feature_extractor.sampling_rate)
inputs = processor(text=prompt, audios=audio, return_tensors="pt")

generated_ids = model.generate(**inputs, max_length=256)
generated_ids = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)"""