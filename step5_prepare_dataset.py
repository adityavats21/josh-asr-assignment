import pandas as pd
import requests
import librosa
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

df = pd.read_excel("FT Data.xlsx")

dataset = []

for i in range(5):  # take first 5 samples
    print(f"\nProcessing sample {i}...")

    row = df.iloc[i]

    old_audio_url = row["rec_url_gcp"]
    parts = old_audio_url.split("/")
    folder = parts[-2]
    file_name = parts[-1].replace("_audio.wav", "")

    audio_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_audio.wav"
    transcription_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_transcription.json"

    audio_data = requests.get(audio_url).content
    audio_file = f"sample_{i}.wav"

    with open(audio_file, "wb") as f:
        f.write(audio_data)

    # -------- LOAD AUDIO --------
    audio, sr = librosa.load(audio_file, sr=16000)

    # -------- GET TEXT --------
    data = requests.get(transcription_url).json()
    text = data[0]["text"]

    # -------- PROCESS FOR WHISPER --------
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    labels = processor.tokenizer(text, return_tensors="pt").input_ids

    # -------- STORE --------
    dataset.append({
        "input_features": inputs.input_features,
        "labels": labels
    })

    print("Done sample", i)

print("\n Dataset ready!")
print("Total samples:", len(dataset))