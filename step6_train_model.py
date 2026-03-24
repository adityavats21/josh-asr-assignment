import pandas as pd
import requests
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ---------------- LOAD MODEL ----------------
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.train()

print("Model loaded ")

# ---------------- LOAD DATA ----------------
df = pd.read_excel("FT Data.xlsx")

dataset = []

for i in range(5):  # small dataset for now
    print(f"\nProcessing sample {i}...")

    row = df.iloc[i]

    # -------- FIX AUDIO URL --------
    old_audio_url = row["rec_url_gcp"]
    parts = old_audio_url.split("/")
    folder = parts[-2]
    file_name = parts[-1].replace("_audio.wav", "")

    audio_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_audio.wav"
    transcription_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_transcription.json"

    # -------- DOWNLOAD AUDIO --------
    audio_data = requests.get(audio_url).content
    audio_file = f"sample_{i}.wav"

    with open(audio_file, "wb") as f:
        f.write(audio_data)

    # -------- LOAD AUDIO --------
    audio, sr = librosa.load(audio_file, sr=16000)

    # -------- GET TEXT --------
    data = requests.get(transcription_url).json()
    text = data[0]["text"]

    # -------- PROCESS --------
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    labels = processor.tokenizer(text, return_tensors="pt").input_ids

    dataset.append({
        "input_features": inputs.input_features,
        "labels": labels
    })

    print("Done sample", i)

print("\n Dataset ready!")
print("Total samples:", len(dataset))

# ---------------- TRAINING ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(2):  
    print(f"\n Epoch {epoch}")

    for i, sample in enumerate(dataset):
        input_features = sample["input_features"]
        labels = sample["labels"]

        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Sample {i} Loss:", loss.item())

# ---------------- SAVE MODEL ----------------
model.save_pretrained("fine_tuned_model")
processor.save_pretrained("fine_tuned_model")

print("\n Model training complete & saved!")