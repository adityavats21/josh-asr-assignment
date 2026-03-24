import pandas as pd
import requests
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer

# -------- LOAD MODEL --------
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()

# -------- LOAD DATA --------
df = pd.read_excel("FT Data.xlsx")

predictions = []
references = []

for i in range(25):
    print(f"\nEvaluating sample {i}...")

    row = df.iloc[i]

    # FIX URL
    old_audio_url = row["rec_url_gcp"]
    parts = old_audio_url.split("/")
    folder = parts[-2]
    file_name = parts[-1].replace("_audio.wav", "")

    audio_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_audio.wav"
    transcription_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_transcription.json"

    # AUDIO
    audio_data = requests.get(audio_url).content
    with open("temp.wav", "wb") as f:
        f.write(audio_data)

    audio, sr = librosa.load("temp.wav", sr=16000)

    # REFERENCE TEXT
    data = requests.get(transcription_url).json()
    ref_text = data[0]["text"]

    # PREDICTION
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    predicted_ids = model.generate(
        inputs.input_features,
        language="hi",
        task="transcribe"
    )
    pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("REF:", ref_text[:50])
    print("PRED:", pred_text[:50])

    if pred_text != ref_text:
        print("\n--- ERROR SAMPLE ---")
        print("REF FULL:", ref_text)
        print("PRED FULL:", pred_text)

    predictions.append(pred_text)
    references.append(ref_text)

# -------- WER --------
final_wer = wer(references, predictions)

print("\nFINAL WER:", final_wer)