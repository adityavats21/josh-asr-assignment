import pandas as pd
import requests
import librosa

df = pd.read_excel("FT Data.xlsx")
row = df.iloc[0]

old_audio_url = row["rec_url_gcp"]

parts = old_audio_url.split("/")
folder = parts[-2]
file_name = parts[-1].replace("_audio.wav", "")

audio_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_audio.wav"

print("Fixed Audio URL:", audio_url)

# -------- DOWNLOAD AUDIO --------
response = requests.get(audio_url)

print("Status Code:", response.status_code)
print("Content-Type:", response.headers.get("Content-Type"))

with open("sample.wav", "wb") as f:
    f.write(response.content)

# -------- LOAD AUDIO --------
audio, sr = librosa.load("sample.wav", sr=16000)

print("SUCCESS ")
print("Sample rate:", sr)