import pandas as pd
import requests

df = pd.read_excel("FT Data.xlsx")
row = df.iloc[0]

audio_url = row["rec_url_gcp"]

parts = audio_url.split("/")
folder = parts[-2]
file_name = parts[-1].replace("_audio.wav", "")

transcription_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_transcription.json"

print("Fixed URL:", transcription_url)

response = requests.get(transcription_url)

data = response.json()

print("First line:", data[0]["text"])