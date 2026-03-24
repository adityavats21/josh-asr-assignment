import pandas as pd
import requests

# Load Excel
df = pd.read_excel("FT Data.xlsx")

# Take first row
row = df.iloc[0]

audio_url = row["rec_url_gcp"]

# Download audio
audio_data = requests.get(audio_url).content

# Save file
with open("sample.wav", "wb") as f:
    f.write(audio_data)

print("Audio downloaded!")