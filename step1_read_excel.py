import pandas as pd

df = pd.read_excel("FT Data.xlsx")

row = df.iloc[0]

print("Audio URL:", row["rec_url_gcp"])
print("Transcription URL:", row["transcription_url_gcp"])