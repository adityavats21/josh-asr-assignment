import requests
import json

def load_transcription(url):
    response = requests.get(url)
    return response.json()

url = "PUT_ONE_TRANSCRIPTION_URL_HERE"
data = load_transcription(url)

print(data)