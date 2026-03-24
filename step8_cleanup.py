import re
import pandas as pd
import requests

number_map = {
    "शून्य": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "तीस": 30,
    "चालीस": 40,
    "पचास": 50,
    "साठ": 60,
    "सत्तर": 70,
    "अस्सी": 80,
    "नब्बे": 90,
    "सौ": 100,
    "हज़ार": 1000
}

def normalize_numbers(text):
    words = text.split()
    result = []
    temp_num = 0

    for w in words:
        if w in number_map:
            val = number_map[w]

            # handle compound numbers
            if val == 100 or val == 1000:
                if temp_num == 0:
                    temp_num = 1
                temp_num *= val
            else:
                temp_num += val
        else:
            if temp_num != 0:
                result.append(str(temp_num))
                temp_num = 0
            result.append(w)

    if temp_num != 0:
        result.append(str(temp_num))

    return " ".join(result)


def should_skip_number_conversion(text):
    # phrases like "दो-चार बातें"
    if re.search(r"\w+-\w+", text):
        return True
    return False


english_hint_words = [
    "इंटरव्यू", "जॉब", "कंप्यूटर", "मोबाइल", "फोन",
    "स्कूल", "कॉलेज", "ट्रेन", "बस", "मीटिंग"
]

def detect_english_words(text):
    words = text.split()
    detected = []

    for w in words:
        if w in english_hint_words:
            detected.append(w)

    return detected

def tag_english_words(text):
    words = text.split()
    tagged = []

    for w in words:
        if w in english_hint_words:
            tagged.append(f"[EN]{w}[/EN]")
        else:
            tagged.append(w)

    return " ".join(tagged)

df = pd.read_excel("FT Data.xlsx")


print("\n===== CLEANUP PIPELINE OUTPUT =====\n")

for i in range(5):  
    print(f"\nSample {i} ------------------------")

    row = df.iloc[i]

    old_audio_url = row["rec_url_gcp"]
    parts = old_audio_url.split("/")
    folder = parts[-2]
    file_name = parts[-1].replace("_audio.wav", "")

    transcription_url = f"https://storage.googleapis.com/upload_goai/{folder}/{file_name}_transcription.json"

    try:
        data = requests.get(transcription_url).json()
        text = data[0]["text"]
    except:
        print("Error fetching transcription")
        continue

    print("ORIGINAL:", text)

    # EDGE CASE CHECK
    if should_skip_number_conversion(text):
        normalized = text
        print("⚠️ Skipped number normalization (edge case)")
    else:
        normalized = normalize_numbers(text)

    print("NORMALIZED:", normalized)

    # ENGLISH DETECTION
    detected = detect_english_words(text)
    tagged = tag_english_words(text)

    print("ENGLISH WORDS:", detected)
    print("TAGGED:", tagged)

print("\n PIPELINE COMPLETE")