import pandas as pd
from rapidfuzz.distance import Levenshtein

# -------- LOAD UNIQUE WORDS FILE --------
df = pd.read_excel("Unique Words Data.xlsx")

# assuming column name is 'word'
words = df.iloc[:, 0].astype(str).dropna().tolist()
words = [w.strip() for w in words if len(w.strip()) > 0]

common_words = set(words[:5000])  # assume first few are common

# -------- CLASSIFICATION FUNCTION --------
def classify_word(word):
    
    # HIGH confidence correct
    if word in common_words:
        return "correct", "high", "frequent/common word"
    
    # MEDIUM: similar to common word
    for cw in common_words:
        if Levenshtein.distance(word, cw) <= 1:
            return "incorrect", "medium", f"similar to {cw}"
    
    # LOW confidence
    return "incorrect", "low", "rare/unseen word"

results = []

for w in words: 
    label, confidence, reason = classify_word(w)
    
    results.append({
        "word": w,
        "label": label,
        "confidence": confidence,
        "reason": reason
    })

df_out = pd.DataFrame(results)
df_out.to_csv("q3_output.csv", index=False)

print("Done! Saved q3_output.csv")