# Hindi ASR Pipeline (Josh Talks Assignment)

This project implements an end-to-end Automatic Speech Recognition (ASR) pipeline for Hindi speech data as part of the Josh Talks AI/ML Engineer (Speech & Audio) Internship Assignment.

---

## Overview

The pipeline covers:
- Data preprocessing from raw audio and metadata
- Fine-tuning Whisper-small model
- Evaluation using Word Error Rate (WER)
- Error analysis and taxonomy
- Text cleanup (number normalization and English detection)
- Spelling correction system for large vocabulary
- Lattice-based ASR evaluation

---

## Key Features

### 1. Data Preprocessing
- Fixed incorrect GCP URLs
- Downloaded audio and transcription JSON files
- Converted audio to 16kHz mono format
- Structured dataset for training

---

### 2. Model Fine-Tuning
- Model: openai/whisper-small
- Framework: HuggingFace Transformers
- Training performed on a small subset (due to compute constraints)
- Observed decreasing loss across epochs

---

### 3. Evaluation (WER)
- Compared baseline and fine-tuned model
- Observed degradation due to overfitting on limited data
- Highlighted importance of sufficient training data

---

### 4. Error Analysis
Identified key error categories:
- Spelling errors
- Word repetition
- Substitutions
- Morphological errors
- Missing and extra words

---

### 5. Cleanup Pipeline
- Number normalization (Hindi words to digits)
- Handling of edge cases (idioms and expressions)
- Detection of English-origin words in Devanagari
- Tagging using [EN][/EN] markers

---

### 6. Spelling Correction System
- Processed approximately 1.75 lakh unique words
- Classified words into correct and incorrect categories
- Used heuristic and edit-distance based approach
- Assigned confidence levels (high / medium / low)

---

### 7. Lattice-Based ASR Evaluation
- Addresses limitations of standard WER
- Supports multiple valid transcription forms
- Uses bin-based alignment across model outputs
- Reduces unfair penalization

---

## Tech Stack

- Python
- HuggingFace Transformers
- Librosa
- Pandas
- JiWER

---

## Results Summary

| Metric | Value |
|------|------|
| Baseline WER | 0.83 |
| Fine-tuned WER | 3.83 |
| Correct Words | 5,027 |
| Incorrect Words | ~1.72 lakh |

---

## Repository Structure
.
├── step1_read_excel.py
├── step2_fetch_transcription.py
├── step3_download_audio.py
├── step4_preprocess.py
├── step5_prepare_dataset.py
├── step6_train_model.py
├── step7_evaluate.py
├── step8_cleanup.py
├── step9_spelling.py
├── README.md


---

## Notes

- Model weights and audio files are excluded due to size constraints
- Fine-tuning performed on a limited subset for demonstration purposes

---

## Author

Aditya Vats  
B.Tech CSE (2026), VIT  

---

## Additional Resources

Google Sheet (Spelling Classification):  
https://docs.google.com/spreadsheets/d/1wMX8P3iWf2d9_poZ4yrca8K_PDXMd1ygMajQnsqW1nM/edit?usp=sharing
