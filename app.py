import os
import re
import sounddevice as sd
import soundfile as sf
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


DURATION = 5
SAMPLERATE = 44100
FILENAME = "file.wav"
CSV_PATH = "Data/boli_combined.csv"
LOG_FILE = "model_evolution/istoric_predictii.csv"
MODEL_NAME = "best_model_trained_model_v1_epoch_6"
# MODEL_NAME = "WebS_model_15_epoch_19"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modele_tokenizare", MODEL_NAME)
TOKENIZER_PATH = "distilgpt2"



load_dotenv()
api_key = os.getenv("API_KEY")
client = Groq(api_key=api_key)


df = pd.read_csv(CSV_PATH)
disease_dict = {
    row["Name"].strip().lower(): row["Symptoms"]
    for _, row in df.iterrows()
}


print("[DEBUG] Model path exists:", os.path.exists(MODEL_PATH))


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()


def record_audio():
    print(f" Recording for {DURATION} seconds...")
    recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
    sd.wait()
    sf.write(FILENAME, recording, SAMPLERATE)
    print(f"Saved File: {FILENAME}")

def transcribe_audio():
    with open(FILENAME, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(FILENAME, file.read()),
            model="distil-whisper-large-v3-en",
            response_format="json",
            language="en",
            temperature=0.0
        )
        text = transcription.text.strip().lower()
        print(f"[TRANSCRIPT]: {text}")
        return text

def detect_disease(text):
    clean_text = text.strip().lower().replace(".", "").replace(",", "")
    if clean_text in disease_dict:
        return clean_text
    for disease in disease_dict:
        if clean_text in disease or disease in clean_text:
            return disease
    return None

def generate_symptoms(disease):
    prompt = f"Generate symptoms for: {disease} |"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=80,
        num_return_sequences=1,
        do_sample=True,
        top_k=8,
        top_p=0.95,
        temperature=0.3,
        repetition_penalty=1.2
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = result.replace(prompt, "")
    result = re.sub(r"(disease\s*:\s*\w+\s*)?(symptoms?\s*:)?", "", result, flags=re.IGNORECASE)
    return result.strip()

def clean_symptoms(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\b(or|and|with|to|of|from)\b', '', text, flags=re.IGNORECASE)
    symptoms = [s.strip().lower() for s in text.split(",")]
    symptoms = list(dict.fromkeys([s for s in symptoms if len(s) > 3 and not s.isnumeric()]))
    return ", ".join(symptoms).capitalize()

def prediction_saved(model_name, intrebare, raspuns):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "timestamp": [timestamp],
        "model": [model_name],
        "input_disease": [intrebare],
        "generated_symptoms": [raspuns]
    }
    df_nou = pd.DataFrame(data)
    if not os.path.exists(LOG_FILE):
        df_nou.to_csv(LOG_FILE, index=False)
    else:
        df_nou.to_csv(LOG_FILE, mode="a", header=False, index=False)

if __name__ == "__main__":
    record_audio()
    text = transcribe_audio()

    disease = text.strip().lower()
    symptoms_raw = generate_symptoms(disease)
    symptoms_cleaned = clean_symptoms(symptoms_raw)
    print(f"\n Symptoms generated for '{disease}': {symptoms_cleaned}")
    prediction_saved(MODEL_NAME, disease, symptoms_cleaned)

