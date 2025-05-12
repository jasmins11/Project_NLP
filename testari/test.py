from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import pandas as pd
from datetime import datetime
import re


LOG_FILE = "evolutia_modelului/istoric_predictii.csv"
MODEL_NAME = "WebS_model__epoch_17"

MODEL_PATH = f"modele_tokenizare/{MODEL_NAME}"
TOKENIZER_PATH = "modele_tokenizare/WebS_tokenizare__epoch_17"


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()


def genereaza_simptome(boala, max_new_tokens=80, temperature=0.8):
    prompt = f"Generate symptoms for: {boala} |"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
           input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=70,
            do_sample=True,
            temperature=0.5,
            top_k=50,
            repetition_penalty=1.8,
            pad_token_id=tokenizer.pad_token_id
    )

    rezultat = tokenizer.decode(outputs[0], skip_special_tokens=True)
   
    rezultat = rezultat.replace(prompt, "")
    rezultat = re.sub(r"(disease\s*:\s*\w+\s*)?(symptoms?\s*:)?", "", rezultat, flags=re.IGNORECASE)
    return rezultat.strip()


def curata_simptome(text):
    text = re.sub(r'\(.*?\)', '', text) 
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\b(or|and|with|to|of|from)\b', '', text, flags=re.IGNORECASE)

    simptome = [s.strip().lower() for s in text.split(",")]
    simptome = list(dict.fromkeys([s for s in simptome if len(s) > 3 and not s.isnumeric()]))
    return ", ".join(simptome).capitalize()


def salveaza_predictie(model_name, intrebare, raspuns):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "timestamp": [timestamp],
        "model": [model_name],
        "boala_introdusa": [intrebare],
        "simptome_generate": [raspuns]
    }
    df_nou = pd.DataFrame(data)
    if not os.path.exists(LOG_FILE):
        df_nou.to_csv(LOG_FILE, index=False)
    else:
        df_nou.to_csv(LOG_FILE, mode="a", header=False, index=False)


if __name__ == "__main__":
    print("📋 Modelul este încărcat și gata de test! Scrie o boală sau mai multe separate prin virgulă.")
    boala_input = input("Introdu o boală (sau mai multe): ")

    for boala in boala_input.split(","):
        boala = boala.strip()
        simptome_raw = genereaza_simptome(boala)
        simptome_curatate = curata_simptome(simptome_raw)
        print(f"💡 {boala}: {simptome_curatate}\n")
        salveaza_predictie(MODEL_NAME, boala, simptome_curatate)
