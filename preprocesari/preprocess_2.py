import json
import pandas as pd
import unicodedata
import re

def clean_unicode(text):
    text = unicodedata.normalize("NFKC", text)             #standardizeaza variatiile de caractere care par la fel vizual, dar sunt codificate diferit
    text = text.replace('\u00A0', ' ')                     #non-breaking space: inlocuiteste spatiile:\u00A0 cu spatiu normal
    text = text.replace("-", "-").replace("—", "-")        #pentru tipuri de En-dash si Em-dash le face "-" simplu. Motiv: pot crea inconsistente la tokenizare
    text = re.sub(r"\s+", " ", text)                       #reduce toate spatiile multiple la un singur spatiu
    return text.strip()

def clean_name(text):
    text = re.sub(r"(?i)^overview\s*[-:\n]*\s*", "", text)
    text = text.replace("-", " ")
    return clean_unicode(text)

# extrage DOAR liniile care incep cu "-" sau bullet point-uri
def extract_symptom_lines(symptom_text):
    bullets = re.findall(r"-\s*(.+)", symptom_text)
    cleaned = []
    for s in bullets:
        s = clean_unicode(s)
        s = re.sub(r"[^a-zA-Z0-9\s,()']", "", s)            #elimina simboluri inutile
        s = s.strip().rstrip(".") 
        if len(s.split()) <= 10:                            #pastreaza doar fraze scurte
            cleaned.append(s.lower())
    return cleaned

def extract_clean_symptoms(data):
    results = []
    for entry in data:
        name = clean_name(entry.get("disease", "").strip())
        symptoms_raw = entry.get("symptoms", "").strip()

        if not name or symptoms_raw.lower() == "n/a":
            continue

        extracted_symptoms = extract_symptom_lines(symptoms_raw)
        if not extracted_symptoms:
            continue

        symptoms = ", ".join(extracted_symptoms)
        results.append({
            "Name": name,
            "Symptoms": symptoms
        })

    return results


with open("Data/boli_nhs.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)


cleaned_data = extract_clean_symptoms(raw_data)


df = pd.DataFrame(cleaned_data)
df.to_csv("Data/boli_nhs_cleaned.csv", index=False)

print(f"✓ Salvate {len(df)} rânduri curate în 'boli_nhs_cleaned.csv'")
