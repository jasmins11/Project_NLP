
from datasets import load_dataset
import pandas as pd
import os


data = load_dataset("QuyenAnhDE/Diseases_Symptoms")['train']


df = pd.DataFrame([{"Name": item["Name"], "Symptoms": item["Symptoms"]} for item in data])
df['Symptoms'] = df['Symptoms'].apply(lambda x: ', '.join(x.split(", ")))
df = df[df['Symptoms'].str.len() > 10]
df.drop_duplicates(inplace=True)
df['Symptoms'] = df['Symptoms'].str.lower().str.strip()
df['Symptoms'] = df['Symptoms'].str.replace(r'\s*,\s*', ', ', regex=True)


if not os.path.exists("cleaned_disease_dataset.csv"):
    df.to_csv("cleaned_disease_dataset.csv", index=False)
    print("Saved in cleaned_disease_dataset.csv")
else:
    print("File cleaned_disease_dataset.csv already exists.")
