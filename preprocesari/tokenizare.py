import pandas as pd
from transformers import GPT2Tokenizer


TOKENIZER_PATH = "distilgpt2"
INPUT_PATH = "Data/boli_combined.csv"
OUTPUT_PATH = "Data/boli_tokenized.csv"
MAX_LENGTH = 160


tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token


df = pd.read_csv(INPUT_PATH)


tokenized_rows = []
for _, row in df.iterrows():
    name = row["Name"]
    symptoms = row["Symptoms"]
    input_text = f"Generate symptoms for: {name} |"
    full_text = input_text + " " + symptoms

    tokens = tokenizer(full_text, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    labels = tokens["input_ids"].copy()
    input_len = len(tokenizer(input_text, truncation=True, max_length=MAX_LENGTH)["input_ids"])
    labels[:input_len] = [-100] * input_len

    tokenized_rows.append({
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels
    })


tokenized_df = pd.DataFrame(tokenized_rows)
tokenized_df.to_csv(OUTPUT_PATH, index=False)
print(f"Tokenization completed and saved in '{OUTPUT_PATH}'")
