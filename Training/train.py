# train.py
import pandas as pd
import time, os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


MODEL_NAME = "distilbert/distilgpt2"
BATCH_SIZE = 4
NUM_EPOCHS = 6
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
SAVE_PREFIX = "trained_model_v1"
CSV_DATA_PATH = "cleaned_disease_dataset.csv"


tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class DiseaseDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe.to_dict(orient="records")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        disease = self.data[idx]['Name']
        symptoms = self.data[idx]['Symptoms']
        input_text = f"Generate symptoms for: {disease} |"
        full_text = input_text + " " + symptoms

        tokens = self.tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        labels = tokens.input_ids.clone()
        input_len = self.tokenizer(input_text, return_tensors='pt')['input_ids'].shape[1]
        labels[0, :input_len] = -100  

        return {
            'input_ids': tokens.input_ids.squeeze(0),
            'attention_mask': tokens.attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }


df = pd.read_csv(CSV_DATA_PATH)
dataset = DiseaseDataset(df, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)


optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

results = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "duration_sec"])
best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)


    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
    avg_val_loss = val_loss / len(val_loader)
    duration = time.time() - start_time

    print(f"Epoch {epoch} - Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

    results.loc[len(results)] = [epoch, avg_train_loss, avg_val_loss, duration]
    results.to_csv(f"training_log_{SAVE_PREFIX}.csv", index=False)

  
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(f"best_model_{SAVE_PREFIX}_epoch_{epoch}")
        tokenizer.save_pretrained(f"best_tokenizer_{SAVE_PREFIX}_epoch_{epoch}")
        print(" Model salvat!")


model.save_pretrained(f"model_{SAVE_PREFIX}_epoch_{epoch}")
tokenizer.save_pretrained(f"tokenizer_{SAVE_PREFIX}_epoch_{epoch}")
print(" Model final salvat.")
