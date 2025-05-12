import pandas as pd
import time, os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from sklearn.utils import resample
import random
import ast

SAVE_DIR = "modele_tokenizare"
MODEL_PATH = "modele_tokenizare/WebS_model_15_epoch_21"
TOKENIZER_PATH = "distilgpt2"
CSV_DATA_PATH = "Data/boli_tokenized.csv"  
RESULTS_FILE = "model_evolution/training_dates.csv"
os.makedirs("model_evolution", exist_ok=True)


BATCH_SIZE = 8
LEARNING_RATE = 2e-5
LABEL_SMOOTHING = 0.05
MAX_LENGTH = 160


NUM_EPOCHS = 8
PATIENCE = 3
START_EPOCH = 11

tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for param in model.parameters():
    param.requires_grad = False
for name, param in model.transformer.h[-3:].named_parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True

class TokenizedDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(ast.literal_eval(
            self.data.loc[idx, 'input_ids']), 
            dtype=torch.long)
        attention_mask = torch.tensor(ast.literal_eval(
            self.data.loc[idx, 'attention_mask']),
              dtype=torch.long)
        labels = torch.tensor(ast.literal_eval(
            self.data.loc[idx, 'labels']), 
            dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

df = pd.read_csv(CSV_DATA_PATH)
dataset = TokenizedDataset(df)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.01)
num_training_steps = NUM_EPOCHS * len(train_loader)
num_warmup_steps = int(0.2 * num_training_steps)

from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=0.5
)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=LABEL_SMOOTHING)

results_file = f"model_evolution/training_dates.csv"
if os.path.exists(results_file):
    results = pd.read_csv(results_file)
else:
    results = pd.DataFrame(columns=["model","epoch", "train_loss", "val_loss", "duration_sec"])

best_val_loss = float("inf")
no_improve_counter = 0


for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch in tqdm(train_loader, desc=f"Training epoch {epoch + 1}", ncols=100):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
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

    print(f"Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

    results.loc[len(results)] = [MODEL_PATH, epoch + 1, avg_train_loss, avg_val_loss, duration]
    results.to_csv(results_file, index=False)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_counter = 0
        model.save_pretrained(f"{SAVE_DIR}/Model_epoch_{epoch + 1}")
        print("Model salvat")

    else:
        no_improve_counter += 1
        if no_improve_counter >= PATIENCE:
            print("Early stopping activated!")
            break

