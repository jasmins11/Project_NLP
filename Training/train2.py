import pandas as pd
import time, os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup



SAVE_PREFIX = "trained_model_v1"
SAVE_DIR = "modele_tokenizare"
MODEL_PATH = f"./{SAVE_DIR}/model__epoch_11"
TOKENIZER_PATH = "distilgpt2"

CSV_DATA_PATH = "Data/boli_nhs_cleaned.csv"
BATCH_SIZE = 8
NUM_EPOCHS = 6
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
PATIENCE = 2


tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for param in model.parameters():
    param.requires_grad = False
for name, param in model.transformer.h[-6:].named_parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True


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
        input_len = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)['input_ids'].shape[1]
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
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)


optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE,weight_decay=0.01)
num_training_steps = NUM_EPOCHS * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.05)

results_file = f"{SAVE_DIR}/training_log_{SAVE_PREFIX}.csv"
if os.path.exists(results_file):
    results = pd.read_csv(results_file)
else:
    results = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "duration_sec"])

best_val_loss = float("inf")
no_improve_counter = 0


start_epoch = 11
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
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

    results.loc[len(results)] = [epoch + 1, avg_train_loss, avg_val_loss, duration]
    results.to_csv(results_file, index=False)


    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_counter = 0
        model.save_pretrained(f"{SAVE_DIR}/model__epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"{SAVE_DIR}/tokenizare__epoch_{epoch + 1}")
        print(" Model salvat ca best!")
    else:
        no_improve_counter += 1
        if no_improve_counter >= PATIENCE:
            print(" Early stopping activat!")
            break


model.save_pretrained(f"{SAVE_DIR}/WebS_model__epoch_{epoch + 1}")
tokenizer.save_pretrained(f"{SAVE_DIR}/WebS_tokenizare__epoch_{epoch + 1}")
print(" Model final salvat.")
