# -*- coding: utf-8 -*-

# ============================
# Install Dependencies
# ============================
!pip install -q transformers datasets tokenizers accelerate scikit-learn tqdm sentencepiece

# ============================
# Imports
# ============================
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================
# Google Drive
# ============================
from google.colab import drive, files
drive.mount('/content/drive')

SAVE_DIR = "/content/drive/MyDrive/text_model_clean"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================
# Constants
# ============================
RANDOM_SEED = 42
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
MODEL_NAME = "microsoft/deberta-v3-base"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ============================
# Reproducibility
# ============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================
# Upload Data
# ============================
print("Upload train.csv, validation.csv, test.csv")
files.upload()

train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("validation.csv")
test_df  = pd.read_csv("test.csv")

# ============================
# Columns
# ============================
TEXT_COL  = "text"
LABEL_COL = "label"

assert LABEL_COL in train_df.columns, "❌ label column not found"
assert TEXT_COL  in train_df.columns, "❌ text column not found"

# labels MUST be integers
assert pd.api.types.is_integer_dtype(train_df[LABEL_COL]), "❌ label must be integer"

num_labels = train_df[LABEL_COL].nunique()
print("Number of labels:", num_labels)

# ============================
# Tokenizer & Model
# ============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config
).to(DEVICE)

# ============================
# Dataset
# ============================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.values
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================
# DataLoaders
# ============================
train_ds = TextDataset(train_df[TEXT_COL], train_df[LABEL_COL], tokenizer, MAX_LEN)
val_ds   = TextDataset(val_df[TEXT_COL],   val_df[LABEL_COL],   tokenizer, MAX_LEN)
test_ds  = TextDataset(test_df[TEXT_COL],  test_df[LABEL_COL],  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ============================
# Optimizer & Scheduler
# ============================
optimizer = optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()

# ============================
# Training Loop
# ============================
best_f1 = 0.0
patience = 3
counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---- Train ----
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    # ---- Validation ----
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention
            ).logits

            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    f1 = f1_score(trues, preds, average="weighted")
    acc = accuracy_score(trues, preds)

    print(f"Val F1: {f1:.4f} | Val Acc: {acc:.4f}")

    # ---- Early Stopping ----
    if f1 > best_f1:
        best_f1 = f1
        counter = 0
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print("✔ Best model saved")
    else:
        counter += 1
        if counter >= patience:
            print("⏹ Early stopping")
            break

# ============================
# Test Evaluation
# ============================
print("\nFinal Test Evaluation")

model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
model.eval()

test_preds, test_trues = [], []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention
        ).logits

        test_preds.extend(torch.argmax(logits, 1).cpu().numpy())
        test_trues.extend(labels.cpu().numpy())

print(classification_report(test_trues, test_preds))
print("Accuracy:", accuracy_score(test_trues, test_preds))
print("F1 weighted:", f1_score(test_trues, test_preds, average="weighted"))
