import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import os
from tqdm import tqdm
import random

# ============================================================
# 1. CONFIGURATION
# ============================================================
BATCH_SIZE = 8 # Smaller batch for CPU
EPOCHS = 1
LR = 1e-5
NEUTRAL_SAMPLES = 500 # Number of neutral samples to use
HATE_SAMPLES = 500    # Number of hate samples to keep (to prevent forgetting)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📍 Device: {DEVICE}")

# ============================================================
# 2. DATA LOADING & MIXING
# ============================================================
print("⏳ Preparing Correction Dataset...")

# Load Neutral Data
df_neutral = pd.read_csv('dataset/neutral_boost.csv')
df_neutral = df_neutral.sample(n=min(len(df_neutral), NEUTRAL_SAMPLES), random_state=42)
print(f"   - Loaded {len(df_neutral)} Neutral samples")

# Load Original Hate Data (to prevent forgetting)
df_hate = pd.read_csv('dataset/UNIFIED_ALL_SPLIT.csv')
# Filter for ONLY Hate samples (hate_type != 0)
df_hate = df_hate[df_hate['hate_type'] != 0]
df_hate = df_hate.sample(n=min(len(df_hate), HATE_SAMPLES), random_state=42)
print(f"   - Loaded {len(df_hate)} Hate samples")

# Combine
df_train = pd.concat([df_neutral, df_hate]).sample(frac=1).reset_index(drop=True)
print(f"✅ Final Training Set: {len(df_train)} samples (Balanced)")

# ============================================================
# 3. DATASET CLASS
# ============================================================
class CorrectionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128): # Shorter length for speed
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'hate_type': torch.tensor(int(row['hate_type']), dtype=torch.long),
            'target_group': torch.tensor(int(row['target_group']), dtype=torch.long),
            'severity': torch.tensor(int(row['severity']), dtype=torch.long),
        }

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
train_dataset = CorrectionDataset(df_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 4. MODEL DEFINITION
# ============================================================
class MultiTaskXLMRRoberta(nn.Module):
    def __init__(self, model_name='xlm-roberta-large', dropout=0.3):
        super().__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.hate_type_head = nn.Linear(hidden_size, 6)
        self.target_group_head = nn.Linear(hidden_size, 4)
        self.severity_head = nn.Linear(hidden_size, 4)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return (
            self.hate_type_head(cls_output),
            self.target_group_head(cls_output),
            self.severity_head(cls_output)
        )

print("⏳ Loading Model...")
model = MultiTaskXLMRRoberta()
# Load the paranoid model
model.load_state_dict(torch.load('checkpoints/xlmr_smart_best.pt', map_location=DEVICE))
model.to(DEVICE)
model.train()

# Freeze Backbone for speed and safety (Only retrain heads to adjust bias)
for param in model.backbone.parameters():
    param.requires_grad = False
print("❄️ Backbone Frozen (Training Heads Only)")

# ============================================================
# 5. TRAINING LOOP
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"🚀 Starting Correction Training ({EPOCHS} Epoch)...")

for epoch in range(EPOCHS):
    total_loss = 0
    loop = tqdm(train_loader)
    
    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        ht_label = batch['hate_type'].to(DEVICE)
        tg_label = batch['target_group'].to(DEVICE)
        sv_label = batch['severity'].to(DEVICE)
        
        optimizer.zero_grad()
        
        ht_logits, tg_logits, sv_logits = model(input_ids, attention_mask)
        
        # Masked Loss Calculation (Ignore -1 labels)
        loss = 0
        
        # Hate Type
        valid_ht = ht_label != -1
        if valid_ht.any():
            loss += criterion(ht_logits[valid_ht], ht_label[valid_ht])
            
        # Target Group
        valid_tg = tg_label != -1
        if valid_tg.any():
            loss += criterion(tg_logits[valid_tg], tg_label[valid_tg])
            
        # Severity
        valid_sv = sv_label != -1
        if valid_sv.any():
            loss += criterion(sv_logits[valid_sv], sv_label[valid_sv])
        
        if loss != 0: # Avoid error if batch has no valid labels (unlikely)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f"Loss: {loss.item():.4f}")

# ============================================================
# 6. SAVE
# ============================================================
output_path = 'checkpoints/xlmr_corrected.pt'
torch.save(model.state_dict(), output_path)
print(f"\n✅ Corrected Model Saved: {output_path}")
print("You can now run 'comprehensive_test.py' (update the path first!)")
