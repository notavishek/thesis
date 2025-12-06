import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import os

# ============================================================
# 1. MODEL DEFINITION
# ============================================================

class MultiTaskXLMRRoberta(nn.Module):
    def __init__(self, model_name='xlm-roberta-large', dropout=0.2,
                 n_hate_type=6, n_target_group=4, n_severity=4):
        super().__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.hate_type_head = nn.Linear(hidden_size, n_hate_type)
        self.target_group_head = nn.Linear(hidden_size, n_target_group)
        self.severity_head = nn.Linear(hidden_size, n_severity)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        return (
            self.hate_type_head(cls_output),
            self.target_group_head(cls_output),
            self.severity_head(cls_output)
        )

# ============================================================
# 2. DATASET DEFINITION
# ============================================================

class HateDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=160):
        self.df = df.reset_index(drop=True)
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
        
        hate_type = int(row['hate_type'])
        target_group = int(row['target_group'])
        severity = int(row['severity'])
        
        hate_type_mask = hate_type != -1
        target_group_mask = target_group != -1
        severity_mask = severity != -1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'hate_type': torch.tensor(max(0, hate_type), dtype=torch.long),
            'target_group': torch.tensor(max(0, target_group), dtype=torch.long),
            'severity': torch.tensor(max(0, severity), dtype=torch.long),
            'hate_type_mask': torch.tensor(hate_type_mask, dtype=torch.bool),
            'target_group_mask': torch.tensor(target_group_mask, dtype=torch.bool),
            'severity_mask': torch.tensor(severity_mask, dtype=torch.bool),
        }

# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def evaluate(model, data_loader, device, verbose=True):
    model.eval()
    all_preds = {'hate_type': [], 'target_group': [], 'severity': []}
    all_labels = {'hate_type': [], 'target_group': [], 'severity': []}
    all_masks = {'hate_type': [], 'target_group': [], 'severity': []}
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = move_batch_to_device(batch, device)
            ht_logits, tg_logits, sv_logits = model(batch['input_ids'], batch['attention_mask'])
            
            targets = {k: batch[k] for k in ['hate_type', 'target_group', 'severity']}
            masks = {k: batch[f'{k}_mask'] for k in targets.keys()}
            
            all_preds['hate_type'].extend(ht_logits.argmax(dim=1).cpu().numpy())
            all_preds['target_group'].extend(tg_logits.argmax(dim=1).cpu().numpy())
            all_preds['severity'].extend(sv_logits.argmax(dim=1).cpu().numpy())
            
            for task in ['hate_type', 'target_group', 'severity']:
                all_labels[task].extend(targets[task].cpu().numpy())
                all_masks[task].extend(masks[task].cpu().numpy())
    
    metrics = {}
    for task in ['hate_type', 'target_group', 'severity']:
        mask = np.array(all_masks[task]).astype(bool)
        if mask.sum() > 0:
            preds = np.array(all_preds[task])[mask]
            labels = np.array(all_labels[task])[mask]
            
            metrics[f'{task}_macro_f1'] = f1_score(labels, preds, average='macro', zero_division=0)
            metrics[f'{task}_micro_f1'] = f1_score(labels, preds, average='micro', zero_division=0)
            
            if verbose:
                print(f'\n{"="*40}')
                print(f'{task.upper()} Classification Report:')
                print(f'{"="*40}')
                print(classification_report(labels, preds, zero_division=0))
        else:
            metrics[f'{task}_macro_f1'] = None
            metrics[f'{task}_micro_f1'] = None
            
    return metrics

# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main():
    # Configuration
    BATCH_SIZE = 16
    MAX_LENGTH = 160
    CHECKPOINT_PATH = 'checkpoints/xlmr_enhanced_best.pt'
    DATASET_PATH = 'dataset/UNIFIED_ALL_SPLIT_ENHANCED.csv'
    
    # Check if files exist
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    if not os.path.exists(DATASET_PATH):
        # Fallback to standard split if enhanced not found locally (might be on Drive)
        DATASET_PATH = 'dataset/UNIFIED_ALL_SPLIT.csv'
        if not os.path.exists(DATASET_PATH):
             print(f"❌ Dataset not found: {DATASET_PATH}")
             return
        print(f"⚠️ Enhanced dataset not found locally, using: {DATASET_PATH}")
    else:
        print(f"✅ Using dataset: {DATASET_PATH}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Device: {device}")

    # Load Dataset
    print("⏳ Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    test_df = df[df['split'] == 'test']
    print(f"📊 Test set size: {len(test_df)} samples")

    # Tokenizer
    print("⏳ Loading tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

    # DataLoader
    test_dataset = HateDataset(test_df, tokenizer, max_length=MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # num_workers=0 for Windows

    # Load Model
    print(f"⏳ Loading model from {CHECKPOINT_PATH}...")
    model = MultiTaskXLMRRoberta().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    print("✅ Model loaded successfully!")

    # Run Evaluation
    metrics = evaluate(model, test_loader, device, verbose=True)

    # Print Summary
    print("\n" + "="*60)
    print("📊 FINAL TEST SET RESULTS")
    print("="*60)
    print(f"Hate Type:    Macro F1: {metrics['hate_type_macro_f1']:.4f} | Micro F1: {metrics['hate_type_micro_f1']:.4f}")
    print(f"Target Group: Macro F1: {metrics['target_group_macro_f1']:.4f} | Micro F1: {metrics['target_group_micro_f1']:.4f}")
    print(f"Severity:     Macro F1: {metrics['severity_macro_f1']:.4f} | Micro F1: {metrics['severity_micro_f1']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
