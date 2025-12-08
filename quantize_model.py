import torch
import torch.nn as nn
from transformers import XLMRobertaModel
import os
from torch.quantization import quantize_dynamic

# ============================================================
# 1. DEFINE MODEL (Must match training)
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
# 2. LOAD & QUANTIZE
# ============================================================
INPUT_PATH = 'checkpoints/xlmr_augmented_best.pt'
OUTPUT_PATH = 'deployment/xlmr_quantized.pt'

print(f"⏳ Loading original model from {INPUT_PATH}...")
device = torch.device('cpu') # Quantization is typically done on CPU
model = MultiTaskXLMRRoberta()
checkpoint = torch.load(INPUT_PATH, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("📉 Quantizing model (Float32 -> Int8)...")
# This converts the heavy Linear layers to 8-bit integers
quantized_model = quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

print(f"💾 Saving quantized model to {OUTPUT_PATH}...")
# We save the FULL model object because the structure has changed
torch.save(quantized_model, OUTPUT_PATH)

# Check sizes
original_size = os.path.getsize(INPUT_PATH) / (1024 * 1024)
new_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print("\n" + "="*50)
print(f"✅ SUCCESS!")
print(f"Original Size: {original_size:.2f} MB")
print(f"Quantized Size: {new_size:.2f} MB")
print(f"Reduction: {original_size/new_size:.1f}x smaller")
print("="*50)
