import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import os

# ============================================================
# 1. MODEL DEFINITION
# ============================================================
class MultiTaskXLMRRoberta(nn.Module):
    def __init__(self, model_name='xlm-roberta-large', dropout=0.3, # Matches Smart Mode
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
# 2. CONFIG
# ============================================================
# Adjust path based on where script is run
if os.path.exists("checkpoints/xlmr_smart_best.pt"):
    CHECKPOINT_PATH = "checkpoints/xlmr_smart_best.pt"
elif os.path.exists("../checkpoints/xlmr_smart_best.pt"):
    CHECKPOINT_PATH = "../checkpoints/xlmr_smart_best.pt"
else:
    CHECKPOINT_PATH = "checkpoints/xlmr_smart_best.pt" # Default fallback

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAFETY_THRESHOLD = 0.90  # If confidence < 90%, force to Not Hate

HATE_TYPE_LABELS = {
    0: 'Not Hate / Other', 1: 'Political', 2: 'Religious', 
    3: 'Gender', 4: 'Personal Attack', 5: 'Geopolitical'
}
TARGET_LABELS = {
    0: 'None / Other', 1: 'Individual', 2: 'Organization / Group', 3: 'Community'
}
SEVERITY_LABELS = {
    0: 'None', 1: 'Low', 2: 'Medium', 3: 'High'
}

# ============================================================
# 3. INITIALIZATION
# ============================================================
print("⏳ Loading Tokenizer...")
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

print(f"⏳ Loading Model from {CHECKPOINT_PATH}...")
model = MultiTaskXLMRRoberta()
if os.path.exists(CHECKPOINT_PATH):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully.")
else:
    print("❌ Checkpoint not found!")
    exit()

model.to(DEVICE)
model.eval()

# ============================================================
# 4. INFERENCE FUNCTION WITH SAFETY
# ============================================================
def predict_safe(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=160).to(DEVICE)
    
    with torch.no_grad():
        ht_logits, tg_logits, sv_logits = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Softmax for probabilities
    ht_probs = torch.softmax(ht_logits, dim=1).squeeze()
    tg_probs = torch.softmax(tg_logits, dim=1).squeeze()
    sv_probs = torch.softmax(sv_logits, dim=1).squeeze()
    
    # Raw Predictions
    ht_pred = torch.argmax(ht_probs).item()
    tg_pred = torch.argmax(tg_probs).item()
    sv_pred = torch.argmax(sv_probs).item()
    
    ht_conf = ht_probs[ht_pred].item()
    
    # 🛡️ SAFETY CHECK 🛡️
    is_safe = False
    original_pred = HATE_TYPE_LABELS[ht_pred]
    
    # If predicted Hate but confidence is low, force to Not Hate
    if ht_pred != 0 and ht_conf < SAFETY_THRESHOLD:
        is_safe = True
        ht_pred = 0
        tg_pred = 0
        sv_pred = 0
        ht_conf = ht_probs[0].item() # Confidence of Not Hate
    
    return {
        'text': text,
        'hate_type': HATE_TYPE_LABELS[ht_pred],
        'target_group': TARGET_LABELS[tg_pred],
        'severity': SEVERITY_LABELS[sv_pred],
        'confidence': f"{ht_conf:.2%}",
        'safety_triggered': is_safe,
        'original_prediction': original_pred if is_safe else "N/A"
    }

# ============================================================
# 5. TEST RUN
# ============================================================
if __name__ == "__main__":
    test_cases = [
        "আজকে আবহাওয়া খুব সুন্দর", # The weather is nice (Paranoia Test)
        "ধন্যবাদ", # Thank you (Paranoia Test)
        "আমি ভালো আছি", # I am fine (Paranoia Test)
        "তুই একটা রাজনৈতিক দালাল", # You are a political broker (Real Hate)
        "মুসলমানরা সব সন্ত্রাসী", # Muslims are terrorists (High Severity Test)
        "You are a stupid idiot", # English Hate
        "I love my country" # English Neutral
    ]
    
    print("\n" + "="*60)
    print(f"🛡️ RUNNING INFERENCE WITH SAFETY THRESHOLD: {SAFETY_THRESHOLD:.0%}")
    print("="*60)
    
    for text in test_cases:
        res = predict_safe(text)
        status = "✅ SAFE" if res['hate_type'] == 'Not Hate / Other' else "❌ HATE"
        if res['safety_triggered']:
            status += " (🛡️ BLOCKED)"
            
        print(f"\n📝 Text: {res['text']}")
        print(f"   Prediction: {res['hate_type']} ({res['confidence']})")
        print(f"   Severity:   {res['severity']}")
        print(f"   Status:     {status}")
        if res['safety_triggered']:
            print(f"   (Original:  {res['original_prediction']})")
