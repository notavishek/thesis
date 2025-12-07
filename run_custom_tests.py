import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_PATH = 'checkpoints/xlmr_augmented_best.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# MODEL CLASS
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
# LABELS
# ============================================================
HATE_TYPE_LABELS = {0: 'not_hate', 1: 'political', 2: 'religious', 3: 'gender', 4: 'personal', 5: 'geopolitical'}
TARGET_LABELS = {0: 'none', 1: 'individual', 2: 'group', 3: 'community'}
SEVERITY_LABELS = {0: 'none', 1: 'low', 2: 'medium', 3: 'high'}

# ============================================================
# TEST CASES
# ============================================================
TEST_CASES = [
    # --- BANGLISH (The main fix) ---
    ("tui ekta boka choda", "Banglish - Personal Attack"),
    ("tor ma ke chudi", "Banglish - High Severity"),
    ("ei desh ta nosto hoye gelo", "Banglish - Geopolitical"),
    ("shala malaun", "Banglish - Religious Slur"),
    
    # --- BENGALI (Regression check) ---
    ("তুই একটা জানোয়ার", "Bengali - Personal Attack"),
    ("মুসলমানরা সব সন্ত্রাসী", "Bengali - Religious (High Severity Check)"),
    
    # --- ENGLISH (General check) ---
    ("You are a disgusting piece of trash", "English - Personal"),
    ("All politicians should be hanged", "English - Political (High Severity)"),
]

def run_tests():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    model = MultiTaskXLMRRoberta().to(DEVICE)
    
    # Load weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print("\n" + "="*80)
    print(f"{'TEXT':<40} | {'PREDICTION':<40}")
    print("="*80)
    
    for text, desc in TEST_CASES:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=160).to(DEVICE)
        with torch.no_grad():
            ht, tg, sv = model(inputs['input_ids'], inputs['attention_mask'])
        
        ht_pred = HATE_TYPE_LABELS[ht.argmax().item()]
        tg_pred = TARGET_LABELS[tg.argmax().item()]
        sv_pred = SEVERITY_LABELS[sv.argmax().item()]
        
        # Format output
        pred_str = f"{ht_pred.upper()} | {tg_pred.upper()} | {sv_pred.upper()}"
        print(f"Input: {text}")
        print(f"Type:  {desc}")
        print(f"Pred:  {pred_str}")
        print("-" * 80)

if __name__ == "__main__":
    run_tests()
