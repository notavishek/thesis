import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch.nn as nn
import sys

# ============================================================
# 1. SETUP
# ============================================================
CHECKPOINT_PATH = 'checkpoints/xlmr_enhanced_best.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 2. MODEL DEFINITION
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
# 3. PREDICTION LOGIC
# ============================================================
HATE_TYPE_LABELS = {0: 'not_hate', 1: 'political', 2: 'religious', 3: 'gender', 4: 'personal', 5: 'geopolitical'}
TARGET_LABELS = {0: 'none', 1: 'individual', 2: 'group', 3: 'community'}
SEVERITY_LABELS = {0: 'none', 1: 'low', 2: 'medium', 3: 'high'}

def predict_interactive():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = MultiTaskXLMRRoberta().to(DEVICE)
        
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "="*60)
    print("🎮 INTERACTIVE PREDICTION MODE")
    print("Type a sentence (Bengali/English/Banglish) and press Enter.")
    print("Type 'q' or 'exit' to quit.")
    print("="*60)

    while True:
        try:
            text = input("\nInput Text > ")
            if text.lower() in ['q', 'quit', 'exit']:
                print("Exiting...")
                break
            if not text.strip():
                continue
                
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=160).to(DEVICE)
            with torch.no_grad():
                ht, tg, sv = model(inputs['input_ids'], inputs['attention_mask'])
            
            ht_pred = HATE_TYPE_LABELS[ht.argmax().item()]
            tg_pred = TARGET_LABELS[tg.argmax().item()]
            sv_pred = SEVERITY_LABELS[sv.argmax().item()]
            
            # Confidence scores
            ht_conf = torch.softmax(ht, dim=1).max().item() * 100
            tg_conf = torch.softmax(tg, dim=1).max().item() * 100
            
            print(f"  ➤ Hate Type:    {ht_pred.upper()} ({ht_conf:.1f}%)")
            print(f"  ➤ Target Group: {tg_pred.upper()} ({tg_conf:.1f}%)")
            print(f"  ➤ Severity:     {sv_pred.upper()}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    predict_interactive()
