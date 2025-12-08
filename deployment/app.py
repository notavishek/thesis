import gradio as gr
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import os

# ============================================================
# 1. MODEL DEFINITION (Must match training exactly)
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
# 2. CONFIG & LABELS
# ============================================================
# MODEL_FILE is defined in section 3 now
DEVICE = torch.device('cpu') # Spaces usually run on CPU unless you pay for GPU

HATE_TYPE_LABELS = {
    0: 'Not Hate / Other', 
    1: 'Political', 
    2: 'Religious', 
    3: 'Gender', 
    4: 'Personal Attack', 
    5: 'Geopolitical'
}

TARGET_LABELS = {
    0: 'None / Other', 
    1: 'Individual', 
    2: 'Organization / Group', 
    3: 'Community'
}

SEVERITY_LABELS = {
    0: 'None', 
    1: 'Low', 
    2: 'Medium', 
    3: 'High'
}

# ============================================================
# 3. LOAD MODEL
# ============================================================
MODEL_FILE = "xlmr_quantized.pt"
DEVICE = torch.device('cpu')

print("Loading tokenizer...")
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

print(f"Loading model from {MODEL_FILE}...")
if os.path.exists(MODEL_FILE):
    try:
        # Load the full quantized model object
        model = torch.load(MODEL_FILE, map_location=DEVICE)
        print("✅ Quantized model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Fallback for debugging
        model = MultiTaskXLMRRoberta()
else:
    print(f"⚠️ Warning: {MODEL_FILE} not found. Please upload it to the Space.")
    model = MultiTaskXLMRRoberta()

model.to(DEVICE)
model.eval()

# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================
def predict_hate(text):
    if not text or not text.strip():
        return "Please enter text.", "", "", ""
        
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=160).to(DEVICE)
    
    with torch.no_grad():
        ht_logits, tg_logits, sv_logits = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Get Predictions
    ht_idx = ht_logits.argmax().item()
    tg_idx = tg_logits.argmax().item()
    sv_idx = sv_logits.argmax().item()
    
    # Get Confidences
    ht_conf = torch.softmax(ht_logits, dim=1).max().item()
    tg_conf = torch.softmax(tg_logits, dim=1).max().item()
    sv_conf = torch.softmax(sv_logits, dim=1).max().item()
    
    # Format Outputs
    ht_out = f"{HATE_TYPE_LABELS[ht_idx]} ({ht_conf:.1%})"
    tg_out = f"{TARGET_LABELS[tg_idx]} ({tg_conf:.1%})"
    sv_out = f"{SEVERITY_LABELS[sv_idx]} ({sv_conf:.1%})"
    
    # Overall Verdict
    is_hate = "🚨 HATE SPEECH DETECTED" if ht_idx != 0 else "✅ Non-Hate / Neutral"
    
    return is_hate, ht_out, tg_out, sv_out

# ============================================================
# 5. GRADIO INTERFACE
# ============================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛡️ Multilingual Hate Speech Detection (Thesis Demo)
        **Languages Supported:** Bengali, English, Banglish (Romanized Bengali)
        
        *Disclaimer: This is a research prototype. Outputs may be incorrect or biased.*
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter Text", 
                placeholder="Type something here... (e.g., 'tui ekta boka', 'I hate politicians')",
                lines=3
            )
            submit_btn = gr.Button("Analyze", variant="primary")
            
            gr.Examples(
                examples=[
                    ["All politicians are corrupt thieves."],
                    ["tui ekta boka choda"],
                    ["মুসলমানরা সব সন্ত্রাসী"],
                    ["I love my country very much."],
                    ["shala malaun"]
                ],
                inputs=input_text
            )
            
        with gr.Column():
            verdict_output = gr.Label(label="Verdict")
            with gr.Group():
                ht_output = gr.Textbox(label="Hate Type")
                tg_output = gr.Textbox(label="Target Group")
                sv_output = gr.Textbox(label="Severity")

    submit_btn.click(
        fn=predict_hate,
        inputs=input_text,
        outputs=[verdict_output, ht_output, tg_output, sv_output]
    )

if __name__ == "__main__":
    demo.launch()
