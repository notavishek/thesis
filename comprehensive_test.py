# Comprehensive Test Suite for Multilingual Hate Detection Model
# =================================================================
# Tests 100+ examples across all combinations of:
# - 3 languages (Bengali, English, Banglish)
# - 6 hate types (0-5: other, political, religious, gender, personal_attack, geopolitical)
# - 4 target groups (0-3: other, individual, org/group, community)
# - 4 severity levels (0-3: none, low, medium, high)

import torch
from transformers import XLMRobertaTokenizer
import torch.nn as nn
from transformers import XLMRobertaModel
import pandas as pd

# ============================================================
# 1. MODEL DEFINITION (Same as training)
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
# 2. LABEL MAPPINGS
# ============================================================

HATE_TYPE_LABELS = {
    0: 'not_hate/other',
    1: 'political',
    2: 'religious',
    3: 'gender',
    4: 'personal_attack',
    5: 'geopolitical'
}

TARGET_GROUP_LABELS = {
    0: 'other/none',
    1: 'individual',
    2: 'organization/group',
    3: 'community'
}

SEVERITY_LABELS = {
    0: 'none',
    1: 'low',
    2: 'medium',
    3: 'high'
}

# ============================================================
# 3. COMPREHENSIVE TEST DATASET (120 examples)
# ============================================================

TEST_EXAMPLES = [
    # ========== BENGALI EXAMPLES ==========
    
    # Political hate
    ("এই সরকার সব চোর", "bengali", "political", "organization/group", "medium"),
    ("তুই একটা রাজনৈতিক দালাল", "bengali", "political", "individual", "medium"),
    ("এই দেশের রাজনীতিবিদরা সব দুর্নীতিবাজ", "bengali", "political", "community", "low"),
    ("প্রধানমন্ত্রী একটা মিথ্যাবাদী", "bengali", "political", "individual", "medium"),
    
    # Religious hate
    ("মুসলমানরা সব সন্ত্রাসী", "bengali", "religious", "community", "high"),
    ("হিন্দুরা সব কাফের", "bengali", "religious", "community", "high"),
    ("এই ধর্মের লোকেরা খারাপ", "bengali", "religious", "community", "medium"),
    ("তুই একটা ধর্মান্ধ", "bengali", "religious", "individual", "medium"),
    
    # Gender hate
    ("মেয়েদের ঘরে থাকা উচিত", "bengali", "gender", "community", "medium"),
    ("তুই একটা খানকি", "bengali", "gender", "individual", "high"),
    ("মহিলারা গাড়ি চালাতে পারে না", "bengali", "gender", "community", "low"),
    ("তুই একটা বেশ্যা", "bengali", "gender", "individual", "high"),
    
    # Personal attack
    ("তুই মরে যা শালা", "bengali", "personal_attack", "individual", "high"),
    ("তুই একটা বোকা", "bengali", "personal_attack", "individual", "low"),
    ("তোর বাপ-মা তোকে কেন জন্ম দিল", "bengali", "personal_attack", "individual", "high"),
    ("তুই একটা জানোয়ার", "bengali", "personal_attack", "individual", "high"),
    ("তুই পাগল", "bengali", "personal_attack", "individual", "low"),
    
    # Geopolitical
    ("ভারত একটা খারাপ দেশ", "bengali", "geopolitical", "organization/group", "medium"),
    ("পাকিস্তানিরা সব খারাপ", "bengali", "geopolitical", "community", "medium"),
    ("এই দেশের মানুষ সব অশিক্ষিত", "bengali", "geopolitical", "community", "medium"),
    
    # Not hate (neutral)
    ("আজকে আবহাওয়া খুব সুন্দর", "bengali", "not_hate/other", "other/none", "none"),
    ("আমি ভালো আছি", "bengali", "not_hate/other", "other/none", "none"),
    ("ধন্যবাদ", "bengali", "not_hate/other", "other/none", "none"),
    
    # ========== ENGLISH EXAMPLES ==========
    
    # Political hate
    ("This government is full of thieves", "english", "political", "organization/group", "medium"),
    ("You're a political sellout", "english", "political", "individual", "medium"),
    ("All politicians are corrupt liars", "english", "political", "community", "medium"),
    ("The president is a complete idiot", "english", "political", "individual", "medium"),
    ("Democrats are destroying America", "english", "political", "community", "medium"),
    ("Republicans are all racists", "english", "political", "community", "high"),
    
    # Religious hate
    ("All Muslims are terrorists", "english", "religious", "community", "high"),
    ("Christians are brainwashed sheep", "english", "religious", "community", "medium"),
    ("Jews control the media", "english", "religious", "community", "high"),
    ("You're a religious fanatic", "english", "religious", "individual", "medium"),
    ("Atheists have no morals", "english", "religious", "community", "medium"),
    
    # Gender hate
    ("Women belong in the kitchen", "english", "gender", "community", "medium"),
    ("You're such a slut", "english", "gender", "individual", "high"),
    ("Men are trash", "english", "gender", "community", "medium"),
    ("She's a worthless whore", "english", "gender", "individual", "high"),
    ("Girls can't do math", "english", "gender", "community", "low"),
    ("He's a pussy", "english", "gender", "individual", "medium"),
    
    # Personal attack
    ("I hope you die alone", "english", "personal_attack", "individual", "high"),
    ("You're such an idiot", "english", "personal_attack", "individual", "low"),
    ("Kill yourself you pathetic loser", "english", "personal_attack", "individual", "high"),
    ("You're a worthless piece of garbage", "english", "personal_attack", "individual", "high"),
    ("Go fuck yourself", "english", "personal_attack", "individual", "medium"),
    ("You're an absolute moron", "english", "personal_attack", "individual", "low"),
    ("Drop dead", "english", "personal_attack", "individual", "high"),
    
    # Geopolitical
    ("Immigrants are ruining this country", "english", "geopolitical", "community", "high"),
    ("China is a threat to the world", "english", "geopolitical", "organization/group", "medium"),
    ("Foreigners should go back home", "english", "geopolitical", "community", "high"),
    ("That country is full of criminals", "english", "geopolitical", "organization/group", "medium"),
    ("Russian people are all spies", "english", "geopolitical", "community", "medium"),
    
    # Not hate (neutral)
    ("Have a nice day!", "english", "not_hate/other", "other/none", "none"),
    ("The weather is beautiful", "english", "not_hate/other", "other/none", "none"),
    ("Thank you for your help", "english", "not_hate/other", "other/none", "none"),
    ("I love this song", "english", "not_hate/other", "other/none", "none"),
    
    # ========== BANGLISH EXAMPLES ==========
    
    # Political hate
    ("ei government sob chor", "banglish", "political", "organization/group", "medium"),
    ("tui ekta political dalal", "banglish", "political", "individual", "medium"),
    ("ei desher politician ra sob kharap", "banglish", "political", "community", "medium"),
    ("prime minister mithyabadi", "banglish", "political", "individual", "medium"),
    
    # Religious hate
    ("muslimra sob terrorist", "banglish", "religious", "community", "high"),
    ("hindura sob kharap", "banglish", "religious", "community", "high"),
    ("tui ekta dharmandho", "banglish", "religious", "individual", "medium"),
    ("ei dhormer lokra pagol", "banglish", "religious", "community", "medium"),
    
    # Gender hate
    ("meyeder ghore thaka uchit", "banglish", "gender", "community", "medium"),
    ("tui ekta khanki", "banglish", "gender", "individual", "high"),
    ("meye ra gaari chalate pare na", "banglish", "gender", "community", "low"),
    ("tui ekta beshya", "banglish", "gender", "individual", "high"),
    
    # Personal attack
    ("tui moira ja shala", "banglish", "personal_attack", "individual", "high"),
    ("tui ekta boka", "banglish", "personal_attack", "individual", "low"),
    ("tor bap-ma toke keno jonmo dilo", "banglish", "personal_attack", "individual", "high"),
    ("tui ekta janoyar", "banglish", "personal_attack", "individual", "high"),
    ("tui pagol", "banglish", "personal_attack", "individual", "low"),
    ("tui ekta harami", "banglish", "personal_attack", "individual", "medium"),
    ("tor jonno duniya kharap", "banglish", "personal_attack", "individual", "medium"),
    
    # Geopolitical
    ("india ekta kharap desh", "banglish", "geopolitical", "organization/group", "medium"),
    ("pakistanira sob kharap", "banglish", "geopolitical", "community", "medium"),
    ("ei desher manush sob oshikkhito", "banglish", "geopolitical", "community", "medium"),
    
    # Not hate (neutral)
    ("ajke abohawa valo", "banglish", "not_hate/other", "other/none", "none"),
    ("ami bhalo achi", "banglish", "not_hate/other", "other/none", "none"),
    ("dhonnobad", "banglish", "not_hate/other", "other/none", "none"),
    
    # ========== EDGE CASES & AMBIGUOUS ==========
    
    # Low severity personal attacks
    ("You're silly", "english", "personal_attack", "individual", "low"),
    ("tui choto", "banglish", "personal_attack", "individual", "low"),
    ("তুই ছোট", "bengali", "personal_attack", "individual", "low"),
    
    # Organization-targeted hate
    ("Facebook is evil", "english", "not_hate/other", "organization/group", "low"),
    ("Google is spying on us", "english", "not_hate/other", "organization/group", "low"),
    ("ei company sob thug", "banglish", "not_hate/other", "organization/group", "medium"),
    
    # Community vs Organization ambiguity
    ("The police are all corrupt", "english", "political", "community", "medium"),
    ("Doctors are greedy", "english", "not_hate/other", "community", "low"),
    ("Teachers don't care about students", "english", "not_hate/other", "community", "low"),
    
    # Mixed language
    ("tui ekta stupid boka", "banglish", "personal_attack", "individual", "low"),
    ("ami hate kori ei government ke", "banglish", "political", "organization/group", "medium"),
    
    # Sarcasm/Context-dependent
    ("What a genius move", "english", "not_hate/other", "other/none", "none"),  # Could be sarcastic
    ("Great job destroying everything", "english", "not_hate/other", "other/none", "none"),
    
    # Additional high-severity examples
    ("তুই আত্মহত্যা কর", "bengali", "personal_attack", "individual", "high"),
    ("Kill all Muslims", "english", "religious", "community", "high"),
    ("Women are inferior to men", "english", "gender", "community", "high"),
    ("tui rape korte parbi", "banglish", "gender", "individual", "high"),
    
    # Additional political examples
    ("বিএনপি সব খারাপ", "bengali", "political", "organization/group", "medium"),
    ("আওয়ামী লীগ চোর", "bengali", "political", "organization/group", "medium"),
    ("BJP is fascist", "english", "political", "organization/group", "medium"),
    ("Congress is useless", "english", "political", "organization/group", "low"),
    
    # Additional geopolitical
    ("Arabs are backward", "english", "geopolitical", "community", "high"),
    ("Europeans are arrogant", "english", "geopolitical", "community", "medium"),
    ("bangladeshira sob gareeb", "banglish", "geopolitical", "community", "medium"),
    
    # Borderline not-hate
    ("This policy is terrible", "english", "not_hate/other", "other/none", "none"),
    ("I disagree with this decision", "english", "not_hate/other", "other/none", "none"),
    ("ei kotha bhul", "banglish", "not_hate/other", "other/none", "none"),
]

# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_batch(texts, model, tokenizer, device):
    """Predict for a batch of texts."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(text, max_length=160, padding='max_length', 
                                truncation=True, return_tensors='pt')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            ht_logits, tg_logits, sv_logits = model(input_ids, attention_mask)
            
            ht_pred = ht_logits.argmax(dim=1).item()
            tg_pred = tg_logits.argmax(dim=1).item()
            sv_pred = sv_logits.argmax(dim=1).item()
            
            results.append({
                'hate_type_pred': HATE_TYPE_LABELS[ht_pred],
                'target_group_pred': TARGET_GROUP_LABELS[tg_pred],
                'severity_pred': SEVERITY_LABELS[sv_pred],
                'hate_type_conf': torch.softmax(ht_logits, dim=1).max().item(),
                'target_group_conf': torch.softmax(tg_logits, dim=1).max().item(),
                'severity_conf': torch.softmax(sv_logits, dim=1).max().item(),
            })
    
    return results

# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

def evaluate_results(test_data, predictions):
    """Calculate accuracy for each task."""
    results = {
        'hate_type_correct': 0,
        'target_group_correct': 0,
        'severity_correct': 0,
        'total': len(test_data),
        'hate_type_by_class': {},
        'target_group_by_class': {},
        'severity_by_class': {},
        'language_breakdown': {}
    }
    
    for (text, lang, exp_ht, exp_tg, exp_sv), pred in zip(test_data, predictions):
        # Overall accuracy
        if pred['hate_type_pred'] == exp_ht:
            results['hate_type_correct'] += 1
        if pred['target_group_pred'] == exp_tg:
            results['target_group_correct'] += 1
        if pred['severity_pred'] == exp_sv:
            results['severity_correct'] += 1
        
        # Per-class accuracy
        if exp_ht not in results['hate_type_by_class']:
            results['hate_type_by_class'][exp_ht] = {'correct': 0, 'total': 0}
        results['hate_type_by_class'][exp_ht]['total'] += 1
        if pred['hate_type_pred'] == exp_ht:
            results['hate_type_by_class'][exp_ht]['correct'] += 1
        
        if exp_tg not in results['target_group_by_class']:
            results['target_group_by_class'][exp_tg] = {'correct': 0, 'total': 0}
        results['target_group_by_class'][exp_tg]['total'] += 1
        if pred['target_group_pred'] == exp_tg:
            results['target_group_by_class'][exp_tg]['correct'] += 1
        
        if exp_sv not in results['severity_by_class']:
            results['severity_by_class'][exp_sv] = {'correct': 0, 'total': 0}
        results['severity_by_class'][exp_sv]['total'] += 1
        if pred['severity_pred'] == exp_sv:
            results['severity_by_class'][exp_sv]['correct'] += 1
        
        # Language breakdown
        if lang not in results['language_breakdown']:
            results['language_breakdown'][lang] = {
                'hate_type_correct': 0, 'target_group_correct': 0, 
                'severity_correct': 0, 'total': 0
            }
        results['language_breakdown'][lang]['total'] += 1
        if pred['hate_type_pred'] == exp_ht:
            results['language_breakdown'][lang]['hate_type_correct'] += 1
        if pred['target_group_pred'] == exp_tg:
            results['language_breakdown'][lang]['target_group_correct'] += 1
        if pred['severity_pred'] == exp_sv:
            results['language_breakdown'][lang]['severity_correct'] += 1
    
    return results

# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE HATE DETECTION MODEL EVALUATION")
    print(f"Testing {len(TEST_EXAMPLES)} examples across 3 languages")
    print("=" * 80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Test different checkpoints:
    # checkpoint_path = 'checkpoints/xlmr_v2_classweights_best (1).pt'  # With class weights - 42.6% acc
    # checkpoint_path = 'checkpoints/xlmr_enhanced_best.pt'  # Enhanced dataset model
    # checkpoint_path = 'checkpoints/xlmr_augmented_best.pt'  # Augmented (Banglish + Severity Fix) model
    checkpoint_path = 'checkpoints/xlmr_hard_best.pt'  # Hard Mode (Dropout 0.5, WD 0.1) model
    print(f"📁 Loading checkpoint: {checkpoint_path}")
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    model = MultiTaskXLMRRoberta().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print("✅ Model loaded successfully!\n")
    
    # Extract texts for prediction
    texts = [ex[0] for ex in TEST_EXAMPLES]
    
    print("🔮 Running predictions...")
    predictions = predict_batch(texts, model, tokenizer, device)
    print(f"✅ Predicted {len(predictions)} examples\n")
    
    # Evaluate
    results = evaluate_results(TEST_EXAMPLES, predictions)
    
    # ============================================================
    # PRINT RESULTS
    # ============================================================
    
    print("=" * 80)
    print("📊 OVERALL ACCURACY")
    print("=" * 80)
    total = results['total']
    print(f"Hate Type:    {results['hate_type_correct']}/{total} ({results['hate_type_correct']/total*100:.1f}%)")
    print(f"Target Group: {results['target_group_correct']}/{total} ({results['target_group_correct']/total*100:.1f}%)")
    print(f"Severity:     {results['severity_correct']}/{total} ({results['severity_correct']/total*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("📈 ACCURACY BY LANGUAGE")
    print("=" * 80)
    for lang, stats in sorted(results['language_breakdown'].items()):
        total_lang = stats['total']
        print(f"\n{lang.upper()} (n={total_lang}):")
        print(f"  Hate Type:    {stats['hate_type_correct']}/{total_lang} ({stats['hate_type_correct']/total_lang*100:.1f}%)")
        print(f"  Target Group: {stats['target_group_correct']}/{total_lang} ({stats['target_group_correct']/total_lang*100:.1f}%)")
        print(f"  Severity:     {stats['severity_correct']}/{total_lang} ({stats['severity_correct']/total_lang*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("🎯 ACCURACY BY HATE TYPE")
    print("=" * 80)
    for hate_type, stats in sorted(results['hate_type_by_class'].items()):
        acc = stats['correct'] / stats['total'] * 100
        print(f"{hate_type:20}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    print("\n" + "=" * 80)
    print("🎯 ACCURACY BY TARGET GROUP")
    print("=" * 80)
    for target, stats in sorted(results['target_group_by_class'].items()):
        acc = stats['correct'] / stats['total'] * 100
        print(f"{target:20}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    print("\n" + "=" * 80)
    print("⚠️ ACCURACY BY SEVERITY")
    print("=" * 80)
    for severity, stats in sorted(results['severity_by_class'].items()):
        acc = stats['correct'] / stats['total'] * 100
        print(f"{severity:10}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # ============================================================
    # DETAILED ERROR ANALYSIS
    # ============================================================
    
    print("\n" + "=" * 80)
    print("❌ ERROR ANALYSIS (First 20 misclassifications)")
    print("=" * 80)
    
    error_count = 0
    for i, ((text, lang, exp_ht, exp_tg, exp_sv), pred) in enumerate(zip(TEST_EXAMPLES, predictions)):
        if pred['hate_type_pred'] != exp_ht or pred['target_group_pred'] != exp_tg or pred['severity_pred'] != exp_sv:
            error_count += 1
            if error_count <= 20:
                print(f"\n❌ Error #{error_count} - {lang}")
                print(f"   Text: {text[:80]}")
                print(f"   Expected: HT={exp_ht}, TG={exp_tg}, SV={exp_sv}")
                print(f"   Predicted: HT={pred['hate_type_pred']}, TG={pred['target_group_pred']}, SV={pred['severity_pred']}")
                print(f"   Confidence: HT={pred['hate_type_conf']:.2f}, TG={pred['target_group_conf']:.2f}, SV={pred['severity_conf']:.2f}")
    
    print(f"\n\nTotal errors: {error_count}/{len(TEST_EXAMPLES)} ({error_count/len(TEST_EXAMPLES)*100:.1f}%)")
    
    # ============================================================
    # SAVE RESULTS TO CSV
    # ============================================================
    
    print("\n" + "=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80)
    
    # Create dataframe
    df_results = pd.DataFrame([
        {
            'text': text,
            'language': lang,
            'expected_hate_type': exp_ht,
            'expected_target_group': exp_tg,
            'expected_severity': exp_sv,
            'predicted_hate_type': pred['hate_type_pred'],
            'predicted_target_group': pred['target_group_pred'],
            'predicted_severity': pred['severity_pred'],
            'hate_type_conf': pred['hate_type_conf'],
            'target_group_conf': pred['target_group_conf'],
            'severity_conf': pred['severity_conf'],
            'hate_type_correct': pred['hate_type_pred'] == exp_ht,
            'target_group_correct': pred['target_group_pred'] == exp_tg,
            'severity_correct': pred['severity_pred'] == exp_sv,
        }
        for (text, lang, exp_ht, exp_tg, exp_sv), pred in zip(TEST_EXAMPLES, predictions)
    ])
    
    df_results.to_csv('comprehensive_test_results.csv', index=False)
    print("✅ Results saved to: comprehensive_test_results.csv")
    
    # Save summary
    with open('comprehensive_test_summary.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE TEST SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Examples: {total}\n")
        f.write(f"Hate Type Accuracy: {results['hate_type_correct']}/{total} ({results['hate_type_correct']/total*100:.1f}%)\n")
        f.write(f"Target Group Accuracy: {results['target_group_correct']}/{total} ({results['target_group_correct']/total*100:.1f}%)\n")
        f.write(f"Severity Accuracy: {results['severity_correct']}/{total} ({results['severity_correct']/total*100:.1f}%)\n")
    
    print("✅ Summary saved to: comprehensive_test_summary.txt")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
