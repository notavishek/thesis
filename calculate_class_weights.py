import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_weights():
    print("🔄 Loading UNIFIED_BALANCED.csv...")
    df = pd.read_csv('dataset/UNIFIED_BALANCED.csv')
    
    # Filter out -1 (missing labels)
    ht_y = df[df['hate_type'] != -1]['hate_type']
    tg_y = df[df['target_group'] != -1]['target_group']
    sv_y = df[df['severity'] != -1]['severity']
    
    # Calculate weights
    # formula: n_samples / (n_classes * n_samples_j)
    ht_weights = compute_class_weight('balanced', classes=np.unique(ht_y), y=ht_y)
    tg_weights = compute_class_weight('balanced', classes=np.unique(tg_y), y=tg_y)
    sv_weights = compute_class_weight('balanced', classes=np.unique(sv_y), y=sv_y)
    
    print("\n⚖️ CALCULATED CLASS WEIGHTS (Copy these to training script):")
    
    print(f"\n# Hate Type Weights (Classes: {np.unique(ht_y)})")
    print(f"ht_weights = torch.tensor({list(ht_weights)}, device=device, dtype=torch.float)")
    
    print(f"\n# Target Group Weights (Classes: {np.unique(tg_y)})")
    print(f"tg_weights = torch.tensor({list(tg_weights)}, device=device, dtype=torch.float)")
    
    print(f"\n# Severity Weights (Classes: {np.unique(sv_y)})")
    print(f"sv_weights = torch.tensor({list(sv_weights)}, device=device, dtype=torch.float)")

if __name__ == "__main__":
    calculate_weights()
