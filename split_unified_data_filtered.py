# split_unified_data_filtered.py - REMOVES toxic_comments dataset
# ===================================================================
# This version creates a cleaner dataset by EXCLUDING toxic_comments
# which only provides severity labels (no hate_type or target_group)
# ===================================================================

import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading UNIFIED_ALL.csv...")
df = pd.read_csv("dataset/UNIFIED_ALL.csv")

print(f"Original dataset: {len(df)} samples")
print("\nSource dataset distribution:")
print(df['source_dataset'].value_counts())

# ===================================================================
# FILTER OUT toxic_comments dataset
# ===================================================================
print("\n🔧 Filtering out toxic_comments dataset (only has severity labels)...")
df_filtered = df[df['source_dataset'] != 'toxic_comments'].copy()

print(f"\nFiltered dataset: {len(df_filtered)} samples")
print(f"Removed: {len(df) - len(df_filtered)} samples ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")

print("\nRemaining source datasets:")
print(df_filtered['source_dataset'].value_counts())

# ===================================================================
# ANALYZE LABEL COMPLETENESS
# ===================================================================
print("\n" + "="*60)
print("LABEL COMPLETENESS ANALYSIS (After Filtering)")
print("="*60)

ht_valid = df_filtered[df_filtered['hate_type'] != -1]
tg_valid = df_filtered[df_filtered['target_group'] != -1]
sv_valid = df_filtered[df_filtered['severity'] != -1]

print(f"\nHate Type labels:   {len(ht_valid)}/{len(df_filtered)} ({len(ht_valid)/len(df_filtered)*100:.1f}%)")
print(f"Target Group labels: {len(tg_valid)}/{len(df_filtered)} ({len(tg_valid)/len(df_filtered)*100:.1f}%)")
print(f"Severity labels:     {len(sv_valid)}/{len(df_filtered)} ({len(sv_valid)/len(df_filtered)*100:.1f}%)")

print("\nHate Type distribution:")
print(ht_valid['hate_type'].value_counts().sort_index())

print("\nTarget Group distribution:")
print(tg_valid['target_group'].value_counts().sort_index())

print("\nSeverity distribution:")
print(sv_valid['severity'].value_counts().sort_index())

# ===================================================================
# CREATE is_hate FLAG
# ===================================================================
df_filtered['is_hate'] = ((df_filtered['hate_type'] > 0) | (df_filtered['severity'] > 0)).astype(int)

print(f"\nis_hate distribution:")
print(df_filtered['is_hate'].value_counts())

# ===================================================================
# STRATIFIED SPLIT (60/20/20)
# ===================================================================
print("\n" + "="*60)
print("CREATING STRATIFIED SPLITS")
print("="*60)

# Use language × is_hate for stratification
df_filtered['strat_key'] = df_filtered['language'] + '_' + df_filtered['is_hate'].astype(str)

print("\nStratification key distribution:")
print(df_filtered['strat_key'].value_counts())

train, temp = train_test_split(
    df_filtered, 
    test_size=0.4, 
    stratify=df_filtered['strat_key'], 
    random_state=42
)

val, test = train_test_split(
    temp, 
    test_size=0.5,  # 0.5 of 0.4 = 0.2 of total
    stratify=temp['strat_key'], 
    random_state=42
)

train['split'] = 'train'
val['split'] = 'val'
test['split'] = 'test'

print(f"\nTrain: {len(train)} samples ({len(train)/len(df_filtered)*100:.1f}%)")
print(f"Val:   {len(val)} samples ({len(val)/len(df_filtered)*100:.1f}%)")
print(f"Test:  {len(test)} samples ({len(test)/len(df_filtered)*100:.1f}%)")

# ===================================================================
# COMBINE AND SAVE
# ===================================================================
final = pd.concat([train, val, test])

# Drop temporary stratification key
final = final.drop(columns=['strat_key'])

output_path = 'dataset/UNIFIED_ALL_SPLIT_FILTERED.csv'
final.to_csv(output_path, index=False)

print(f"\n✅ Saved filtered dataset to: {output_path}")

# ===================================================================
# VALIDATION: Check label completeness in each split
# ===================================================================
print("\n" + "="*60)
print("VALIDATION: Label Completeness by Split")
print("="*60)

for split_name in ['train', 'val', 'test']:
    split_df = final[final['split'] == split_name]
    ht_valid = len(split_df[split_df['hate_type'] != -1])
    tg_valid = len(split_df[split_df['target_group'] != -1])
    sv_valid = len(split_df[split_df['severity'] != -1])
    
    print(f"\n{split_name.upper()} ({len(split_df)} samples):")
    print(f"  Hate Type:    {ht_valid}/{len(split_df)} ({ht_valid/len(split_df)*100:.1f}%)")
    print(f"  Target Group: {tg_valid}/{len(split_df)} ({tg_valid/len(split_df)*100:.1f}%)")
    print(f"  Severity:     {sv_valid}/{len(split_df)} ({sv_valid/len(split_df)*100:.1f}%)")

print("\n" + "="*60)
print("🎯 FILTERED DATASET SUMMARY")
print("="*60)
print(f"✅ Removed toxic_comments dataset (30K+ samples with incomplete labels)")
print(f"✅ Kept {len(df_filtered)} high-quality samples with better label coverage")
ht_coverage = (len(df_filtered[df_filtered['hate_type'] != -1]) / len(df_filtered)) * 100
tg_coverage = (len(df_filtered[df_filtered['target_group'] != -1]) / len(df_filtered)) * 100
print(f"✅ Hate Type coverage improved from ~25% to ~{ht_coverage:.0f}%")
print(f"✅ Target Group coverage improved from ~11% to ~{tg_coverage:.0f}%")
print("\n⚠️ Next steps:")
print("   1. Update main.ipynb Cell 1 to load 'UNIFIED_ALL_SPLIT_FILTERED.csv'")
print("   2. Recompute class weights (will be different)")
print("   3. Retrain model with cleaner data")
print("="*60)
