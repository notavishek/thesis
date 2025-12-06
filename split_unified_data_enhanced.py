# split_unified_data_enhanced.py - Create splits for ENHANCED dataset
# =====================================================================
# Uses UNIFIED_ALL_ENHANCED.csv with auto-labeled toxic_comments
# =====================================================================

import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading UNIFIED_ALL_ENHANCED.csv...")
df = pd.read_csv("dataset/UNIFIED_ALL_ENHANCED.csv")

print(f"Total samples: {len(df)}")
print("\n📊 Label Coverage:")
ht_valid = len(df[df['hate_type'] != -1])
tg_valid = len(df[df['target_group'] != -1])
print(f"  hate_type:    {ht_valid}/{len(df)} ({ht_valid/len(df)*100:.1f}%)")
print(f"  target_group: {tg_valid}/{len(df)} ({tg_valid/len(df)*100:.1f}%)")

# Create is_hate flag
df['is_hate'] = ((df['hate_type'] > 0) | (df['severity'] > 0)).astype(int)

print(f"\nis_hate distribution:")
print(df['is_hate'].value_counts())

# Stratified split by language × is_hate
df['strat_key'] = df['language'] + '_' + df['is_hate'].astype(str)

train, temp = train_test_split(
    df, 
    test_size=0.4, 
    stratify=df['strat_key'], 
    random_state=42
)

val, test = train_test_split(
    temp, 
    test_size=0.5,
    stratify=temp['strat_key'], 
    random_state=42
)

train['split'] = 'train'
val['split'] = 'val'
test['split'] = 'test'

print(f"\nTrain: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
print(f"Val:   {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
print(f"Test:  {len(test)} samples ({len(test)/len(df)*100:.1f}%)")

# Combine
final = pd.concat([train, val, test])
final = final.drop(columns=['strat_key'])

output_path = 'dataset/UNIFIED_ALL_SPLIT_ENHANCED.csv'
final.to_csv(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")

# Validation
print("\n" + "="*60)
print("LABEL COMPLETENESS BY SPLIT")
print("="*60)

for split_name in ['train', 'val', 'test']:
    split_df = final[final['split'] == split_name]
    ht_valid = len(split_df[split_df['hate_type'] != -1])
    tg_valid = len(split_df[split_df['target_group'] != -1])
    
    print(f"\n{split_name.upper()} ({len(split_df)} samples):")
    print(f"  hate_type:    {ht_valid}/{len(split_df)} ({ht_valid/len(split_df)*100:.1f}%)")
    print(f"  target_group: {tg_valid}/{len(split_df)} ({tg_valid/len(split_df)*100:.1f}%)")

print("\n" + "="*60)
print("✅ ENHANCED DATASET READY FOR TRAINING!")
print("="*60)
