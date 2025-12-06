import pandas as pd

df = pd.read_csv('dataset/UNIFIED_ALL_SPLIT_FILTERED.csv')
train = df[df['split']=='train']
ht_valid = train[train['hate_type'] != -1]

print('=== TRAINING SET HATE TYPE DISTRIBUTION (FILTERED) ===')
print(ht_valid['hate_type'].value_counts().sort_index())
print(f'\nTotal with valid hate_type: {len(ht_valid)}/{len(train)} ({len(ht_valid)/len(train)*100:.1f}%)')

gender_count = len(ht_valid[ht_valid['hate_type']==3])
print(f'\n✅ Gender class (3) now has {gender_count} training samples! (was 0)')

if gender_count > 0:
    print(f'\nSample gender-labeled texts:')
    gender = train[train['hate_type']==3]
    for idx, row in gender.head(10).iterrows():
        print(f'  [{row["source_dataset"]}] {row["text"][:70]}')
