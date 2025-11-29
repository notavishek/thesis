# split_unified_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/UNIFIED_ALL.csv")

df['is_hate'] = ((df['hate_type'] > 0) | (df['severity'] > 0)).astype(int)  # Adjust if needed

train, temp = train_test_split(df, test_size=0.4, stratify=df[['language', 'is_hate']], random_state=42)
val, test = train_test_split(temp, test_size=0.625, stratify=temp[['language', 'is_hate']], random_state=42)

train['split'] = 'train'
val['split'] = 'val'
test['split'] = 'test'

final = pd.concat([train, val, test])
final.to_csv('dataset/UNIFIED_ALL_SPLIT.csv', index=False)