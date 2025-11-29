import pandas as pd
import numpy as np
from pathlib import Path

UNIFIED_COLUMNS = [
    'id',
    'text',
    'language',
    'hate_type',
    'target_group',
    'severity',
    'confidence',
    'source_dataset'
]

def map_bengali_hate_v1(df):
    df.columns = df.columns.str.strip()
    label_map = {
        'Political': 1, 'political': 1,
        'Religious': 2, 'religious': 2,
        'Gender abusive': 3, 'gender': 3, 'Gender': 3,
        'Personal': 4, 'personal': 4,
        'Geopolitical': 5, 'geopolitical': 5,
        'other': 0, 'Other': 0
    }
    out = pd.DataFrame()
    out['id'] = np.arange(len(df))
    out['text'] = df['text'] if 'text' in df.columns else df.iloc[:,0]
    out['language'] = 'bangla'
    out['hate_type'] = df['label'].map(label_map).fillna(-1).astype(int) if 'label' in df.columns else -1
    out['target_group'] = -1
    out['severity'] = 1
    out['confidence'] = 1.0
    out['source_dataset'] = 'bengali_hate_v1'
    return out

def map_bengali_hate_v2(df):
    df.columns = df.columns.str.strip()
    label_map = {
        'Political': 1, 'political': 1,
        'Religious': 2, 'religious': 2,
        'Gender abusive': 3, 'gender': 3, 'Gender': 3,
        'Personal': 4, 'personal': 4,
        'Geopolitical': 5, 'geopolitical': 5,
        'other': 0, 'Other': 0
    }
    target_map = {0: 1, 1: 2, 2: 3}
    out = pd.DataFrame()
    out['id'] = np.arange(len(df))
    out['text'] = df['text'] if 'text' in df.columns else df.iloc[:,0]
    out['language'] = 'bangla'
    out['hate_type'] = df['label'].map(label_map).fillna(-1).astype(int) if 'label' in df.columns else -1
    out['target_group'] = df['target'].map(target_map).fillna(-1).astype(int) if 'target' in df.columns else -1
    out['severity'] = np.where(out['hate_type'] != 0, 1, 0)
    out['confidence'] = 1.0
    out['source_dataset'] = 'bengali_hate_v2'
    return out

def map_blp25(df):
    df.columns = df.columns.str.strip()
    target_map = {
        'Individual': 1,
        'Organization': 2, 'Group': 2, 'Group/Organization': 2,
        'Community': 3,
        'Other': 0, 'other': 0,
    }
    out = pd.DataFrame()
    out['id'] = np.arange(len(df))
    out['text'] = df['text'] if 'text' in df.columns else df.iloc[:,0]
    out['language'] = 'bangla'
    out['hate_type'] = -1
    out['target_group'] = df['label'].map(target_map).fillna(-1).astype(int) if 'label' in df.columns else -1
    out['severity'] = 1
    out['confidence'] = 1.0
    out['source_dataset'] = 'blp25_subtask_1b'
    return out

def map_ethos(df):
    df.columns = df.columns.str.strip()
    def sev(x):
        try:
            x = float(x)
            if x <= 0.3:
                return 0
            elif x <= 0.6:
                return 1
            else:
                return 2
        except:
            return -1
    out = pd.DataFrame()
    out['id'] = np.arange(len(df))
    out['text'] = df['comment'] if 'comment' in df.columns else df.iloc[:,0]
    out['language'] = 'english'
    out['hate_type'] = -1
    out['target_group'] = -1
    out['severity'] = df['isHate'].apply(sev) if 'isHate' in df.columns else -1
    out['confidence'] = 1.0
    out['source_dataset'] = 'ethos'
    return out

def map_olid(df):
    df.columns = df.columns.str.strip()
    hate_type_map = {'OFF': 4, 'NOT': 0}
    target_map = {'IND': 1, 'GRP': 2, 'OTH': 0}
    out = pd.DataFrame()
    out['id'] = df['id'] if 'id' in df.columns else np.arange(len(df))
    out['text'] = df['tweet'] if 'tweet' in df.columns else df.iloc[:,1] # usually tweet is col 2
    out['language'] = 'english'
    out['hate_type'] = df['subtask_a'].map(hate_type_map).fillna(-1).astype(int) if 'subtask_a' in df.columns else -1
    out['target_group'] = df['subtask_c'].map(target_map).fillna(-1).astype(int) if 'subtask_c' in df.columns else -1
    out['severity'] = df['subtask_a'].apply(lambda x: 1 if str(x).strip() == 'OFF' else 0) if 'subtask_a' in df.columns else -1
    out['confidence'] = 1.0
    out['source_dataset'] = 'olid'
    return out

def map_toxic(df):
    df.columns = df.columns.str.strip()
    lang_map = {'Bangla': 'bangla', 'English': 'english', 'Mixed': 'banglish'}
    out = pd.DataFrame()
    out['id'] = df['comment_id'] if 'comment_id' in df.columns else np.arange(len(df))
    out['text'] = df['comment_text'] if 'comment_text' in df.columns else df.iloc[:,0]
    out['language'] = df['language'].map(lang_map).fillna('other') if 'language' in df.columns else 'other'
    out['hate_type'] = -1
    out['target_group'] = -1
    out['severity'] = df['label'].apply(lambda x: 1 if 'toxic' in str(x).lower() else 0) if 'label' in df.columns else -1
    out['confidence'] = 1.0
    out['source_dataset'] = 'toxic_comments'
    return out

def pad_missing(df):
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = -1 if 'hate_type' in col or 'target_group' in col or 'severity' in col else ''
    return df

def save_unified(df, name):
    df = df[UNIFIED_COLUMNS]
    df.to_csv(f'dataset/unified_{name}.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    df_bhv1 = pd.read_csv('dataset/bengali_ hate_v1.0.csv')
    bhv1 = map_bengali_hate_v1(df_bhv1)
    bhv1 = pad_missing(bhv1)
    save_unified(bhv1, 'bhv1')

    df_bhv2 = pd.read_csv('dataset/bengali_hate_v2.0.csv')
    bhv2 = map_bengali_hate_v2(df_bhv2)
    bhv2 = pad_missing(bhv2)
    save_unified(bhv2, 'bhv2')

    df_blp25 = pd.read_csv('dataset/blp25_hatespeech_subtask_1B_dev.tsv', sep='\t')
    blp25 = map_blp25(df_blp25)
    blp25 = pad_missing(blp25)
    save_unified(blp25, 'blp25')

    df_ethos = pd.read_csv('dataset/Ethos_Dataset_Binary.csv', sep=';')
    ethos = map_ethos(df_ethos)
    ethos = pad_missing(ethos)
    save_unified(ethos, 'ethos')

    df_olid = pd.read_csv('dataset/olid-training-v1.0.tsv', sep='\t')
    olid = map_olid(df_olid)
    olid = pad_missing(olid)
    save_unified(olid, 'olid')

    df_toxic = pd.read_csv('dataset/toxic_comments_dataset.csv')
    toxic = map_toxic(df_toxic)
    toxic = pad_missing(toxic)
    save_unified(toxic, 'toxic')

    all_df = pd.concat([bhv1, bhv2, blp25, ethos, olid, toxic], ignore_index=True)
    all_df.to_csv('dataset/UNIFIED_ALL.csv', index=False, encoding='utf-8')

    if __name__ == "__main__":
    # ... all your existing main.py code for mapping & concatenation ...

    # SPLIT THE UNIFIED CSV (post-processing phase)
    df = pd.read_csv("dataset/UNIFIED_ALL.csv")

    df['is_hate'] = ((df['hate_type'] > 0) | (df['severity'] > 0)).astype(int)
    train, temp = train_test_split(df, test_size=0.4, stratify=df[['language', 'is_hate']], random_state=42)
    val, test = train_test_split(temp, test_size=0.625, stratify=temp[['language', 'is_hate']], random_state=42)

    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'

    final = pd.concat([train, val, test])
    final.to_csv('dataset/UNIFIED_ALL_SPLIT.csv', index=False)