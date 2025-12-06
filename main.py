import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

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

# =============================================================================
# SEVERITY KEYWORDS for smart severity detection
# =============================================================================

# Bengali/Banglish high-severity keywords (death threats, extreme profanity)
BENGALI_HIGH_SEVERITY = [
    'মরে যা', 'মইরা যা', 'moira ja', 'more ja', 'mori ja',
    'খুন', 'khun', 'kill', 'মেরে ফেল', 'mere fel',
    'চোদ', 'chod', 'chud', 'মাগি', 'magi', 'খানকি', 'khanki',
    'বোদা', 'boda', 'গুদ', 'gud', 'ধর্ষণ', 'rape',
    'হারামি', 'harami', 'শালা', 'shala', 'sala',
    'কুত্তা', 'kutta', 'জানোয়ার', 'janoyar', 'janwar',
    'suicide', 'আত্মহত্যা', 'die', 'death',
]

# Bengali/Banglish medium-severity keywords (insults, slurs)
BENGALI_MEDIUM_SEVERITY = [
    'বোকা', 'boka', 'pagol', 'পাগল', 'gadha', 'গাধা',
    'চোর', 'chor', 'মিথ্যাবাদী', 'liar',
    'stupid', 'idiot', 'fool', 'dumb',
    'worst', 'trash', 'garbage', 'pathetic',
    'hate', 'ঘৃণা', 'খারাপ', 'kharap', 'bad',
]

# English high-severity keywords
ENGLISH_HIGH_SEVERITY = [
    'kill yourself', 'kys', 'die', 'death threat', 'murder',
    'rape', 'suicide', 'hang yourself', 'shoot yourself',
    'fucking', 'f*ck', 'shit', 'bitch', 'cunt', 'whore',
    'n-word', 'nigger', 'faggot', 'retard',
    'terrorist', 'subhuman', 'vermin', 'cockroach',
]

# English medium-severity keywords  
ENGLISH_MEDIUM_SEVERITY = [
    'idiot', 'stupid', 'dumb', 'moron', 'fool',
    'trash', 'garbage', 'worthless', 'pathetic', 'loser',
    'hate', 'disgusting', 'ugly', 'fat', 'pig',
    'liar', 'corrupt', 'evil', 'criminal',
]


def compute_severity(text, base_severity=0):
    """
    Compute severity based on content analysis.
    0 = none, 1 = low, 2 = medium, 3 = high
    """
    if pd.isna(text):
        return base_severity
    
    text_lower = str(text).lower()
    
    # Check for high severity keywords
    for keyword in BENGALI_HIGH_SEVERITY + ENGLISH_HIGH_SEVERITY:
        if keyword.lower() in text_lower:
            return 3  # High
    
    # Check for medium severity keywords
    for keyword in BENGALI_MEDIUM_SEVERITY + ENGLISH_MEDIUM_SEVERITY:
        if keyword.lower() in text_lower:
            return max(2, base_severity)  # At least medium
    
    return base_severity


def map_bengali_hate_v1(df):
    """Map Bengali Hate v1.0 dataset"""
    # Fix delimiter issue - file is tab-separated
    if len(df.columns) == 1:
        df = pd.read_csv('dataset/bengali_ hate_v1.0.csv', sep='\t')
    
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
    
    # CRITICAL FIX: Remap gender slurs from Personal → Gender
    gender_keywords = ['খানকি', 'khanki', 'বেশ্যা', 'beshya', 'মাগি', 'magi', 'রান্ডি', 'randi', 'whore', 'slut', 'bitch']
    for keyword in gender_keywords:
        mask = out['text'].str.contains(keyword, case=False, na=False)
        out.loc[mask, 'hate_type'] = 3  # Remap to Gender
    
    # Smart severity: base on hate_type, then analyze content
    base_sev = np.where(out['hate_type'] > 0, 1, 0)
    out['severity'] = out.apply(lambda row: compute_severity(row['text'], base_sev[row.name]), axis=1)
    
    out['confidence'] = 1.0
    out['source_dataset'] = 'bengali_hate_v1'
    return out

def map_bengali_hate_v2(df):
    """Map Bengali Hate v2.0 dataset"""
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
    
    # CRITICAL FIX: Remap gender slurs from Personal → Gender
    gender_keywords = ['খানকি', 'khanki', 'বেশ্যা', 'beshya', 'মাগি', 'magi', 'রান্ডি', 'randi', 'whore', 'slut', 'bitch']
    for keyword in gender_keywords:
        mask = out['text'].str.contains(keyword, case=False, na=False)
        out.loc[mask, 'hate_type'] = 3  # Remap to Gender
    
    # Smart severity based on content
    base_sev = np.where(out['hate_type'] > 0, 1, 0)
    out['severity'] = out.apply(lambda row: compute_severity(row['text'], base_sev[row.name]), axis=1)
    
    out['confidence'] = 1.0
    out['source_dataset'] = 'bengali_hate_v2'
    return out

def map_blp25(df):
    """Map BLP25 Subtask 1B dataset (target group only)"""
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
    
    # Smart severity based on content (assume hate since it's a hate speech dataset)
    out['severity'] = out['text'].apply(lambda x: compute_severity(x, 1))
    
    out['confidence'] = 1.0
    out['source_dataset'] = 'blp25_subtask_1b'
    return out

def map_ethos(df):
    """Map ETHOS Binary dataset"""
    df.columns = df.columns.str.strip()
    def sev(x):
        """Map ETHOS score to severity (0-1 scale -> 0-3)"""
        try:
            x = float(x)
            if x <= 0.2:
                return 0  # None
            elif x <= 0.4:
                return 1  # Low
            elif x <= 0.7:
                return 2  # Medium
            else:
                return 3  # High
        except:
            return -1
    out = pd.DataFrame()
    out['id'] = np.arange(len(df))
    out['text'] = df['comment'] if 'comment' in df.columns else df.iloc[:,0]
    out['language'] = 'english'
    out['hate_type'] = -1
    out['target_group'] = -1
    
    # Use ETHOS score + content analysis
    base_sev = df['isHate'].apply(sev) if 'isHate' in df.columns else pd.Series([0] * len(df))
    out['severity'] = out.apply(
        lambda row: max(compute_severity(row['text'], 0), base_sev.iloc[row.name] if row.name < len(base_sev) else 0), 
        axis=1
    )
    
    out['confidence'] = 1.0
    out['source_dataset'] = 'ethos'
    return out

def map_olid(df):
    """Map OLID dataset"""
    df.columns = df.columns.str.strip()
    hate_type_map = {'OFF': 4, 'NOT': 0}
    target_map = {'IND': 1, 'GRP': 2, 'OTH': 0}
    out = pd.DataFrame()
    out['id'] = df['id'] if 'id' in df.columns else np.arange(len(df))
    out['text'] = df['tweet'] if 'tweet' in df.columns else df.iloc[:,1]
    out['language'] = 'english'
    out['hate_type'] = df['subtask_a'].map(hate_type_map).fillna(-1).astype(int) if 'subtask_a' in df.columns else -1
    out['target_group'] = df['subtask_c'].map(target_map).fillna(-1).astype(int) if 'subtask_c' in df.columns else -1
    
    # Smart severity: OFF = at least low, then analyze content
    is_offensive = df['subtask_a'].apply(lambda x: str(x).strip() == 'OFF') if 'subtask_a' in df.columns else pd.Series([False] * len(df))
    base_sev = np.where(is_offensive, 1, 0)
    out['severity'] = out.apply(lambda row: compute_severity(row['text'], base_sev[row.name]), axis=1)
    
    out['confidence'] = 1.0
    out['source_dataset'] = 'olid'
    return out

def map_toxic(df):
    """Map Toxic Comments dataset"""
    df.columns = df.columns.str.strip()
    lang_map = {'Bangla': 'bangla', 'English': 'english', 'Mixed': 'banglish'}
    out = pd.DataFrame()
    out['id'] = df['comment_id'] if 'comment_id' in df.columns else np.arange(len(df))
    out['text'] = df['comment_text'] if 'comment_text' in df.columns else df.iloc[:,0]
    out['language'] = df['language'].map(lang_map).fillna('banglish') if 'language' in df.columns else 'banglish'
    out['hate_type'] = -1
    out['target_group'] = -1
    
    # Smart severity: toxic = at least low, then analyze content
    is_toxic = df['label'].apply(lambda x: 'toxic' in str(x).lower()) if 'label' in df.columns else pd.Series([False] * len(df))
    base_sev = np.where(is_toxic, 1, 0)
    out['severity'] = out.apply(lambda row: compute_severity(row['text'], base_sev[row.name]), axis=1)
    
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
    # Fix: Bengali v1 is tab-separated
    df_bhv1 = pd.read_csv('dataset/bengali_ hate_v1.0.csv', sep='\t')
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
    print(f"Saved: dataset/UNIFIED_ALL.csv ({len(all_df)} rows)")

    # SPLIT THE UNIFIED CSV
    print("\nCreating train/val/test splits...")
    df = pd.read_csv("dataset/UNIFIED_ALL.csv")

    df['is_hate'] = ((df['hate_type'] > 0) | (df['severity'] > 0)).astype(int)
    train, temp = train_test_split(df, test_size=0.4, stratify=df[['language', 'is_hate']], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[['language', 'is_hate']], random_state=42)

    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'

    final = pd.concat([train, val, test])
    final.to_csv('dataset/UNIFIED_ALL_SPLIT.csv', index=False)
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"\nSeverity distribution:")
    print(final['severity'].value_counts().sort_index())
    print("\nDone!")