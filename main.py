import pandas as pd
import numpy as np
import re
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

# =============================================================================
# HATE TYPE KEYWORDS for smart classification
# =============================================================================

GENDER_KEYWORDS = [
    'খানকি', 'khanki', 'বেশ্যা', 'beshya', 'মাগি', 'magi', 'রান্ডি', 'randi', 
    'whore', 'slut', 'bitch', 'cunt', 'ধর্ষণ', 'rape', 'chud', 'chod', 'চোদ',
    'nude', 'sex', 'xxx', 'porn', 'বৌ', 'bou', 'mey', 'meye', 'girl', 'woman',
    'মহিলা', 'নারী', 'মেয়ে', 'boda', 'বোদা', 'gud', 'গুদ', 'faggot', 'gay', 'lesbian',
    'hijra', 'হিজড়া', 'trans', 'shemale'
]

RELIGIOUS_KEYWORDS = [
    'মালাউন', 'malaun', 'কাফের', 'kafer', 'kafir', 'হিন্দু', 'hindu', 'muslim', 'মুসলমান',
    'nastik', 'নাস্তিক', 'jihad', 'জিহাদ', 'allah', 'god', 'bhagwan', 'ram', 'nabi',
    'ধর্ম', 'religion', 'islam', 'christian', 'buddhist', 'jew', 'jewish', 'yahudi',
    'mullah', 'মোল্লা', 'pujari', 'পুরোহিত', 'আল্লাহ', 'khoda', 'ভগবান'
]

POLITICAL_KEYWORDS = [
    'bal', 'bnp', 'awami', 'league', 'chatra', 'shibir', 'hasina', 'khaleda',
    'politics', 'net', 'vote', 'election', 'mp', 'minister', 'government', 'sarkar',
    'league', 'lig', 'dal', 'party', 'modi', 'trump', 'biden', 'putin', 'leader',
    'neta', 'নেতা', 'চেয়ারম্যান', 'chairman', 'taka', 'টাকা', 'corruption', 'durniti',
    'desh', 'country', 'india', 'pakistan', 'bharat', 'bangladesh'
]

# Compile Regex Patterns for Performance and Accuracy (Whole Word Matching)
def compile_regex(keywords):
    # Sort by length to match longest first
    keywords = sorted(keywords, key=len, reverse=True)
    # Escape keywords
    escaped = [re.escape(k) for k in keywords]
    # Create pattern: \b(k1|k2|...)\b
    pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

RE_GENDER = compile_regex(GENDER_KEYWORDS)
RE_RELIGIOUS = compile_regex(RELIGIOUS_KEYWORDS)
RE_POLITICAL = compile_regex(POLITICAL_KEYWORDS)

RE_COMMUNITY = compile_regex([
    'muslims', 'hindus', 'christians', 'jews', 'women', 'men', 'gays', 'blacks', 'whites', 
    'asians', 'indians', 'bengalis', 'lokjon', 'manus', 'society', 'desh', 'country',
    'india', 'bharat', 'bangladesh', 'pakistan', 'people', 'public', 'জনগণ', 'মানুষ', 'সমাজ'
])

RE_GROUP = compile_regex([
    'party', 'league', 'team', 'group', 'organization', 'company', 'government', 'police', 'army', 'media', 'rab', 'police'
])

RE_INDIVIDUAL = compile_regex([
    'you', 'ur', 'your', 'tui', 'tumi', 'tor', 'tomar', 'tomi', 'tmi', 
    '@user', 'user', 'তুই', 'তুমি', 'তোর', 'তোমার'
])


def infer_hate_type(text, current_label=-1, severity=1):
    """
    Infer hate type from text using regex if current_label is missing or generic.
    Only infers if severity > 0 (to avoid false positives in safe text).
    Returns: 1=Political, 2=Religious, 3=Gender, 4=Personal, 0=Not Hate
    """
    if pd.isna(text):
        return current_label
        
    # If severity is 0, force Not Hate (unless current_label says otherwise, but we trust severity here)
    if severity == 0 and current_label <= 0:
        return 0
        
    text = str(text)
    
    # If we already have a specific label (1, 2, 3), keep it.
    if current_label in [1, 2, 3]:
        return current_label
        
    # Check Keywords (Strongest signal)
    if RE_GENDER.search(text): return 3
    if RE_RELIGIOUS.search(text): return 2
    if RE_POLITICAL.search(text): return 1
        
    # If no keywords found, return original label (likely 4 or -1)
    return current_label

def infer_target_group(text, current_label=-1, severity=1):
    """
    Infer target group from text using regex if current_label is missing.
    Only infers if severity > 0.
    Returns: 1=Individual, 2=Group/Org, 3=Community, 0=None
    """
    if pd.isna(text):
        return current_label
        
    if severity == 0 and current_label <= 0:
        return 0
        
    if current_label != -1:
        return current_label
        
    text = str(text)
    
    if RE_COMMUNITY.search(text): return 3
    if RE_GROUP.search(text): return 2
    if RE_INDIVIDUAL.search(text): return 1
        
    return -1


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
    
    # Smart severity: base on hate_type, then analyze content
    base_sev = np.where(out['hate_type'] > 0, 1, 0)
    out['severity'] = out.apply(lambda row: compute_severity(row['text'], base_sev[row.name]), axis=1)
    
    # Infer target group instead of leaving as -1
    out['target_group'] = out.apply(lambda row: infer_target_group(row['text'], -1, row['severity']), axis=1)
    
    # CRITICAL FIX: Remap gender slurs from Personal → Gender
    # Use regex for better matching
    def remap_gender(row):
        if RE_GENDER.search(str(row['text'])):
            return 3
        return row['hate_type']
        
    out['hate_type'] = out.apply(remap_gender, axis=1)
    
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
    def remap_gender(row):
        if RE_GENDER.search(str(row['text'])):
            return 3
        return row['hate_type']
    out['hate_type'] = out.apply(remap_gender, axis=1)
    
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
        'None': 0, 'none': 0  # Map None to No Target
    }
    out = pd.DataFrame()
    out['id'] = np.arange(len(df))
    out['text'] = df['text'] if 'text' in df.columns else df.iloc[:,0]
    out['language'] = 'bangla'
    
    # If label is None, it's Not Hate (0). Otherwise, try to infer hate type.
    out['hate_type'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'none' else -1)
    
    out['target_group'] = df['label'].map(target_map).fillna(-1).astype(int) if 'label' in df.columns else -1
    
    # Smart severity based on content (assume hate since it's a hate speech dataset, unless None)
    out['severity'] = out.apply(lambda row: 0 if row['hate_type'] == 0 else compute_severity(row['text'], 1), axis=1)
    
    # Infer hate type using severity check
    out['hate_type'] = out.apply(lambda row: infer_hate_type(row['text'], row['hate_type'], row['severity']), axis=1)
    
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
    
    # Use ETHOS score + content analysis
    base_sev = df['isHate'].apply(sev) if 'isHate' in df.columns else pd.Series([0] * len(df))
    out['severity'] = out.apply(
        lambda row: max(compute_severity(row['text'], 0), base_sev.iloc[row.name] if row.name < len(base_sev) else 0), 
        axis=1
    )
    
    # Infer hate type and target group instead of leaving as -1
    out['hate_type'] = out.apply(lambda row: infer_hate_type(row['text'], -1, row['severity']), axis=1)
    out['target_group'] = out.apply(lambda row: infer_target_group(row['text'], -1, row['severity']), axis=1)
    
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
    
    # Initial mapping
    initial_hate_type = df['subtask_a'].map(hate_type_map).fillna(-1).astype(int) if 'subtask_a' in df.columns else -1
    
    # Smart severity: OFF = at least low, then analyze content
    is_offensive = df['subtask_a'].apply(lambda x: str(x).strip() == 'OFF') if 'subtask_a' in df.columns else pd.Series([False] * len(df))
    base_sev = np.where(is_offensive, 1, 0)
    out['severity'] = out.apply(lambda row: compute_severity(row['text'], base_sev[row.name]), axis=1)
    
    # Refine hate type using keywords (e.g. if OFF -> Personal, but contains 'muslim', change to Religious)
    out['hate_type'] = df.apply(lambda row: infer_hate_type(row['tweet'], initial_hate_type.iloc[row.name] if row.name < len(initial_hate_type) else -1, out['severity'].iloc[row.name]), axis=1)
    
    out['target_group'] = df['subtask_c'].map(target_map).fillna(-1).astype(int) if 'subtask_c' in df.columns else -1
    
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
    
    # Smart severity: toxic = at least low, then analyze content
    is_toxic = df['label'].apply(lambda x: 'toxic' in str(x).lower()) if 'label' in df.columns else pd.Series([False] * len(df))
    base_sev = np.where(is_toxic, 1, 0)
    out['severity'] = out.apply(lambda row: compute_severity(row['text'], base_sev[row.name]), axis=1)
    
    def get_hate_type(row):
        text = str(row['comment_text']).lower()
        label = str(row['label']).lower()
        severity = row['severity']
        
        # 1. Check Keywords (Strongest signal) - Use the shared function logic
        inferred = infer_hate_type(text, -1, severity)
        if inferred != -1:
            return inferred
            
        # 2. Check Label
        if 'threat' in label: return 4 # Personal
        if 'insult' in label: return 4 # Personal
        if 'obscene' in label: return 4 # Personal
        if 'severe_toxic' in label: return 4 # Personal
        
        # Default to Personal for generic toxic/hate_speech if no specific keyword found
        if 'toxic' in label or 'hate_speech' in label:
            return 4
            
        return -1

    # We need to pass the severity we just calculated
    out['hate_type'] = out.apply(lambda row: get_hate_type({'comment_text': row['text'], 'label': df.iloc[row.name]['label'], 'severity': row['severity']}), axis=1)
    
    # Infer target group
    out['target_group'] = out.apply(lambda row: infer_target_group(row['text'], -1, row['severity']), axis=1)
    
    out['confidence'] = 1.0
    out['source_dataset'] = 'toxic_comments'
    return out

def pad_missing(df):
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = -1 if 'hate_type' in col or 'target_group' in col or 'severity' in col else ''
    return df

def enforce_consistency(df):
    """
    Enforce logical consistency rules across the dataset.
    1. If severity is 0, hate_type and target_group must be 0.
    2. If hate_type or target_group > 0, severity must be at least 1.
    """
    # Rule 1: Safe samples must have safe labels
    mask_safe = (df['severity'] == 0)
    df.loc[mask_safe, 'hate_type'] = 0
    df.loc[mask_safe, 'target_group'] = 0
    
    # Rule 2: Hate samples must have hate severity
    # If original dataset said it's hate (hate_type > 0), we trust it and bump severity.
    mask_hate = (df['hate_type'] > 0) | (df['target_group'] > 0)
    df.loc[mask_hate & (df['severity'] == 0), 'severity'] = 1
    
    return df

def save_unified(df, name):
    df = enforce_consistency(df)
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
    final.to_csv('dataset/UNIFIED_ALL_SPLIT_ENHANCED.csv', index=False)
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Saved to: dataset/UNIFIED_ALL_SPLIT_ENHANCED.csv")
    print(f"\nSeverity distribution:")
    print(final['severity'].value_counts().sort_index())
    print("\nDone!")