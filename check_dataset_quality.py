import pandas as pd
import re

def check_dataset():
    print("Loading dataset...")
    df = pd.read_csv('dataset/UNIFIED_ALL_SPLIT_ENHANCED.csv')
    
    print(f"Total rows: {len(df)}")
    
    # 1. Consistency Checks
    print("\n--- 1. Consistency Checks ---")
    
    # Rule: If severity is 0, hate_type and target_group must be 0
    inconsistent_safe = df[
        (df['severity'] == 0) & 
        ((df['hate_type'] != 0) | (df['target_group'] != 0))
    ]
    print(f"Safe samples (sev=0) with hate labels: {len(inconsistent_safe)}")
    if len(inconsistent_safe) > 0:
        print(inconsistent_safe[['text', 'severity', 'hate_type', 'target_group']].head())

    # Rule: If hate_type > 0 or target_group > 0, severity must be > 0
    inconsistent_hate = df[
        ((df['hate_type'] > 0) | (df['target_group'] > 0)) & 
        (df['severity'] == 0)
    ]
    print(f"Hate samples (type/target>0) with sev=0: {len(inconsistent_hate)}")
    if len(inconsistent_hate) > 0:
        print(inconsistent_hate[['text', 'severity', 'hate_type', 'target_group']].head())

    # 2. Keyword False Positive Check
    print("\n--- 2. Keyword False Positive Check ---")
    # Check for 'net' (Political) matching inside other words
    # We suspect 'net' might match 'internet', 'network', etc.
    
    political_suspects = df[
        (df['hate_type'] == 1) & 
        (df['text'].str.contains('net', case=False, na=False)) &
        (~df['text'].str.contains(r'\bnet\b', case=False, regex=True, na=False))
    ]
    print(f"Potential 'net' false positives (substring match but not word match): {len(political_suspects)}")
    if len(political_suspects) > 0:
        print(political_suspects['text'].head().tolist())

    # Check for 'bal' (Political? No, 'bal' is usually profanity in Bangla, but I put it in Political?)
    # Wait, 'bal' in Bangla is often used as "Chaal" (BNP-Jamaat-Bal), but 'bal' is also "Pubic Hair" (Profanity).
    # In my code: POLITICAL_KEYWORDS = ['bal', ...]
    # If 'bal' matches 'global', that's bad.
    bal_suspects = df[
        (df['hate_type'] == 1) & 
        (df['text'].str.contains('bal', case=False, na=False)) &
        (~df['text'].str.contains(r'\bbal\b', case=False, regex=True, na=False))
    ]
    print(f"Potential 'bal' false positives: {len(bal_suspects)}")
    if len(bal_suspects) > 0:
        print(bal_suspects['text'].head().tolist())

    # 3. Missing Label Stats
    print("\n--- 3. Missing Label Stats (Severity > 0) ---")
    missing_hate = df[(df['hate_type'] == -1) & (df['severity'] > 0)]
    print(f"Missing Hate Type: {len(missing_hate)} / {len(df[df['severity']>0])}")
    print("By Dataset:")
    print(missing_hate['source_dataset'].value_counts())

    missing_target = df[(df['target_group'] == -1) & (df['severity'] > 0)]
    print(f"\nMissing Target Group: {len(missing_target)} / {len(df[df['severity']>0])}")
    print("By Dataset:")
    print(missing_target['source_dataset'].value_counts())

if __name__ == "__main__":
    check_dataset()
