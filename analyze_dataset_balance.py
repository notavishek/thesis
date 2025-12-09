import pandas as pd

def analyze_balance():
    print("🔄 Loading UNIFIED_BALANCED.csv...")
    df = pd.read_csv('dataset/UNIFIED_BALANCED.csv')
    
    languages = ['english', 'bangla', 'banglish']
    
    print(f"\n📊 TOTAL SAMPLES: {len(df)}")
    
    for lang in languages:
        print(f"\n{'='*40}")
        print(f"🌍 LANGUAGE: {lang.upper()}")
        print(f"{'='*40}")
        
        lang_df = df[df['language'] == lang]
        print(f"Total: {len(lang_df)}")
        
        # Hate Type Breakdown
        print("\n--- Hate Type Distribution ---")
        ht_counts = lang_df['hate_type'].value_counts().sort_index()
        # Map codes to names if possible (0: Not Hate, 1: Political, 2: Religious, 3: Gender, 4: Personal, 5: Geopolitical)
        ht_names = {0: 'Not Hate', 1: 'Political', 2: 'Religious', 3: 'Gender', 4: 'Personal', 5: 'Geopolitical', -1: 'Missing'}
        for code, count in ht_counts.items():
            name = ht_names.get(code, str(code))
            print(f"{name:<15}: {count:>5} ({count/len(lang_df)*100:.1f}%)")

        # Target Group Breakdown
        print("\n--- Target Group Distribution ---")
        tg_counts = lang_df['target_group'].value_counts().sort_index()
        tg_names = {0: 'None/Other', 1: 'Individual', 2: 'Group', 3: 'Community', -1: 'Missing'}
        for code, count in tg_counts.items():
            name = tg_names.get(code, str(code))
            print(f"{name:<15}: {count:>5} ({count/len(lang_df)*100:.1f}%)")

        # Severity Breakdown
        print("\n--- Severity Distribution ---")
        sv_counts = lang_df['severity'].value_counts().sort_index()
        sv_names = {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High', -1: 'Missing'}
        for code, count in sv_counts.items():
            name = sv_names.get(code, str(code))
            print(f"{name:<15}: {count:>5} ({count/len(lang_df)*100:.1f}%)")

if __name__ == "__main__":
    analyze_balance()
