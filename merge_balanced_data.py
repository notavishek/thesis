import pandas as pd
import numpy as np

def merge_datasets():
    print("🔄 Loading datasets...")
    
    # 1. Load AUGMENTED unified data (Contains Transliterated Banglish)
    original_df = pd.read_csv('dataset/UNIFIED_ALL_SPLIT_AUGMENTED.csv')
    print(f"Original (Augmented) data shape: {original_df.shape}")
    
    # 2. Load Neutral Boost data (35k)
    neutral_df = pd.read_csv('dataset/neutral_boost_large.csv')
    print(f"Neutral Boost data shape: {neutral_df.shape}")
    neutral_df['is_hate'] = 0
    
    # 3. Load NEW Specific Hate Gen (102k) - The "Excellent" Fix
    gen_df = pd.read_csv('dataset/specific_hate_gen.csv')
    print(f"Specific Hate Gen data shape: {gen_df.shape}")
    gen_df['is_hate'] = 1
    
    # Normalize language codes
    lang_map = {
        'bn': 'bangla',
        'en': 'english',
        'bl': 'banglish'
    }
    neutral_df['language'] = neutral_df['language'].replace(lang_map)
    gen_df['language'] = gen_df['language'].replace(lang_map)
    
    # Ensure column order matches
    columns = original_df.columns.tolist()
    
    # Fix ID generation
    current_max_id = original_df['id'].max()
    
    def prepare_df(df, start_id):
        # Align columns
        for col in columns:
            if col not in df.columns:
                df[col] = -1
        
        # Assign new IDs
        df['id'] = range(start_id + 1, start_id + 1 + len(df))
        
        # Ensure split is set (default to train if missing)
        if 'split' not in df.columns or df['split'].isnull().all():
             df['split'] = 'train'
             
        return df[columns]

    print("🛠️  Aligning columns and generating IDs...")
    neutral_df = prepare_df(neutral_df, current_max_id)
    current_max_id = neutral_df['id'].max()
    
    gen_df = prepare_df(gen_df, current_max_id)
    current_max_id = gen_df['id'].max()
    
    # Concatenate
    print("🔗 Merging all datasets...")
    final_df = pd.concat([original_df, neutral_df, gen_df], ignore_index=True)
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=1337).reset_index(drop=True)
    
    # Save
    output_path = 'dataset/UNIFIED_BALANCED_GENERATED.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Successfully created {output_path}")
    print(f"📊 Final Shape: {final_df.shape}")
    print("\nLanguage Distribution:")
    print(final_df['language'].value_counts())
    print("\nHate Type Distribution:")
    print(final_df['hate_type'].value_counts())

if __name__ == "__main__":
    merge_datasets()
