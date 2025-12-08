import pandas as pd
import numpy as np

def merge_datasets():
    print("🔄 Loading datasets...")
    
    # Load original unified data
    original_df = pd.read_csv('dataset/UNIFIED_ALL_SPLIT.csv')
    print(f"Original data shape: {original_df.shape}")
    
    # Load new neutral data
    neutral_df = pd.read_csv('dataset/neutral_boost_large.csv')
    print(f"New neutral data shape: {neutral_df.shape}")
    
    # Add missing 'is_hate' column to neutral data
    neutral_df['is_hate'] = 0
    
    # Normalize language codes
    lang_map = {
        'bn': 'bangla',
        'en': 'english',
        'bl': 'banglish'
    }
    neutral_df['language'] = neutral_df['language'].replace(lang_map)
    
    # Ensure column order matches
    columns = original_df.columns.tolist()
    # Check if all columns exist in neutral_df, if not add them with default values or handle
    for col in columns:
        if col not in neutral_df.columns:
            print(f"Warning: {col} missing in neutral data. Filling with default.")
            if col == 'id':
                # Should have been generated, but just in case
                start_id = original_df['id'].max() + 1
                neutral_df['id'] = range(start_id, start_id + len(neutral_df))
            else:
                neutral_df[col] = -1 # Default for unknown
                
    neutral_df = neutral_df[columns]
    
    # Concatenate
    print("➕ Merging...")
    balanced_df = pd.concat([original_df, neutral_df], ignore_index=True)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=1337).reset_index(drop=True)
    
    # Save
    output_path = 'dataset/UNIFIED_BALANCED.csv'
    balanced_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved balanced dataset to {output_path}")
    print(f"Total samples: {len(balanced_df)}")
    
    # Stats
    print("\n📊 Class Distribution (is_hate):")
    print(balanced_df['is_hate'].value_counts(normalize=True))
    print(balanced_df['is_hate'].value_counts())
    
    print("\n📊 Language Distribution:")
    print(balanced_df['language'].value_counts())

if __name__ == "__main__":
    merge_datasets()
