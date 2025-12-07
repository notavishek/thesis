import pandas as pd
import numpy as np
from indic_transliteration import sanscript
# Define schemes manually if import fails
SCHEME_BENGALI = 'bengali'
SCHEME_ITRANS = 'itrans'
from tqdm import tqdm
import re

def transliterate_bengali_to_banglish(text):
    """
    Transliterates Bengali script to Roman (Banglish) using ITRANS scheme
    and then cleans it up to look more like natural Banglish.
    """
    if not isinstance(text, str):
        return str(text)
        
    # 1. Basic Transliteration (ITRANS scheme is closest to phonetic typing)
    text = sanscript.transliterate(text, SCHEME_BENGALI, SCHEME_ITRANS)
    
    # 2. Post-processing to make it look like natural "Banglish"
    # ITRANS uses 'z' for some sounds, 'RRi' for others. Let's normalize.
    
    # Common replacements to match colloquial typing
    replacements = [
        ('RRi', 'ri'),
        ('sh', 'sh'),
        ('S', 'sh'),
        ('Sh', 'sh'),
        ('ch', 'ch'),
        ('Ch', 'ch'),
        ('Th', 'th'),
        ('T', 't'),
        ('D', 'd'),
        ('Dh', 'dh'),
        ('N', 'n'),
        ('y', 'y'),
        ('Y', 'y'),
        ('w', 'b'), # Often 'w' sound is written as 'b' or 'o' in banglish
        ('v', 'b'),
        ('aa', 'a'), # People rarely type double 'aa'
        ('ii', 'i'),
        ('uu', 'u'),
        ('ee', 'i'),
        ('oo', 'u'),
        ('~', ''),   # Remove nasalization marker often left by ITRANS
        ('.N', 'n'),
        ('.n', 'n'),
        ('M', 'n'),  # Anusvara
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
        
    return text.lower()

def main():
    print("🚀 Starting Banglish Augmentation...")
    
    # 1. Load the Enhanced Dataset
    input_path = 'dataset/UNIFIED_ALL_SPLIT_ENHANCED.csv'
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Loaded {len(df)} rows from {input_path}")
    except FileNotFoundError:
        print(f"❌ Could not find {input_path}")
        return

    # 2. Filter for Bengali rows
    # Note: The dataset uses 'bangla' not 'bengali'
    bengali_df = df[df['language'] == 'bangla'].copy()
    print(f"ℹ️ Found {len(bengali_df)} Bengali samples to transliterate.")
    
    # 3. Create Banglish copies
    banglish_rows = []
    
    print("🔄 Transliterating...")
    for _, row in tqdm(bengali_df.iterrows(), total=len(bengali_df)):
        original_text = row['text']
        banglish_text = transliterate_bengali_to_banglish(original_text)
        
        new_row = row.copy()
        new_row['text'] = banglish_text
        new_row['language'] = 'banglish'
        new_row['source_dataset'] = 'augmented_transliteration'
        
        # Keep the same split (train/val/test) to prevent leakage!
        # If original was in 'train', banglish version should also be in 'train'
        
        banglish_rows.append(new_row)
        
    banglish_df = pd.DataFrame(banglish_rows)
    
    # 4. Combine with original dataset
    # We append the new Banglish rows to the original dataframe
    augmented_df = pd.concat([df, banglish_df], ignore_index=True)
    
    # 5. Shuffle (optional but good practice)
    augmented_df = augmented_df.sample(frac=1, random_state=1337).reset_index(drop=True)
    
    # 6. Save
    output_path = 'dataset/UNIFIED_ALL_SPLIT_AUGMENTED.csv'
    augmented_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(f"✅ Augmentation Complete!")
    print(f"Original Size: {len(df)}")
    print(f"New Banglish Samples: {len(banglish_df)}")
    print(f"Final Size: {len(augmented_df)}")
    print(f"Saved to: {output_path}")
    print("="*50)
    
    # Verify distribution
    print("\nNew Language Distribution:")
    print(augmented_df['language'].value_counts())

if __name__ == "__main__":
    main()
