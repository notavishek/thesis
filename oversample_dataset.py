import pandas as pd
from sklearn.utils import resample

def oversample_minorities():
    print("🔄 Loading UNIFIED_BALANCED.csv...")
    df = pd.read_csv('dataset/UNIFIED_BALANCED.csv')
    
    # Target count per class (aiming for ~4000 to bridge the gap with Personal Attack's 10k+)
    TARGET_COUNT = 4000
    
    # We only oversample HATE classes (hate_type > 0)
    # We leave 'Not Hate' (0) alone as it's the majority
    
    augmented_dfs = [df] # Start with original data
    
    print(f"\n🎯 Target per hate class: {TARGET_COUNT} samples")
    
    # Iterate through each language to preserve language balance
    for lang in df['language'].unique():
        print(f"\nProcessing {lang}...")
        lang_df = df[df['language'] == lang]
        
        # Iterate through hate types 1-5
        for ht in [1, 2, 3, 5]: # Political, Religious, Gender, Geopolitical
            subset = lang_df[lang_df['hate_type'] == ht]
            count = len(subset)
            
            if count == 0:
                continue
                
            if count < TARGET_COUNT:
                # Calculate how many to add
                needed = TARGET_COUNT - count
                print(f"   - Hate Type {ht}: Found {count}, Oversampling +{needed}...")
                
                # Resample with replacement
                oversampled = resample(subset, 
                                     replace=True,     # Sample with replacement
                                     n_samples=needed, # Match target
                                     random_state=1337)
                
                augmented_dfs.append(oversampled)
            else:
                print(f"   - Hate Type {ht}: Found {count} (Sufficient)")

    # Combine
    print("\n➕ Merging oversampled data...")
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=1337).reset_index(drop=True)
    
    output_path = 'dataset/UNIFIED_BALANCED_OVERSAMPLED.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved to {output_path}")
    print(f"Original Size: {len(df)}")
    print(f"New Size:      {len(final_df)}")
    
    # Quick Stats
    print("\n📊 New Class Distribution (Hate Types):")
    print(final_df['hate_type'].value_counts().sort_index())

if __name__ == "__main__":
    oversample_minorities()
