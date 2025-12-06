import pandas as pd

print("=== ANALYZING GENDER LABELS IN DATASETS ===\n")

# Check Bengali Hate v1
df1 = pd.read_csv('dataset/bengali_ hate_v1.0.csv')
print("Bengali Hate v1 Labels:")
print(df1['label'].value_counts())

# Check Bengali Hate v2
df2 = pd.read_csv('dataset/bengali_hate_v2.0.csv')
print("\n\nBengali Hate v2 Labels:")
print(df2['label'].value_counts())

# Find gender-related samples
print("\n\n=== GENDER-RELATED SAMPLES ===")
gender_keywords = ['খানকি', 'khanki', 'বেশ্যা', 'beshya', 'মাগি', 'magi', 'রান্ডি', 'randi', 'whore', 'slut']

for keyword in gender_keywords:
    matches_v1 = df1[df1['text'].str.contains(keyword, case=False, na=False)]
    matches_v2 = df2[df2['text'].str.contains(keyword, case=False, na=False)]
    
    if len(matches_v1) > 0:
        print(f"\nKeyword '{keyword}' in v1 ({len(matches_v1)} matches):")
        for idx, row in matches_v1.head(3).iterrows():
            print(f"  Label: {row['label']} | Text: {row['text'][:60]}")
    
    if len(matches_v2) > 0:
        print(f"\nKeyword '{keyword}' in v2 ({len(matches_v2)} matches):")
        for idx, row in matches_v2.head(3).iterrows():
            print(f"  Label: {row['label']} | Text: {row['text'][:60]}")

# Check if any are labeled as "Gender"
print("\n\n=== CHECKING FOR 'Gender' LABELS ===")
gender_v1 = df1[df1['label'].str.contains('Gender|gender', case=False, na=False)]
gender_v2 = df2[df2['label'].str.contains('Gender|gender', case=False, na=False)]

print(f"Bengali v1 with 'Gender' label: {len(gender_v1)}")
print(f"Bengali v2 with 'Gender' label: {len(gender_v2)}")

if len(gender_v2) > 0:
    print("\nSample Gender-labeled texts from v2:")
    for idx, row in gender_v2.head(5).iterrows():
        print(f"  {row['text'][:80]}")
