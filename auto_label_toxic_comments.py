# Auto-Label Toxic Comments Dataset
# ===================================================================
# This script adds hate_type and target_group labels to toxic_comments
# dataset using rule-based heuristics and keyword matching
# ===================================================================

import pandas as pd
import re

# ===================================================================
# KEYWORD DICTIONARIES FOR HATE TYPE CLASSIFICATION
# ===================================================================

POLITICAL_KEYWORDS = [
    # English
    'politician', 'government', 'congress', 'democrat', 'republican', 'liberal', 'conservative',
    'trump', 'biden', 'minister', 'party', 'election', 'vote', 'political', 'policy',
    # Bengali/Banglish
    'সরকার', 'sorkar', 'government', 'রাজনীতি', 'rajniti', 'minister', 'মন্ত্রী',
    'আওয়ামী', 'awami', 'বিএনপি', 'bnp', 'জামাত', 'jamat', 'দল', 'dal'
]

RELIGIOUS_KEYWORDS = [
    # English
    'muslim', 'islam', 'christian', 'hindu', 'jew', 'buddhist', 'atheist',
    'religion', 'religious', 'allah', 'god', 'jesus', 'church', 'mosque', 'temple',
    'quran', 'bible', 'kafir', 'infidel', 'heathen',
    # Bengali/Banglish
    'মুসলিম', 'muslim', 'হিন্দু', 'hindu', 'খ্রিস্টান', 'christian', 
    'ধর্ম', 'dhormo', 'আল্লাহ', 'allah', 'ঈশ্বর', 'ishwar',
    'কাফের', 'kafer', 'kafir', 'ধর্মান্ধ', 'dhormandho', 'মসজিদ', 'mosjid',
    'মন্দির', 'mondir', 'গির্জা', 'girja'
]

GENDER_KEYWORDS = [
    # English - gender slurs and sexist terms
    'bitch', 'slut', 'whore', 'cunt', 'pussy', 'dick', 'cock',
    'rape', 'rapist', 'women belong', 'kitchen', 'sexist', 'misogyn',
    'feminist', 'feminazi', 'man up', 'grow balls',
    # Bengali/Banglish
    'খানকি', 'khanki', 'বেশ্যা', 'beshya', 'মাগি', 'magi', 'রান্ডি', 'randi',
    'মেয়ে', 'meye', 'ছেলে', 'chele', 'নারী', 'nari', 'পুরুষ', 'purush',
    'ধর্ষণ', 'rape', 'ধর্ষক', 'dhorsok'
]

GEOPOLITICAL_KEYWORDS = [
    # Countries and nationalities
    'immigrant', 'foreigner', 'refugee', 'migrant', 'border', 'illegal',
    'american', 'chinese', 'indian', 'pakistani', 'russian', 'arab', 'mexican',
    'go back to', 'deport', 'invasion', 'country', 'nation',
    # Bengali/Banglish
    'ভারত', 'india', 'পাকিস্তান', 'pakistan', 'বাংলাদেশ', 'bangladesh',
    'আমেরিকা', 'america', 'চীন', 'china', 'দেশ', 'desh',
    'বিদেশী', 'bideshi', 'foreigner', 'শরণার্থী', 'refugee'
]

PERSONAL_ATTACK_KEYWORDS = [
    # General insults
    'idiot', 'stupid', 'moron', 'fool', 'dumb', 'retard', 'loser', 'pathetic',
    'worthless', 'trash', 'garbage', 'scum', 'shit', 'fuck', 'ass', 'bastard',
    'kill yourself', 'die', 'death', 'suicide', 'murder', 'kill',
    # Bengali/Banglish
    'বোকা', 'boka', 'পাগল', 'pagol', 'গাধা', 'gadha', 'মূর্খ', 'murkho',
    'হারামি', 'harami', 'শালা', 'shala', 'sala', 'কুত্তা', 'kutta',
    'মরে যা', 'more ja', 'moira ja', 'খুন', 'khun', 'জানোয়ার', 'janoyar',
    'শুয়োর', 'shuor', 'pig', 'dog', 'animal'
]

# ===================================================================
# TARGET GROUP KEYWORDS
# ===================================================================

INDIVIDUAL_INDICATORS = [
    'you', 'your', 'you are', 'you\'re', "you've", 'yourself',
    'তুই', 'tui', 'তুমি', 'tumi', 'তোর', 'tor', 'তোমার', 'tomar',
    'he', 'she', 'him', 'her', 'his', 'person', 'guy', 'man', 'woman'
]

ORGANIZATION_INDICATORS = [
    'company', 'corporation', 'organization', 'government', 'administration',
    'party', 'group', 'team', 'department', 'agency', 'congress', 'parliament',
    'facebook', 'google', 'twitter', 'media', 'press', 'news',
    'কোম্পানি', 'company', 'সংস্থা', 'songtha', 'দল', 'dal'
]

COMMUNITY_INDICATORS = [
    'all', 'every', 'they', 'them', 'their', 'these people', 'those people',
    'muslims', 'christians', 'hindus', 'jews', 'women', 'men', 'immigrants',
    'refugees', 'liberals', 'conservatives', 'americans', 'chinese',
    'সব', 'sob', 'সকল', 'sokol', 'তারা', 'tara', 'এরা', 'era',
    'মুসলমানরা', 'muslimra', 'হিন্দুরা', 'hindura'
]

# ===================================================================
# CLASSIFICATION FUNCTIONS
# ===================================================================

def classify_hate_type(text):
    """
    Classify hate type based on keywords.
    Priority: Political > Religious > Gender > Geopolitical > Personal Attack
    Returns: 0-5 (0=not_hate, 1=political, 2=religious, 3=gender, 4=personal, 5=geopolitical)
    """
    if pd.isna(text):
        return 0
    
    text_lower = str(text).lower()
    
    # Count matches for each category
    political_score = sum(1 for kw in POLITICAL_KEYWORDS if kw.lower() in text_lower)
    religious_score = sum(1 for kw in RELIGIOUS_KEYWORDS if kw.lower() in text_lower)
    gender_score = sum(1 for kw in GENDER_KEYWORDS if kw.lower() in text_lower)
    geopolitical_score = sum(1 for kw in GEOPOLITICAL_KEYWORDS if kw.lower() in text_lower)
    personal_score = sum(1 for kw in PERSONAL_ATTACK_KEYWORDS if kw.lower() in text_lower)
    
    # Find max score
    scores = {
        1: political_score,
        2: religious_score,
        3: gender_score,
        5: geopolitical_score,
        4: personal_score
    }
    
    max_score = max(scores.values())
    
    # If no matches, return not_hate (0)
    if max_score == 0:
        return 0
    
    # Return category with highest score (with priority order if tied)
    for category in [1, 2, 3, 5, 4]:  # Priority order
        if scores[category] == max_score:
            return category
    
    return 4  # Default to personal attack if hate but no clear category

def classify_target_group(text, hate_type):
    """
    Classify target group based on pronouns and context.
    Returns: 0-3 (0=other/none, 1=individual, 2=organization, 3=community)
    """
    if pd.isna(text) or hate_type == 0:
        return 0  # No target if not hate
    
    text_lower = str(text).lower()
    
    # Count indicators
    individual_count = sum(1 for ind in INDIVIDUAL_INDICATORS if ind.lower() in text_lower)
    org_count = sum(1 for ind in ORGANIZATION_INDICATORS if ind.lower() in text_lower)
    community_count = sum(1 for ind in COMMUNITY_INDICATORS if ind.lower() in text_lower)
    
    # Religious/geopolitical hate usually targets communities
    if hate_type in [2, 5] and community_count > 0:
        return 3
    
    # Determine by highest count
    if individual_count > org_count and individual_count > community_count:
        return 1
    elif org_count > individual_count and org_count > community_count:
        return 2
    elif community_count > 0:
        return 3
    else:
        return 1  # Default to individual for personal attacks

# ===================================================================
# MAIN PROCESSING
# ===================================================================

def auto_label_toxic_comments():
    print("=" * 80)
    print("AUTO-LABELING TOXIC COMMENTS DATASET")
    print("=" * 80)
    
    # Load toxic comments
    print("\n📂 Loading toxic_comments_dataset.csv...")
    df_toxic = pd.read_csv('dataset/toxic_comments_dataset.csv')
    print(f"   Loaded {len(df_toxic)} samples")
    
    # Check current labels
    print("\n📊 Current label status:")
    print(f"   hate_type: All values are -1 (unlabeled)")
    print(f"   target_group: All values are -1 (unlabeled)")
    print(f"   severity: {len(df_toxic[df_toxic['label'].str.contains('toxic', case=False, na=False)])} labeled as toxic")
    
    # Apply auto-labeling
    print("\n🤖 Applying rule-based auto-labeling...")
    
    # Get text column
    text_col = 'comment_text' if 'comment_text' in df_toxic.columns else df_toxic.columns[0]
    
    # Classify hate type
    df_toxic['predicted_hate_type'] = df_toxic[text_col].apply(classify_hate_type)
    
    # Classify target group
    df_toxic['predicted_target_group'] = df_toxic.apply(
        lambda row: classify_target_group(row[text_col], row['predicted_hate_type']), 
        axis=1
    )
    
    # Show results
    print("\n✅ Auto-labeling complete!")
    print("\n📊 Predicted Hate Type Distribution:")
    print(df_toxic['predicted_hate_type'].value_counts().sort_index())
    print(f"\n   0 (not_hate):        {len(df_toxic[df_toxic['predicted_hate_type']==0])}")
    print(f"   1 (political):       {len(df_toxic[df_toxic['predicted_hate_type']==1])}")
    print(f"   2 (religious):       {len(df_toxic[df_toxic['predicted_hate_type']==2])}")
    print(f"   3 (gender):          {len(df_toxic[df_toxic['predicted_hate_type']==3])}")
    print(f"   4 (personal_attack): {len(df_toxic[df_toxic['predicted_hate_type']==4])}")
    print(f"   5 (geopolitical):    {len(df_toxic[df_toxic['predicted_hate_type']==5])}")
    
    print("\n📊 Predicted Target Group Distribution:")
    print(df_toxic['predicted_target_group'].value_counts().sort_index())
    
    # Save labeled version
    output_path = 'dataset/toxic_comments_labeled.csv'
    df_toxic.to_csv(output_path, index=False)
    print(f"\n💾 Saved auto-labeled dataset to: {output_path}")
    
    # Show sample predictions
    print("\n📋 Sample Auto-Labeled Examples:")
    print("=" * 80)
    
    for hate_type in [1, 2, 3, 4, 5]:
        samples = df_toxic[df_toxic['predicted_hate_type'] == hate_type].head(2)
        if len(samples) > 0:
            label_names = {1: 'Political', 2: 'Religious', 3: 'Gender', 4: 'Personal', 5: 'Geopolitical'}
            print(f"\n{label_names[hate_type]} Examples:")
            for idx, row in samples.iterrows():
                print(f"   Text: {row[text_col][:70]}")
                print(f"   → hate_type={hate_type}, target_group={row['predicted_target_group']}")
    
    # Statistics on improvement
    print("\n" + "=" * 80)
    print("📈 IMPACT ON DATASET")
    print("=" * 80)
    
    total_toxic = len(df_toxic[df_toxic['label'].str.contains('toxic', case=False, na=False)])
    labeled_hate = len(df_toxic[df_toxic['predicted_hate_type'] > 0])
    
    print(f"\nToxic samples: {total_toxic}")
    print(f"Auto-labeled as hate: {labeled_hate}")
    print(f"Coverage: {labeled_hate/len(df_toxic)*100:.1f}% now have hate_type labels")
    
    target_labeled = len(df_toxic[df_toxic['predicted_target_group'] > 0])
    print(f"Target group labeled: {target_labeled} ({target_labeled/len(df_toxic)*100:.1f}%)")
    
    return df_toxic

# ===================================================================
# INTEGRATION WITH MAIN.PY
# ===================================================================

def create_enhanced_unified_dataset():
    """
    Create UNIFIED_ALL_ENHANCED.csv with auto-labeled toxic_comments
    """
    print("\n" + "=" * 80)
    print("CREATING ENHANCED UNIFIED DATASET")
    print("=" * 80)
    
    # Load all individual unified datasets
    print("\n📂 Loading individual datasets...")
    bhv1 = pd.read_csv('dataset/unified_bhv1.csv')
    bhv2 = pd.read_csv('dataset/unified_bhv2.csv')
    blp25 = pd.read_csv('dataset/unified_blp25.csv')
    ethos = pd.read_csv('dataset/unified_ethos.csv')
    olid = pd.read_csv('dataset/unified_olid.csv')
    
    print(f"   Bengali v1: {len(bhv1)}")
    print(f"   Bengali v2: {len(bhv2)}")
    print(f"   BLP25: {len(blp25)}")
    print(f"   ETHOS: {len(ethos)}")
    print(f"   OLID: {len(olid)}")
    
    # Load auto-labeled toxic comments
    print("\n📂 Loading auto-labeled toxic_comments...")
    toxic_labeled = pd.read_csv('dataset/toxic_comments_labeled.csv')
    
    # Convert to unified format
    lang_map = {'Bangla': 'bangla', 'English': 'english', 'Mixed': 'banglish'}
    
    toxic_unified = pd.DataFrame()
    toxic_unified['id'] = toxic_labeled['comment_id'] if 'comment_id' in toxic_labeled.columns else range(len(toxic_labeled))
    toxic_unified['text'] = toxic_labeled['comment_text'] if 'comment_text' in toxic_labeled.columns else toxic_labeled.iloc[:, 0]
    toxic_unified['language'] = toxic_labeled['language'].map(lang_map).fillna('banglish') if 'language' in toxic_labeled.columns else 'banglish'
    toxic_unified['hate_type'] = toxic_labeled['predicted_hate_type']
    toxic_unified['target_group'] = toxic_labeled['predicted_target_group']
    toxic_unified['severity'] = toxic_labeled['severity'] if 'severity' in toxic_labeled.columns else 0
    toxic_unified['confidence'] = 0.7  # Lower confidence for auto-labeled data
    toxic_unified['source_dataset'] = 'toxic_comments_labeled'
    
    print(f"   Toxic (auto-labeled): {len(toxic_unified)}")
    
    # Combine all datasets
    print("\n🔗 Combining datasets...")
    all_df = pd.concat([bhv1, bhv2, blp25, ethos, olid, toxic_unified], ignore_index=True)
    
    # Save enhanced unified dataset
    output_path = 'dataset/UNIFIED_ALL_ENHANCED.csv'
    all_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Saved enhanced dataset to: {output_path}")
    print(f"   Total samples: {len(all_df)}")
    
    # Show label coverage
    print("\n📊 Label Coverage in Enhanced Dataset:")
    ht_valid = len(all_df[all_df['hate_type'] != -1])
    tg_valid = len(all_df[all_df['target_group'] != -1])
    sv_valid = len(all_df[all_df['severity'] != -1])
    
    print(f"   hate_type:    {ht_valid}/{len(all_df)} ({ht_valid/len(all_df)*100:.1f}%)")
    print(f"   target_group: {tg_valid}/{len(all_df)} ({tg_valid/len(all_df)*100:.1f}%)")
    print(f"   severity:     {sv_valid}/{len(all_df)} ({sv_valid/len(all_df)*100:.1f}%)")
    
    print("\n📊 Hate Type Distribution:")
    print(all_df[all_df['hate_type'] != -1]['hate_type'].value_counts().sort_index())
    
    return all_df

# ===================================================================
# MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    # Step 1: Auto-label toxic comments
    df_labeled = auto_label_toxic_comments()
    
    # Step 2: Create enhanced unified dataset
    df_enhanced = create_enhanced_unified_dataset()
    
    print("\n" + "=" * 80)
    print("✅ AUTO-LABELING COMPLETE!")
    print("=" * 80)
    print("\n📝 Next Steps:")
    print("   1. Review sample predictions to verify quality")
    print("   2. Run: python split_unified_data_enhanced.py")
    print("   3. This will create UNIFIED_ALL_SPLIT_ENHANCED.csv")
    print("   4. Update main.ipynb to use enhanced dataset")
    print("   5. Retrain model with 100% label coverage!")
    print("=" * 80)
