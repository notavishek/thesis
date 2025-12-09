import pandas as pd
import random

# ============================================================
# 1. TEMPLATES & VOCABULARY
# ============================================================

# --- RELIGIOUS HATE (Target: 1200) ---
rel_groups = ['Muslims', 'Christians', 'Hindus', 'Jews', 'Buddhists', 'Atheists', 'religious people', 'believers']
rel_templates = [
    "All {group} are terrorists.",
    "{group} are destroying our culture.",
    "I hate {group} so much.",
    "{group} should be banned from this country.",
    "{group} are brainwashed sheep.",
    "We need to get rid of all {group}.",
    "{group} are the cause of all problems.",
    "Don't trust {group}.",
    "{group} are evil.",
    "The world would be better without {group}."
]

# --- GENDER HATE (Target: 1200) ---
gender_groups = ['Women', 'Men', 'Feminists', 'Trans people', 'Gays', 'Lesbians']
gender_templates = [
    "{group} are inferior.",
    "{group} should stay in the kitchen.",
    "{group} are trash.",
    "I hate {group}.",
    "{group} are too emotional to lead.",
    "{group} are ruining society.",
    "All {group} are the same.",
    "{group} don't deserve rights.",
    "{group} are weak.",
    "Stop listening to {group}."
]

# --- POLITICAL HATE (Target: 1200) ---
pol_groups = ['Democrats', 'Republicans', 'Liberals', 'Conservatives', 'Politicians', 'Leftists', 'Right-wingers', 'Socialists', 'Communists']
pol_templates = [
    "{group} are corrupt liars.",
    "{group} are destroying the country.",
    "All {group} are traitors.",
    "We should lock up all {group}.",
    "{group} are enemies of the state.",
    "I hate {group} with a passion.",
    "{group} are stealing our money.",
    "{group} are useless.",
    "Death to {group} ideology.",
    "{group} are brainless."
]

# --- GEOPOLITICAL HATE (Target: 1200) ---
geo_countries = ['America', 'China', 'Russia', 'India', 'Pakistan', 'Israel', 'Palestine', 'Iran', 'North Korea', 'Europe', 'The West']
geo_templates = [
    "{country} is a terrorist state.",
    "{country} is the enemy of the world.",
    "We should nuke {country}.",
    "I hate {country}.",
    "{country} people are dirty.",
    "{country} is full of scammers.",
    "Boycott {country}.",
    "{country} is evil.",
    "{country} should be destroyed.",
    "All people from {country} are bad."
]

# ============================================================
# 2. GENERATOR
# ============================================================

def generate_data(count, templates, groups, hate_type_code, target_group_code):
    data = []
    for _ in range(count):
        tmpl = random.choice(templates)
        group = random.choice(groups)
        text = tmpl.format(group=group, country=group) # Handle both keys
        
        # Add variation
        if random.random() > 0.5:
            text = text.upper()
        if random.random() > 0.8:
            text = text + "!!!"
            
        data.append({
            'text': text,
            'language': 'english',
            'hate_type': hate_type_code,
            'target_group': target_group_code,
            'severity': random.choice([1, 2, 3]), # Random severity Low-High
            'confidence': 1.0,
            'source_dataset': 'synthetic_gap_filler'
        })
    return data

# ============================================================
# 3. EXECUTION
# ============================================================

if __name__ == "__main__":
    print("🚀 Generating Gap-Filling Data...")
    
    # Codes:
    # Hate Type: 1=Political, 2=Religious, 3=Gender, 5=Geopolitical
    # Target Group: 2=Group, 3=Community (Mostly Community for these)
    
    religious_data = generate_data(1200, rel_templates, rel_groups, 2, 3)
    gender_data = generate_data(1200, gender_templates, gender_groups, 3, 3)
    political_data = generate_data(1200, pol_templates, pol_groups, 1, 2) # Political parties often Group/Community
    geopolitical_data = generate_data(1200, geo_templates, geo_countries, 5, 3) # Countries are communities
    
    all_data = religious_data + gender_data + political_data + geopolitical_data
    df = pd.DataFrame(all_data)
    
    # Add ID
    df['id'] = range(300000, 300000 + len(df))
    df['split'] = 'train'
    
    output_path = 'dataset/english_gap_fillers.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} samples.")
    print(f"💾 Saved to {output_path}")
    print(df['hate_type'].value_counts())
