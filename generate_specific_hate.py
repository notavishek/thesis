import pandas as pd
import random

# ============================================================
# 1. MASSIVE VOCABULARY & TEMPLATES
# ============================================================

# --- ENGLISH ---
en_groups = {
    'political': ['Democrats', 'Republicans', 'Liberals', 'Conservatives', 'Politicians', 'Leftists', 'Right-wingers', 'Socialists', 'Communists', 'The government', 'Senators', 'MPs', 'Congressmen', 'The opposition', 'Party members', 'Activists', 'Leaders', 'Officials', 'Voters', 'Supporters'],
    'religious': ['Muslims', 'Christians', 'Hindus', 'Jews', 'Buddhists', 'Atheists', 'religious people', 'believers', 'Pagans', 'Sikhs', 'Clerics', 'Priests', 'Imams', 'Monks', 'Worshippers', 'Fundamentalists', 'Missionaries', 'Converts', 'Faithful', 'Infidels'],
    'gender': ['Women', 'Men', 'Feminists', 'Trans people', 'Gays', 'Lesbians', 'Queers', 'LGBT people', 'Girls', 'Boys', 'Males', 'Females', 'Wives', 'Husbands', 'Mothers', 'Fathers', 'Daughters', 'Sons', 'Ladies', 'Gentlemen'],
    'geopolitical': ['America', 'China', 'Russia', 'India', 'Pakistan', 'Israel', 'Palestine', 'Iran', 'North Korea', 'Europe', 'The West', 'UK', 'France', 'Germany', 'Saudi Arabia', 'Syria', 'Afghanistan', 'Iraq', 'Immigrants', 'Foreigners']
}

en_templates = [
    "{group} are corrupt liars.", "{group} are destroying the country.", "All {group} are traitors.",
    "We should lock up all {group}.", "{group} are enemies of the state.", "I hate {group} with a passion.",
    "{group} are stealing our money.", "{group} are useless.", "Death to {group} ideology.", "{group} are brainless.",
    "The {group} agenda is evil.", "Stop the {group} now.", "{group} are ruining everything.", "Vote out the {group}.",
    "{group} are puppets.", "{group} have no shame.", "{group} are the real problem.", "Expose the {group}.",
    "{group} are selling us out.", "Never trust {group}.", "{group} are a cancer.", "{group} must be stopped.",
    "Get rid of {group}.", "{group} are filth.", "{group} don't belong here.", "Kick out the {group}.",
    "{group} are parasites.", "{group} are scum.", "{group} deserve to die.", "Wipe out {group}.",
    "{group} are disgusting.", "{group} make me sick.", "{group} are subhuman.", "{group} are animals.",
    "{group} are garbage.", "{group} are a plague.", "{group} are a virus.", "{group} are toxic.",
    "{group} are dangerous.", "{group} are violent.", "{group} are criminals.", "{group} are terrorists.",
    "{group} are rapists.", "{group} are murderers.", "{group} are thieves.", "{group} are liars.",
    "{group} are cheats.", "{group} are hypocrites.", "{group} are cowards.", "{group} are weak."
]

# --- BANGLA ---
bn_groups = {
    'political': ['আওয়ামী লীগ', 'বিএনপি', 'জামাত', 'রাজনীতিবিদরা', 'বামপন্থীরা', 'ডানপন্থীরা', 'সরকার', 'মন্ত্রীরা', 'নেতারা', 'ছাত্রলীগ', 'ছাত্রদল', 'যুবদল', 'যুবলীগ', 'শিবির', 'কর্মীরা', 'সমর্থকরা', 'ভোটাররা', 'এমপিরা', 'চেয়ারম্যান', 'মেম্বার'],
    'religious': ['মুসলমানরা', 'হিন্দুরা', 'খ্রিস্টানরা', 'বৌদ্ধরা', 'নাস্তিকরা', 'কাফেররা', 'মালাউনরা', 'বিধর্মীরা', 'মোল্লারা', 'পুরোহিতরা', 'হুজুররা', 'ঠাকুররা', 'পাদ্রীরা', 'ভক্তরা', 'মুরতাদরা', 'মুনাফিকরা', 'মুশরিকরা', 'আস্তিকরা', 'ধর্মপ্রাণরা', 'উগ্রবাদীরা'],
    'gender': ['মেয়েরা', 'ছেলেরা', 'নারীবাদীরা', 'হিজড়ারা', 'সমকামীরা', 'মহিলারা', 'পুরুষরা', 'বউরা', 'স্বামিরা', 'কন্যারা', 'পুত্ররা', 'মায়েরা', 'বাবারা', 'বোনেরা', 'ভাইয়েরা', 'নারীরা', 'নররা', 'গৃহিণীরা', 'কর্মজীবীরা', 'ছাত্রীরা'],
    'geopolitical': ['ভারত', 'পাকিস্তান', 'আমেরিকা', 'চীন', 'রাশিয়া', 'ইসরায়েল', 'মায়ানমার', 'পশ্চিমারা', 'ইউরোপ', 'সৌদি আরব', 'বিদেশীরা', 'ভারতীয়রা', 'পাকিস্তানিরা', 'আমেরিকানরা', 'চীনারা', 'রাশিয়ানরা', 'রোহিঙ্গারা', 'বর্ডার গার্ড', 'বিএসএফ', 'সেনাবাহিনী']
}

bn_templates = [
    "{group} সব চোর।", "{group} দেশটা ধ্বংস করে দিল।", "{group} সব দালাল।", "{group} কে ধিক্কার জানাই।",
    "{group} দেশের শত্রু।", "{group} সব মিথ্যাবাদী।", "{group} কে বিশ্বাস করবেন না।", "{group} সব দুর্নীতিবাজ।",
    "{group} নিপাত যাক।", "{group} সব শয়তান।", "{group} আমাদের টাকা মেরে খাচ্ছে।", "{group} সব গুন্ডা।",
    "{group} কে জেলে ভরো।", "{group} সব খুনি।", "{group} এর বিচার চাই।", "{group} সব ভণ্ড।",
    "{group} দেশ বিক্রি করে দিচ্ছে।", "{group} সব অমানুষ।", "{group} কে বয়কট করুন।", "{group} সব রাজাকার।",
    "{group} সব জানোয়ার।", "{group} সব পশু।", "{group} সব কুত্তা।", "{group} সব শুয়োর।",
    "{group} সব নাপাক।", "{group} সব খারাপ।", "{group} সব নষ্ট।", "{group} সব পচা।",
    "{group} সব আবর্জনা।", "{group} সব নোংরা।", "{group} সব ছোটলোক।", "{group} সব ইতর।",
    "{group} সব বেয়াদব।", "{group} সব অসভ্য।", "{group} সব বর্বর।", "{group} সব মূর্খ।",
    "{group} সব পাগল।", "{group} সব ছাগল।", "{group} সব গাধা।", "{group} সব বানর।",
    "{group} সব রাক্ষস।", "{group} সব খবিশ।", "{group} সব হারামি।", "{group} সব বেইমান।",
    "{group} সব নিমকহারাম।", "{group} সব দেশদ্রোহী।", "{group} সব জঙ্গি।", "{group} সব সন্ত্রাসী।",
    "{group} সব লুটেরা।", "{group} সব ডাকাত।"
]

# --- BANGLISH ---
bl_groups = {
    'political': ['awami league', 'bnp', 'jamaat', 'rajnitibidra', 'bampanthira', 'danpanthira', 'sorkar', 'montrira', 'netara', 'chatroleague', 'chatrodol', 'jubodol', 'juboleague', 'shibir', 'kormira', 'somorthokra', 'voterra', 'mpra', 'chairman', 'member'],
    'religious': ['musolmanra', 'hindura', 'kristanra', 'bouddhora', 'nastikra', 'kaferra', 'malaunra', 'bidhormira', 'mollara', 'purohitra', 'hujurra', 'thakur', 'padrira', 'voktora', 'murtadra', 'munafikra', 'mushrikra', 'astikra', 'dhormopran', 'ugrobadira'],
    'gender': ['meyera', 'chelera', 'naribadira', 'hijrara', 'somokamira', 'mohilara', 'purushra', 'boura', 'shamira', 'konnara', 'putrora', 'mayera', 'babara', 'bonera', 'bhaiera', 'narira', 'norra', 'grihinira', 'kormojibira', 'chatrira'],
    'geopolitical': ['india', 'pakistan', 'america', 'china', 'russia', 'israel', 'myanmar', 'poschimara', 'europe', 'saudi arab', 'bideshira', 'bharotiyora', 'pakistanira', 'americanra', 'chinara', 'russianra', 'rohingara', 'border guard', 'bsf', 'senabahini']
}

bl_templates = [
    "{group} shob chor.", "{group} deshta dhongsho kore dilo.", "{group} shob dalal.", "{group} ke dhikkar janai.",
    "{group} desher shotru.", "{group} shob mitthabadi.", "{group} ke bishshash korben na.", "{group} shob durnitibaj.",
    "{group} nipat jak.", "{group} shob shoytan.", "{group} amader taka mere khacche.", "{group} shob gunda.",
    "{group} ke jele bhoro.", "{group} shob khuni.", "{group} er bichar chai.", "{group} shob vondo.",
    "{group} desh bikri kore dicche.", "{group} shob omanush.", "{group} ke boycott korun.", "{group} shob rajakar.",
    "{group} shob janwar.", "{group} shob poshu.", "{group} shob kutta.", "{group} shob shuor.",
    "{group} shob napak.", "{group} shob kharap.", "{group} shob noshto.", "{group} shob pocha.",
    "{group} shob aborjona.", "{group} shob nongra.", "{group} shob chotolok.", "{group} shob itor.",
    "{group} shob beyadob.", "{group} shob oshobvo.", "{group} shob borbor.", "{group} shob murkho.",
    "{group} shob pagol.", "{group} shob chagol.", "{group} shob gadha.", "{group} shob banor.",
    "{group} shob rakkhosh.", "{group} shob khobish.", "{group} shob harami.", "{group} shob beiman.",
    "{group} shob nimokharam.", "{group} shob deshdrohi.", "{group} shob jongi.", "{group} shob sontrashi.",
    "{group} shob lutera.", "{group} shob dakat."
]

# ============================================================
# 2. GENERATOR FUNCTION
# ============================================================

def generate_samples(count, templates, groups, lang, hate_type, target_group):
    data = []
    seen = set()
    attempts = 0
    
    while len(data) < count and attempts < count * 10:
        attempts += 1
        tmpl = random.choice(templates)
        group = random.choice(groups)
        base_text = tmpl.format(group=group)
        
        # Massive Variation Strategy
        r = random.random()
        text = base_text
        
        # 1. Case transformations (20%)
        if r < 0.2:
            if random.random() > 0.5:
                text = text.upper()
            else:
                text = text.lower()
        
        # 2. Punctuation (20%)
        elif r < 0.4:
            if text.endswith('.'):
                text = text[:-1] + '!' * random.randint(1, 3)
            elif text.endswith('!'):
                text = text + '!!'
        
        # 3. Prefixes (30%)
        elif r < 0.7:
            prefixes = [
                "I think ", "Honestly, ", "Believe me, ", "Everyone knows ", 
                "It is true that ", "Basically, ", "Literally, ", "You know, ",
                "I say, ", "People say, ", "They say, ", "We know, ", "Look, ",
                "Listen, ", "Fact: ", "Truth is, ", "Real talk: ", "Opinion: "
            ]
            prefix = random.choice(prefixes)
            # Adjust case of base text if needed
            if base_text[0].isupper():
                text = prefix + base_text[0].lower() + base_text[1:]
            else:
                text = prefix + base_text
                
        # 4. Suffixes (30%)
        else:
            suffixes = [
                " !!!", " ...", " !!" , " ??", " #truth", " #facts", 
                " #real", " #wakeup", " 100%", " for real.", " seriously.",
                " no doubt.", " absolutely.", " definitely."
            ]
            text = text + random.choice(suffixes)
            
        if text not in seen:
            seen.add(text)
            data.append({
                'text': text,
                'language': lang,
                'hate_type': hate_type,
                'target_group': target_group,
                'severity': random.choice([1, 2, 3]), # Random severity
                'confidence': 1.0,
                'source_dataset': 'synthetic_specific_gen_v2'
            })
            
    return data

# ============================================================
# 3. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("🚀 Generating MASSIVE Specific Hate Samples (Target: 8500 per class)...")
    
    all_data = []
    TARGET_COUNT = 8500
    
    # --- ENGLISH ---
    print("Generating English...")
    all_data.extend(generate_samples(TARGET_COUNT, en_templates, en_groups['political'], 'english', 1, 2))
    all_data.extend(generate_samples(TARGET_COUNT, en_templates, en_groups['religious'], 'english', 2, 3))
    all_data.extend(generate_samples(TARGET_COUNT, en_templates, en_groups['gender'], 'english', 3, 3))
    all_data.extend(generate_samples(TARGET_COUNT, en_templates, en_groups['geopolitical'], 'english', 5, 3))
    
    # --- BANGLA ---
    print("Generating Bangla...")
    all_data.extend(generate_samples(TARGET_COUNT, bn_templates, bn_groups['political'], 'bangla', 1, 2))
    all_data.extend(generate_samples(TARGET_COUNT, bn_templates, bn_groups['religious'], 'bangla', 2, 3))
    all_data.extend(generate_samples(TARGET_COUNT, bn_templates, bn_groups['gender'], 'bangla', 3, 3))
    all_data.extend(generate_samples(TARGET_COUNT, bn_templates, bn_groups['geopolitical'], 'bangla', 5, 3))
    
    # --- BANGLISH ---
    print("Generating Banglish...")
    all_data.extend(generate_samples(TARGET_COUNT, bl_templates, bl_groups['political'], 'banglish', 1, 2))
    all_data.extend(generate_samples(TARGET_COUNT, bl_templates, bl_groups['religious'], 'banglish', 2, 3))
    all_data.extend(generate_samples(TARGET_COUNT, bl_templates, bl_groups['gender'], 'banglish', 3, 3))
    all_data.extend(generate_samples(TARGET_COUNT, bl_templates, bl_groups['geopolitical'], 'banglish', 5, 3))
    
    df = pd.DataFrame(all_data)
    
    # Add ID
    df['id'] = range(500000, 500000 + len(df))
    df['split'] = 'train'
    df['is_hate'] = 1
    
    output_path = 'dataset/specific_hate_gen.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} unique samples.")
    print(f"💾 Saved to {output_path}")
    print(df['language'].value_counts())
    print(df['hate_type'].value_counts())
