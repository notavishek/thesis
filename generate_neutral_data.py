import pandas as pd
import random

# ============================================================
# 1. EXPANDED VOCABULARY & TEMPLATES
# ============================================================

# --- BENGALI ---
bn_subjects = [
    'আমি', 'তুমি', 'সে', 'আমরা', 'তারা', 'আমার মা', 'আমার বাবা', 'রহিম', 'করিম', 'মানুষ', 'ছাত্ররা', 'শিক্ষক',
    'আমার বন্ধু', 'ডাক্তার', 'নার্স', 'দোকানদার', 'ড্রাইভার', 'পুলিশ', 'লেখক', 'কবি', 'শিল্পী', 'খেলোয়াড়'
]
bn_objects = [
    'বই', 'গান', 'সিনেমা', 'ফুল', 'পাখি', 'দেশ', 'কাজ', 'খাবার', 'স্কুল', 'অফিস', 'ক্রিকেট', 'ফুটবল',
    'চা', 'কফি', 'ভাত', 'মাছ', 'মাংস', 'সবজি', 'ফল', 'জল', 'নদী', 'আকাশ', 'চাঁদ', 'তারা', 'সূর্য',
    'কম্পিউটার', 'মোবাইল', 'ইন্টারনেট', 'গাড়ি', 'বাস', 'ট্রেন', 'বিমান', 'রাস্তা', 'বাড়ি', 'ঘর'
]
bn_verbs = [
    'পছন্দ করি', 'ভালোবাসি', 'দেখছি', 'করছি', 'খাবো', 'যাবো', 'খেলছে', 'পড়ছে', 'লিখছে', 'শুনছে',
    'কিনবো', 'বিক্রি করবো', 'রান্না করছি', 'ঘুমাচ্ছি', 'হাঁটছি', 'দৌড়াচ্ছি', 'হাসছি', 'কাঁদছি', 'ভাবছি', 'বলছি'
]
bn_adjectives = [
    'ভালো', 'সুন্দর', 'শান্ত', 'খুশি', 'সৎ', 'মেধাবী', 'পরিশ্রমী', 'ভদ্র', 'সুস্থ',
    'বড়', 'ছোট', 'নতুন', 'পুরানো', 'লাল', 'নীল', 'সবুজ', 'হলুদ', 'সাদা', 'কালো', 'গরম', 'ঠান্ডা',
    'সুস্বাদু', 'মজার', 'কঠিন', 'সহজ', 'দামী', 'সস্তা', 'দ্রুত', 'ধীর'
]
bn_sentences = [
    "আজকের আবহাওয়া খুব {adj}",
    "{sub} {obj} {verb}",
    "{sub} খুব {adj}",
    "বাংলাদেশ একটি {adj} দেশ",
    "ধন্যবাদ তোমাকে",
    "শুভ সকাল",
    "শুভ রাত্রি",
    "কেমন আছো?",
    "আমি ভালো আছি",
    "দেখা হবে",
    "এখন সময় কত?",
    "বৃষ্টি হচ্ছে",
    "সূর্য পূর্ব দিকে ওঠে",
    "পানি জীবন",
    "গাছ আমাদের বন্ধু",
    "সত্য কথা বলা ভালো",
    "বড়দের সম্মান করা উচিত",
    "চলো {obj} খেলি",
    "আমি {obj} পছন্দ করি না",
    "সে আজ আসবে না",
    "আমার {obj} খুব প্রিয়",
    "{sub} প্রতিদিন {obj} {verb}",
    "আজকে আমার মন {adj}",
    "তোমার নাম কি?",
    "তুমি কোথায় থাকো?",
    "আমি {obj} খেতে ভালোবাসি",
    "চলো ঘুরতে যাই",
    "আজকে ছুটি",
    "পড়াশোনা করা জরুরি",
    "স্বাস্থ্যের যত্ন নেওয়া উচিত"
]

# --- ENGLISH ---
en_subjects = [
    'I', 'You', 'He', 'She', 'We', 'They', 'My mother', 'My father', 'The teacher', 'The student', 'John',
    'My friend', 'The doctor', 'The nurse', 'The shopkeeper', 'The driver', 'The police', 'The writer', 'The poet', 'The artist', 'The player'
]
en_objects = [
    'book', 'song', 'movie', 'flower', 'bird', 'country', 'work', 'food', 'school', 'office', 'cricket', 'football',
    'tea', 'coffee', 'rice', 'fish', 'meat', 'vegetables', 'fruits', 'water', 'river', 'sky', 'moon', 'stars', 'sun',
    'computer', 'mobile', 'internet', 'car', 'bus', 'train', 'plane', 'road', 'house', 'room'
]
en_verbs = [
    'like', 'love', 'watch', 'do', 'eat', 'go', 'play', 'read', 'write', 'listen to',
    'buy', 'sell', 'cook', 'sleep', 'walk', 'run', 'laugh', 'cry', 'think', 'say'
]
en_adjectives = [
    'good', 'beautiful', 'calm', 'happy', 'honest', 'smart', 'hardworking', 'polite', 'healthy',
    'big', 'small', 'new', 'old', 'red', 'blue', 'green', 'yellow', 'white', 'black', 'hot', 'cold',
    'tasty', 'funny', 'hard', 'easy', 'expensive', 'cheap', 'fast', 'slow'
]
en_sentences = [
    "The weather is very {adj} today",
    "{sub} {verb} the {obj}",
    "{sub} is very {adj}",
    "Bangladesh is a {adj} country",
    "Thank you",
    "Good morning",
    "Good night",
    "How are you?",
    "I am fine",
    "See you later",
    "What time is it?",
    "It is raining",
    "The sun rises in the east",
    "Water is life",
    "Trees are our friends",
    "It is good to tell the truth",
    "We should respect elders",
    "Let's play {obj}",
    "I do not like {obj}",
    "He will not come today",
    "My {obj} is very favorite",
    "{sub} {verb} {obj} every day",
    "Today my mind is {adj}",
    "What is your name?",
    "Where do you live?",
    "I love to eat {obj}",
    "Let's go for a walk",
    "Today is a holiday",
    "Studying is important",
    "We should take care of health"
]

# --- BANGLISH (Transliterated) ---
bl_subjects = [
    'ami', 'tumi', 'she', 'amra', 'tara', 'amar ma', 'amar baba', 'rahim', 'karim', 'manush',
    'amar bondhu', 'doctor', 'nurse', 'dokanadar', 'driver', 'police', 'lekhok', 'kobi', 'shilpi', 'kheloar'
]
bl_objects = [
    'boi', 'gaan', 'cinema', 'ful', 'pakhi', 'desh', 'kaaj', 'khabar', 'school', 'office', 'cricket',
    'cha', 'coffee', 'vat', 'mach', 'mangsho', 'sobji', 'fol', 'jol', 'nodi', 'akash', 'chad', 'tara', 'surjo',
    'computer', 'mobile', 'internet', 'gari', 'bus', 'train', 'biman', 'rasta', 'bari', 'ghor'
]
bl_verbs = [
    'pochondo kori', 'valobashi', 'dekhchi', 'korchi', 'khabo', 'jabo', 'khelche', 'porche', 'likhche', 'shunche',
    'kinbo', 'bikri korbo', 'ranna korchi', 'ghumacchi', 'hatchi', 'douracchi', 'hashchi', 'kadchi', 'vabchi', 'bolchi'
]
bl_adjectives = [
    'valo', 'shundor', 'shanto', 'khushi', 'shot', 'medhabi', 'porishromi', 'vodro',
    'boro', 'choto', 'notun', 'purano', 'lal', 'nil', 'sobuj', 'holud', 'shada', 'kalo', 'gorom', 'thanda',
    'shushadu', 'mojar', 'kothin', 'shohoj', 'dami', 'shosta', 'druto', 'dhir'
]
bl_sentences = [
    "ajker weather khub {adj}",
    "{sub} {obj} {verb}",
    "{sub} khub {adj}",
    "bangladesh ekta {adj} desh",
    "dhonnobad tomake",
    "shuvo shokal",
    "shuvo ratri",
    "kemon acho?",
    "ami valo achi",
    "dekha hobe",
    "ekhon somoy koto?",
    "brishti hocche",
    "pani jibon",
    "gach amader bondhu",
    "shotti kotha bola valo",
    "cholo {obj} kheli",
    "ami {obj} pochondo kori na",
    "she aj ashbe na",
    "amar {obj} khub priyo",
    "{sub} protidin {obj} {verb}",
    "ajke amar mon {adj}",
    "tomar nam ki?",
    "tumi kothay thako?",
    "ami {obj} khete valobashi",
    "cholo ghurte jai",
    "ajke chuti",
    "porashona kora joruri",
    "shasther jotno neya uchit"
]

# ============================================================
# 2. GENERATOR FUNCTION
# ============================================================
def generate_samples(count, lang_code, subjects, objects, verbs, adjectives, templates):
    data = []
    for _ in range(count):
        tmpl = random.choice(templates)
        text = tmpl.format(
            sub=random.choice(subjects),
            obj=random.choice(objects),
            verb=random.choice(verbs),
            adj=random.choice(adjectives)
        )
        # Add some variation
        if random.random() > 0.8:
            text = text + "."
        elif random.random() > 0.8:
            text = text + "!"
            
        data.append({
            'text': text,
            'language': lang_code,
            'hate_type': 0,       # Not Hate
            'target_group': 0,    # None
            'severity': 0,        # None
            'confidence': 1.0,
            'source_dataset': 'synthetic_neutral'
        })
    return data

# ============================================================
# 3. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("🚀 Generating Expanded Neutral Dataset...")
    
    # Generate 12000 Bengali
    bn_data = generate_samples(12000, 'bn', bn_subjects, bn_objects, bn_verbs, bn_adjectives, bn_sentences)
    
    # Generate 12000 English
    en_data = generate_samples(12000, 'en', en_subjects, en_objects, en_verbs, en_adjectives, en_sentences)
    
    # Generate 11000 Banglish
    bl_data = generate_samples(11000, 'bl', bl_subjects, bl_objects, bl_verbs, bl_adjectives, bl_sentences)
    
    all_data = bn_data + en_data + bl_data
    df = pd.DataFrame(all_data)
    
    # Add required columns for compatibility
    df['id'] = range(200000, 200000 + len(df))
    df['split'] = 'train' # All for training
    
    output_path = 'dataset/neutral_boost_large.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} neutral samples.")
    print(f"💾 Saved to {output_path}")
    print(df['language'].value_counts())
