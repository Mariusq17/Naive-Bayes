import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ============================================================
# CLASIFICATOR MULTINOMIAL NAIVE BAYES
# Proiect pentru clasificarea articolelor de stiri BBC
# ============================================================

print("=" * 70)
print("CLASIFICATOR MULTINOMIAL NAIVE BAYES")
print("Versiune Optimizata - Categorii Simplificate")
print("=" * 70)

# ============================================================
# FAZA 1: PREPARAREA SI CURATAREA DATELOR
# ============================================================

print("\n" + "=" * 70)
print("FAZA 1: PREPARAREA SI CURATAREA DATELOR")
print("=" * 70)

print("\n[1.1] Incarcarea datelor...")
df = pd.read_csv('bbc_news.csv')
print(f"✅ Incarcate {len(df)} articole")

print("\n[1.2] Extragerea categoriilor din URL-uri...")

def extract_category(url):
    """Extrage categoria dintr-un URL BBC News."""
    try:
        match = re.search(r'/news/([a-z-]+)-\d+', url)
        if match:
            return match.group(1)
        match = re.search(r'/news/([a-z-]+)/', url)
        if match:
            return match.group(1)
        return 'unknown'
    except:
        return 'unknown'

df['category_raw'] = df['link'].apply(extract_category)
print(f"✅ Categorii extrase: {df['category_raw'].nunique()} categorii unice")

print("\n[1.3] Simplificarea si gruparea categoriilor...")

def simplify_category(cat):
    """Grupeaza categoriile similare in mega-categorii."""
    if cat.startswith('world') or cat in ['world-europe', 'world-us-canada', 
                                           'world-asia', 'world-middle-east',
                                           'world-africa', 'world-latin-america',
                                           'world-asia-india']:
        return 'world'
    
    if cat.startswith('uk'):
        return 'uk'
    
    if 'entertainment' in cat or 'arts' in cat or cat == 'newsbeat':
        return 'entertainment'
    
    if cat in ['technology', 'science-environment']:
        return 'tech-science'
    
    if cat in ['health', 'education']:
        return 'health-education'
    
    if cat == 'business':
        return 'business'
    
    return 'other'

df['category'] = df['category_raw'].apply(simplify_category)

MIN_ARTICLES = 200
category_counts = df['category'].value_counts()
valid_categories = category_counts[category_counts >= MIN_ARTICLES].index.tolist()

for remove_cat in ['unknown', 'other']:
    if remove_cat in valid_categories:
        valid_categories.remove(remove_cat)

df_filtered = df[df['category'].isin(valid_categories)].copy()

print(f"✅ Categorii finale: {len(valid_categories)}")
print(f"   Categorii: {', '.join(sorted(valid_categories))}")
print(f"✅ Total articole: {len(df_filtered)}")
print("\n📊 Distributie categorii:")
print(df_filtered['category'].value_counts())

print("\n[1.4] Curatarea si tokenizarea textului...")

def clean_text(text):
    """Curata si tokenizeaza textul."""
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their'
    }
    
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words

df_filtered['text'] = df_filtered['title'].fillna('') + ' ' + df_filtered['description'].fillna('')
df_filtered['tokens'] = df_filtered['text'].apply(clean_text)

print(f"✅ Textele au fost curatate si tokenizate")

print("\n[1.5] Impartirea in train (80%) si test (20%)...")

df_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(0.8 * len(df_shuffled))

train_df = df_shuffled[:split_index]
test_df = df_shuffled[split_index:]

print(f"✅ Train: {len(train_df)} articole")
print(f"✅ Test: {len(test_df)} articole")

print("\n✅ FAZA 1 COMPLETA!")

# ============================================================
# FAZA 2: IMPLEMENTAREA CLASIFICATORULUI
# ============================================================

print("\n" + "=" * 70)
print("FAZA 2: IMPLEMENTAREA CLASIFICATORULUI NAIVE BAYES")
print("=" * 70)

print("\n[2.1] Antrenarea modelului...")

def train(train_data):
    """
    Invata parametrii modelului Multinomial Naive Bayes.
    Returns: vocabulary, prior_probs, cond_probs
    """
    
    # Probabilitati a priori: P(categorie)
    category_counts = train_data['category'].value_counts()
    total_docs = len(train_data)
    prior_probs = {cat: count/total_docs for cat, count in category_counts.items()}
    
    # Vocabular: toate cuvintele unice
    vocabulary = set()
    for tokens in train_data['tokens']:
        vocabulary.update(tokens)
    
    # Probabilitati conditionate cu Laplace Smoothing
    word_counts = defaultdict(lambda: defaultdict(int))
    category_word_totals = defaultdict(int)
    
    for idx, row in train_data.iterrows():
        category = row['category']
        tokens = row['tokens']
        for word in tokens:
            word_counts[category][word] += 1
            category_word_totals[category] += 1
    
    cond_probs = defaultdict(lambda: defaultdict(float))
    alpha = 1
    vocab_size = len(vocabulary)
    
    for category in prior_probs.keys():
        total_words = category_word_totals[category]
        for word in vocabulary:
            word_count = word_counts[category][word]
            cond_probs[category][word] = (word_count + alpha) / (total_words + alpha * vocab_size)
    
    return vocabulary, prior_probs, cond_probs

vocabulary, prior_probs, cond_probs = train(train_df)

print(f"✅ Model antrenat cu succes:")
print(f"   - Vocabular: {len(vocabulary)} cuvinte unice")
print(f"   - Categorii: {len(prior_probs)}")
print(f"   - Probabilitati calculate: {len(prior_probs) * len(vocabulary):,}")

print("\n[2.2] Implementarea functiei de predictie...")

def predict(tokens, vocabulary, prior_probs, cond_probs):
    """
    Clasifica un text nou pe baza token-ilor sai.
    Foloseste logaritmi pentru stabilitate numerica.
    """
    scores = {}
    
    for category in prior_probs.keys():
        score = np.log(prior_probs[category])
        
        for word in tokens:
            if word in vocabulary:
                score += np.log(cond_probs[category][word])
        
        scores[category] = score
    
    return max(scores, key=scores.get)

print(f"✅ Functie de predictie implementata")

print("\n✅ FAZA 2 COMPLETA!")

# ============================================================
# FAZA 3: EVALUAREA MODELULUI
# ============================================================

print("\n" + "=" * 70)
print("FAZA 3: EVALUAREA MODELULUI")
print("=" * 70)

print("\n[3.1] Rularea predictiilor pe setul de test...")

predictions = []
true_labels = []

for idx, row in test_df.iterrows():
    tokens = row['tokens']
    true_category = row['category']
    
    predicted_category = predict(tokens, vocabulary, prior_probs, cond_probs)
    
    predictions.append(predicted_category)
    true_labels.append(true_category)

print(f"✅ Completate {len(predictions)} predictii")

print("\n[3.2] Calcularea acuratetii...")

correct = sum([1 for pred, true in zip(predictions, true_labels) if pred == true])
total = len(predictions)
accuracy = (correct / total) * 100

print("\n" + "=" * 70)
print("REZULTATE FINALE")
print("=" * 70)
print(f"\n📊 ACURATETEA MODELULUI: {accuracy:.2f}%")
print(f"   - Predictii corecte: {correct}/{total}")
print(f"   - Predictii gresite: {total - correct}/{total}")
print(f"\n🎯 Numar de categorii: {len(prior_probs)}")
print(f"   Categorii: {', '.join(sorted(prior_probs.keys()))}")

print("\n📈 Acuratete per categorie:")
print("-" * 50)
category_correct = defaultdict(int)
category_total = defaultdict(int)

for pred, true in zip(predictions, true_labels):
    category_total[true] += 1
    if pred == true:
        category_correct[true] += 1

for cat in sorted(prior_probs.keys()):
    if category_total[cat] > 0:
        cat_acc = (category_correct[cat] / category_total[cat]) * 100
        print(f"  {cat:20} | {cat_acc:6.2f}% ({category_correct[cat]}/{category_total[cat]})")

print("\n" + "=" * 70)
print("EXEMPLE DE PREDICTII")
print("=" * 70)

for i in range(10):
    row = test_df.iloc[i]
    pred = predictions[i]
    true = true_labels[i]
    
    status = "✅ CORECT" if pred == true else "❌ GRESIT"
    
    print(f"\nExemplu {i+1}: {status}")
    print(f"  Titlu: {row['title'][:70]}...")
    print(f"  Categorie reala: {true}")
    print(f"  Categorie prezisa: {pred}")

print("\n" + "=" * 70)
print("✅ PROIECT COMPLET!")
print("=" * 70)
print(f"\nModelul Multinomial Naive Bayes a fost implementat si evaluat cu succes!")
print(f"Acuratete finala: {accuracy:.2f}%")