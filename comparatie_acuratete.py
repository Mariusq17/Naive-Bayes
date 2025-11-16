import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ============================================================
# CLASIFICATOR NAIVE BAYES - COMPARAȚIE ACURATEȚE
# Demonstrează cum numărul de categorii afectează performanța
# ============================================================

print("=" * 70)
print("CLASIFICATOR NAIVE BAYES - ANALIZA COMPARATIVE")
print("=" * 70)

# ============================================================
# ÎNCĂRCARE ȘI PROCESARE INIȚIALĂ
# ============================================================

print("\n[PREGĂTIRE] Încărcarea și procesarea datelor...")

df = pd.read_csv('bbc_news.csv')

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

df['category_original'] = df['link'].apply(extract_category)

def clean_text(text):
    """Curăță și tokenizează textul."""
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

df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
df['tokens'] = df['text'].apply(clean_text)

print(f"✅ Încărcate {len(df)} articole")

# ============================================================
# FUNCȚII PENTRU CATEGORII SIMPLIFICATE
# ============================================================

def simplify_category_conservative(cat):
    """
    Simplificare conservativă: Grupează doar categoriile world-*
    Păstrează: business, uk, entertainment-arts, technology, health, etc.
    Grupează: world-europe, world-asia, etc. → world
    """
    if cat.startswith('world-'):
        return 'world'
    elif cat.startswith('uk-'):
        return 'uk'
    return cat

def simplify_category_aggressive(cat):
    """
    Simplificare agresivă: Grupează în 7 mega-categorii
    """
    # Categorii world
    if cat.startswith('world') or cat in ['world-europe', 'world-us-canada', 
                                           'world-asia', 'world-middle-east',
                                           'world-africa', 'world-latin-america',
                                           'world-asia-india']:
        return 'world'
    
    # Categorii UK
    if cat.startswith('uk'):
        return 'uk'
    
    # Entertainment
    if 'entertainment' in cat or 'arts' in cat or cat == 'newsbeat':
        return 'entertainment'
    
    # Tehnologie & Știință
    if cat in ['technology', 'science-environment']:
        return 'tech-science'
    
    # Sănătate & Educație
    if cat in ['health', 'education']:
        return 'health-education'
    
    # Business rămâne business
    if cat == 'business':
        return 'business'
    
    # Restul în "other"
    return 'other'

# ============================================================
# FUNCȚII MODEL NAIVE BAYES
# ============================================================

def train(train_data):
    """Antrenează modelul Naive Bayes."""
    category_counts = train_data['category'].value_counts()
    total_docs = len(train_data)
    prior_probs = {cat: count/total_docs for cat, count in category_counts.items()}
    
    vocabulary = set()
    for tokens in train_data['tokens']:
        vocabulary.update(tokens)
    
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

def predict(tokens, vocabulary, prior_probs, cond_probs):
    """Clasifică un text."""
    scores = {}
    for category in prior_probs.keys():
        score = np.log(prior_probs[category])
        for word in tokens:
            if word in vocabulary:
                score += np.log(cond_probs[category][word])
        scores[category] = score
    return max(scores, key=scores.get)

def evaluate_model(train_df, test_df):
    """Evaluează modelul și returnează acuratețea."""
    vocabulary, prior_probs, cond_probs = train(train_df)
    
    predictions = []
    true_labels = []
    
    for idx, row in test_df.iterrows():
        predicted = predict(row['tokens'], vocabulary, prior_probs, cond_probs)
        predictions.append(predicted)
        true_labels.append(row['category'])
    
    correct = sum([1 for p, t in zip(predictions, true_labels) if p == t])
    accuracy = (correct / len(predictions)) * 100
    
    return accuracy, len(prior_probs), predictions, true_labels

# ============================================================
# EXPERIMENTE: 3 CONFIGURAȚII DIFERITE
# ============================================================

results = []

# EXPERIMENT 1: 25 CATEGORII (Original - Multe categorii)
print("\n" + "=" * 70)
print("EXPERIMENT 1: 25 CATEGORII (ORIGINAL)")
print("=" * 70)

MIN_ARTICLES = 200
category_counts = df['category_original'].value_counts()
valid_cats = category_counts[category_counts >= MIN_ARTICLES].index.tolist()
if 'unknown' in valid_cats:
    valid_cats.remove('unknown')

df_exp1 = df[df['category_original'].isin(valid_cats)].copy()
df_exp1['category'] = df_exp1['category_original']

df_shuffled = df_exp1.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:split]
test_df = df_shuffled[split:]

print(f"Articole: {len(df_exp1)} | Categorii: {len(valid_cats)}")
print("Categorii:", ', '.join(sorted(valid_cats)[:10]), "...")

acc1, n_cats1, pred1, true1 = evaluate_model(train_df, test_df)
results.append(("25 categorii (Original)", n_cats1, acc1))
print(f"✅ ACURATEȚE: {acc1:.2f}%")

# EXPERIMENT 2: ~15 CATEGORII (Simplificare Conservativă)
print("\n" + "=" * 70)
print("EXPERIMENT 2: ~15 CATEGORII (SIMPLIFICARE CONSERVATIVĂ)")
print("=" * 70)

df['category_simple'] = df['category_original'].apply(simplify_category_conservative)

category_counts2 = df['category_simple'].value_counts()
valid_cats2 = category_counts2[category_counts2 >= MIN_ARTICLES].index.tolist()
if 'unknown' in valid_cats2:
    valid_cats2.remove('unknown')

df_exp2 = df[df['category_simple'].isin(valid_cats2)].copy()
df_exp2['category'] = df_exp2['category_simple']

df_shuffled = df_exp2.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:split]
test_df = df_shuffled[split:]

print(f"Articole: {len(df_exp2)} | Categorii: {len(valid_cats2)}")
print("Categorii:", ', '.join(sorted(valid_cats2)))

acc2, n_cats2, pred2, true2 = evaluate_model(train_df, test_df)
results.append(("~15 categorii (Conservativ)", n_cats2, acc2))
print(f"✅ ACURATEȚE: {acc2:.2f}%")

# EXPERIMENT 3: 6-7 CATEGORII (Simplificare Agresivă)
print("\n" + "=" * 70)
print("EXPERIMENT 3: 6-7 CATEGORII (SIMPLIFICARE AGRESIVĂ)")
print("=" * 70)

df['category_mega'] = df['category_original'].apply(simplify_category_aggressive)

category_counts3 = df['category_mega'].value_counts()
valid_cats3 = category_counts3[category_counts3 >= MIN_ARTICLES].index.tolist()
if 'unknown' in valid_cats3:
    valid_cats3.remove('unknown')
if 'other' in valid_cats3:
    valid_cats3.remove('other')

df_exp3 = df[df['category_mega'].isin(valid_cats3)].copy()
df_exp3['category'] = df_exp3['category_mega']

df_shuffled = df_exp3.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:split]
test_df = df_shuffled[split:]

print(f"Articole: {len(df_exp3)} | Categorii: {len(valid_cats3)}")
print("Categorii:", ', '.join(sorted(valid_cats3)))
print("\nDistribuție:")
print(df_exp3['category'].value_counts())

acc3, n_cats3, pred3, true3 = evaluate_model(train_df, test_df)
results.append(("6-7 categorii (Agresiv)", n_cats3, acc3))
print(f"✅ ACURATEȚE: {acc3:.2f}%")

# ============================================================
# COMPARAȚIE FINALĂ
# ============================================================

print("\n" + "=" * 70)
print("📊 COMPARAȚIE FINALĂ - ACURATEȚE vs NUMĂR DE CATEGORII")
print("=" * 70)

print("\n{:<35} | {:>12} | {:>12}".format("Configurație", "Nr. Categorii", "Acuratețe"))
print("-" * 70)
for name, n_cats, acc in results:
    print("{:<35} | {:>12} | {:>11.2f}%".format(name, n_cats, acc))

print("\n" + "=" * 70)
print("📈 CONCLUZIE")
print("=" * 70)

best_config = max(results, key=lambda x: x[2])
print(f"\n✅ Cea mai bună configurație: {best_config[0]}")
print(f"   Acuratețe: {best_config[2]:.2f}% cu {best_config[1]} categorii")

improvement = best_config[2] - results[0][2]
print(f"\n💡 Îmbunătățire față de configurația originală: {improvement:+.2f}%")

print("\n📝 Observații:")
print("   • Cu mai puține categorii → acuratețe mai mare")
print("   • Cu prea multe categorii → confuzie între clase similare")
print("   • Trade-off: granularitate vs. acuratețe")

print("\n" + "=" * 70)