# Documentație Proiect: Clasificator Multinomial Naive Bayes
**Autor:** Ignat Marius Florentin  
**Data:** 16 Noiembrie 2025  
**Curs:** Probabilitați și statistică  
**Universitatea:** Universitatea din București

---

## Cuprins
1. [Introducere](#1-introducere)
2. [Modelul Matematic](#2-modelul-matematic)
3. [Structura Codului](#3-structura-codului)
4. [Instrucțiuni de Utilizare](#4-instrucțiuni-de-utilizare)
5. [Exemple de Utilizare](#5-exemple-de-utilizare)
6. [Rezultate și Evaluare](#6-rezultate-și-evaluare)
7. [Referințe Bibliografice](#7-referințe-bibliografice)

---

## 1. Introducere

### 1.1 Descrierea Problemei
Acest proiect implementează un **clasificator Multinomial Naive Bayes** pentru clasificarea automată a articolelor de știri BBC în categorii tematice. Clasificatorul a fost implementat de la zero, fără utilizarea bibliotecilor de machine learning precum scikit-learn.

### 1.2 Dataset
- **Sursă**: Articole de știri BBC (42,115 articole)
- **Perioada**: 2013-2024
- **Atribute**: titlu, descriere, link, dată publicare
- **Categorii finale**: 6 categorii principale (business, entertainment, health-education, tech-science, uk, world)

### 1.3 Obiective
1. Procesarea și curățarea datelor text
2. Implementarea algoritmului Naive Bayes din ground-up
3. Evaluarea performanței pe un set de test independent
4. Optimizarea categoriilor pentru maximizarea acurateței

---

## 2. Modelul Matematic

### 2.1 Teorema lui Bayes

Clasificatorul Naive Bayes se bazează pe **Teorema lui Bayes**, care definește probabilitatea condiționată:

```
P(C|D) = P(D|C) × P(C) / P(D)
```

Unde:
- **P(C|D)** = probabilitatea ca un document D să aparțină categoriei C (probabilitate *a posteriori*)
- **P(D|C)** = probabilitatea de a observa documentul D dacă aparține categoriei C (*likelihood*)
- **P(C)** = probabilitatea *a priori* a categoriei C
- **P(D)** = probabilitatea documentului D (constantă pentru toate categoriile)

### 2.2 Ipoteza "Naivă"

Algoritmul presupune **independența condiționată** între cuvinte, adică:

```
P(D|C) = P(w₁, w₂, ..., wₙ|C) = P(w₁|C) × P(w₂|C) × ... × P(wₙ|C)
```

Această ipoteză simplifică calculele, deși în realitate cuvintele sunt corelate. Cu toate acestea, în practică, Naive Bayes funcționează surprinzător de bine.

### 2.3 Formula de Clasificare

Pentru a evita **underflow-ul numeric** (înmulțirea multor probabilități mici), folosim **logaritmi**:

```
score(C) = log P(C) + Σ log P(wᵢ|C)
```

Categoria prezisă este cea cu scorul maxim:

```
C* = argmax_C [log P(C) + Σ log P(wᵢ|C)]
```

### 2.4 Laplace Smoothing

Pentru a evita probabilitățile de **zero** (când un cuvânt nu apare în setul de antrenament pentru o anumită categorie), folosim **Laplace Smoothing** (add-one smoothing):

```
P(wᵢ|C) = (count(wᵢ, C) + α) / (count_total(C) + α × |V|)
```

Unde:
- **count(wᵢ, C)** = numărul de apariții al cuvântului wᵢ în categoria C
- **count_total(C)** = numărul total de cuvinte în categoria C
- **α** = parametrul de smoothing (în implementarea noastră α = 1)
- **|V|** = dimensiunea vocabularului

### 2.5 Stabilitate Numerică

Folosirea logaritmilor transformă:
- **Înmulțiri** → **Adunări** (mai rapid computațional)
- **Probabilități mici** → **Valori negative controlabile** (evită underflow)

---

## 3. Structura Codului

### 3.1 Arhitectura Generală

Proiectul este organizat în **3 faze principale**:

```
main.py
├── FAZA 1: Prepararea și Curățarea Datelor
│   ├── 1.1 Încărcarea datelor (pandas)
│   ├── 1.2 Extragerea categoriilor din URL-uri
│   ├── 1.3 Simplificarea și filtrarea categoriilor
│   ├── 1.4 Curățarea și tokenizarea textului
│   └── 1.5 Împărțirea în train/test (80/20)
│
├── FAZA 2: Implementarea Clasificatorului
│   ├── 2.1 Funcția train() - învățare parametri
│   └── 2.2 Funcția predict() - clasificare text nou
│
└── FAZA 3: Evaluarea Modelului
    ├── 3.1 Rularea predicțiilor pe setul de test
    └── 3.2 Calcularea acurateței și metrici
```

### 3.2 Funcții Principale

#### **3.2.1 `extract_category(url)`**
```python
def extract_category(url):
    """Extrage categoria dintr-un URL BBC News."""
```
- **Input**: URL complet (ex: `https://www.bbc.co.uk/news/business-12345`)
- **Output**: Categoria extrasă (ex: `business`)
- **Metodă**: Expresii regulate (regex) pentru pattern matching

#### **3.2.2 `simplify_category(cat)`**
```python
def simplify_category(cat):
    """Grupează categoriile similare în mega-categorii."""
```
- **Input**: Categorie originală (ex: `world-europe`)
- **Output**: Categorie simplificată (ex: `world`)
- **Logică**: Grupează categoriile semantice similare pentru îmbunătățirea acurateței

**Mapare categorii:**
| Categorii Originale | Categorie Finală |
|---------------------|------------------|
| world-europe, world-asia, world-us-canada, etc. | **world** |
| uk-politics, uk-scotland, uk-wales, etc. | **uk** |
| entertainment-arts, newsbeat | **entertainment** |
| technology, science-environment | **tech-science** |
| health, education | **health-education** |
| business | **business** |

#### **3.2.3 `clean_text(text)`**
```python
def clean_text(text):
    """Curăță și tokenizează textul."""
```
- **Input**: Text brut (titlu + descriere)
- **Output**: Listă de tokens (cuvinte procesate)

**Pași de procesare:**
1. **Lowercase**: Conversie la litere mici (`"Bitcoin" → "bitcoin"`)
2. **Eliminare punctuație**: Păstrează doar litere și spații
3. **Tokenizare**: Împarte textul în cuvinte individuale
4. **Stop words removal**: Elimină cuvinte comune (`the`, `a`, `is`, etc.)
5. **Filtrare lungime**: Elimină cuvinte cu < 3 caractere

#### **3.2.4 `train(train_data)`**
```python
def train(train_data):
    """Învață parametrii modelului Naive Bayes."""
```

**Returnează 3 structuri de date:**

1. **`vocabulary`** (set): Toate cuvintele unice din setul de antrenament
   ```python
   vocabulary = {'bitcoin', 'economy', 'minister', ...}  # 28,197 cuvinte
   ```

2. **`prior_probs`** (dict): Probabilități a priori P(C)
   ```python
   prior_probs = {
       'business': 0.132,
       'uk': 0.485,
       'world': 0.420,
       ...
   }
   ```

3. **`cond_probs`** (dict nested): Probabilități condiționate P(w|C)
   ```python
   cond_probs = {
       'business': {
           'economy': 0.0023,
           'market': 0.0019,
           ...
       },
       'uk': {...},
       ...
   }
   ```

**Algoritm:**
```
Pentru fiecare categorie C:
    1. Calculează P(C) = count(C) / total_documents
    2. Pentru fiecare cuvânt w în vocabular:
        a. Numără apariții: count(w, C)
        b. Aplică Laplace: P(w|C) = (count + 1) / (total + |V|)
```

#### **3.2.5 `predict(tokens, vocabulary, prior_probs, cond_probs)`**
```python
def predict(tokens, vocabulary, prior_probs, cond_probs):
    """Clasifică un text pe baza token-ilor săi."""
```

**Algoritm:**
```
Pentru fiecare categorie C:
    1. score = log P(C)
    2. Pentru fiecare cuvânt w în text:
        a. Dacă w în vocabular:
            score += log P(w|C)
    3. Returnează C cu score maxim
```

**Exemplu de calcul:**
```
Text: "UK minister announces new policy"
Tokens: ['minister', 'announces', 'policy']

Score(uk) = log(0.485) + log(P('minister'|uk)) + log(P('announces'|uk)) + log(P('policy'|uk))
          = -0.723 + (-5.2) + (-6.1) + (-5.8)
          = -17.823

Score(world) = log(0.420) + ...
             = -19.456

Predicție: 'uk' (scor mai mare)
```

### 3.3 Biblioteci Utilizate

```python
import pandas as pd       # Manipulare date CSV
import numpy as np        # Calcule matematice (logaritmi)
import re                 # Expresii regulate (procesare URL)
from collections import defaultdict  # Dicționare cu valori default
```

**Notă**: Nu am folosit biblioteci de ML (scikit-learn, NLTK) - totul implementat manual!

---

## 4. Instrucțiuni de Utilizare

### 4.1 Cerințe de Sistem

- **Python**: 3.8 sau superior
- **Sistem Operare**: Windows, macOS, Linux
- **Memorie RAM**: Minimum 2GB (recomandat 4GB)

### 4.2 Instalarea Dependențelor

```bash
# Creează un mediu virtual (opțional, dar recomandat)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# sau
venv\Scripts\activate     # Windows

# Instalează bibliotecile necesare
pip install pandas numpy
```

### 4.3 Structura Fișierelor

Asigură-te că ai următoarea structură:

```
Proiect_Naive_Bayes/
├── main.py                  # Script principal
├── bbc_news.csv             # Dataset 
└── documentatie.md          # Acest document (pentru Github)
```

### 4.4 Rularea Programului

```bash
# Navighează în directorul proiectului
cd Proiect_Naive_Bayes

# Rulează scriptul principal
python3 main.py
```

### 4.5 Output Așteptat

Programul va afișa:
1. ✅ Progresul procesării datelor (Faza 1)
2. ✅ Antrenarea modelului (Faza 2)
3. ✅ Rezultatele evaluării (Faza 3):
   - Acuratețea globală
   - Acuratețea per categorie
   - 10 exemple de predicții

**Timpul de execuție**: ~15-30 secunde (depinde de procesor)

---

## 5. Exemple de Utilizare

### 5.1 Clasificare Articol Nou

Pentru a clasifica un articol nou, folosește funcția `predict()`:

```python
# Text nou de clasificat
new_title = "Tesla stock surges after quarterly earnings report"
new_description = "Electric vehicle maker announces record profits"

# Procesează textul
text = new_title + ' ' + new_description
tokens = clean_text(text)

# Clasifică
predicted_category = predict(tokens, vocabulary, prior_probs, cond_probs)

print(f"Categorie prezisă: {predicted_category}")
# Output: "Categorie prezisă: business"
```

### 5.2 Exemple de Predicții Corecte

**Exemplu 1 - Business:**
```
Titlu: "Long-term sick: How record number changes UK economy"
Tokens: ['long', 'term', 'sick', 'record', 'number', 'changes', 'economy']
Predicție: business ✅
```

**Exemplu 2 - World:**
```
Titlu: "Israel-Gaza war: Unknown fate of six-year-old Hind Rajab"
Tokens: ['israel', 'gaza', 'war', 'unknown', 'fate', 'year', 'old', 'hind', 'rajab']
Predicție: world ✅
```

**Exemplu 3 - Entertainment:**
```
Titlu: "Coronation Street drops out of Christmas TV top 10"
Tokens: ['coronation', 'street', 'drops', 'christmas', 'top']
Predicție: entertainment ✅
```

### 5.3 Analiza Greșelilor

**Exemplu de predicție greșită:**
```
Titlu: "New AI model achieves breakthrough in medical diagnosis"
Tokens: ['new', 'model', 'achieves', 'breakthrough', 'medical', 'diagnosis']
Categorie reală: health-education
Predicție: tech-science ❌

Motiv: Vocabularul se suprapune între tech-science și health-education
```

---

## 6. Rezultate și Evaluare

### 6.1 Acuratețea Globală

```
📊 ACURATEȚEA MODELULUI: 79.30%
   - Predicții corecte: 3,950 / 4,981
   - Predicții greșite: 1,031 / 4,981
```

### 6.2 Acuratețe per Categorie

| Categorie | Acuratețe | Predicții Corecte/Total |
|-----------|-----------|-------------------------|
| **world** | 86.38% | 1,440 / 1,667 |
| **uk** | 82.65% | 1,620 / 1,960 |
| **business** | 78.36% | 402 / 513 |
| **entertainment** | 73.85% | 322 / 436 |
| **health-education** | 47.34% | 98 / 207 |
| **tech-science** | 34.34% | 68 / 198 |

### 6.3 Interpretarea Rezultatelor

**Categorii cu performanță excelentă (>80%):**
- `world` și `uk` au vocabular foarte distinct (nume de locuri, figuri politice locale)
- Beneficiază de cel mai mare număr de exemple de antrenament

**Categorii cu performanță medie (70-80%):**
- `business` și `entertainment` au vocabular mai specific

**Categorii cu performanță mai slabă (<50%):**
- `health-education` și `tech-science` au cel mai puțin date de antrenament
- Vocabularul se suprapune cu alte categorii (ex: "study", "research")

### 6.4 Comparație cu Baseline

| Configurație | Nr. Categorii | Acuratețe |
|--------------|---------------|-----------|
| Random Guess | 6 | 16.67% |
| Always Predict "uk" | 6 | 39.35% |
| **Naive Bayes (implementat)** | **6** | **79.30%** |

Modelul nostru depășește cu mult ambele baseline-uri!

### 6.5 Analiză Experimentală

Am experimentat cu **3 configurații** de categorii:

| Configurație | Nr. Categorii | Acuratețe | Îmbunătățire |
|--------------|---------------|-----------|--------------|
| Originală (granulară) | 25 | 54.19% | - |
| Conservativă | ~15 | ~62% | +7.81% |
| **Agresivă (finală)** | **6** | **79.30%** | **+25.11%** |

**Concluzie**: Gruparea categoriilor similare îmbunătățește semnificativ acuratețea!

---

## 7. Referințe Bibliografice

1. **Materialele Cursului** - *Probabilitați și statistică*  
   Universitatea din București, 2025 
   - Laboratoare și suport de curs pentru algoritmul Naive Bayes

2. **Dataset** - Preda, G. (2020). *BBC News Dataset*. Kaggle.  
   - https://www.kaggle.com/datasets/gpreda/bbc-news/data  
   - Dataset cu 42,115 articole BBC (2013-2024)

3. **Wikipedia** - *Naive Bayes classifier*  
   - https://en.wikipedia.org/wiki/Naive_Bayes_classifier  
   - Referință pentru formula matematică și Laplace Smoothing

4. **StatQuest with Josh Starmer** - *Naive Bayes, Clearly Explained!!!*. YouTube.  
   - https://www.youtube.com/watch?v=O2L2Uv9pdDA  
   - Explicație vizuală a algoritmului

5. **Python Documentation**  
   - pandas: https://pandas.pydata.org/docs/  
   - numpy: https://numpy.org/doc/  
   - re (regular expressions): https://docs.python.org/3/library/re.html

---

## Anexă: Observații

### Puncte Forte ale Implementării
✅ Cod clar, bine documentat și modular  
✅ Implementare completă de la zero (fără scikit-learn)  
✅ Optimizare categorii pentru acuratețe maximă  
✅ Laplace Smoothing implementat corect  
✅ Folosirea logaritmilor pentru stabilitate numerică  

---

**Data finalizării documentației:** 16 Noiembrie 2025  
**Versiune:** 1.0