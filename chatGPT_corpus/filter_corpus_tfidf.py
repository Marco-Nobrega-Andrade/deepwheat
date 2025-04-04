import os
import re
import time
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# === CONFIGURATION ===
INPUT_FILE = "data/corpus/complete_corpus.txt"
OUTPUT_FILE = "filtered_sentences.txt"
THRESHOLD = 0.05
MAX_FEATURES = 10000

DOMAIN_KEYWORDS = [
    # Diseases
    "yellow rust", "brown rust", "stripe rust", "leaf rust", "stem rust",
    "septoria", "mildew", "powdery mildew", "blight",

    # Treatments / Chemicals
    "fungicide", "triazole", "strobilurin", "spray program", "chemical treatment",
    "biological control", "disease resistance", "pesticide", "protectant",

    # Impact
    "yield loss", "grain loss", "productivity decline", "infection severity",
    "disease spread", "crop failure", "economic loss", "financial impact",

    # Management
    "crop rotation", "field monitoring", "early detection", "scouting",
    "disease management", "pathogen control", "integrated pest management", "threshold level",

    # Symptoms
    "chlorosis", "lesion", "spore", "pustule", "discoloration"
]

SKIP_PATTERNS = [
    r'\b(et al|doi|http[s]?://|www\.)\b',
    r'\[\d+\]',
    r'\(.*\d{4}.*\)',
    r'\b[A-Z]{1,2}\d{1,3}\b',
    r'acknowledg(e)?ments?',
    r'\btable\b|\bfigure\b',
    r'^[^a-zA-Z]{0,3}$'
]

# === CLEAN & SPLIT ===
def is_irrelevant(sentence):
    return any(re.search(p, sentence.lower()) for p in SKIP_PATTERNS)

def clean_and_split_sentences(file_path):
    all_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        text = re.sub(r'\s+', ' ', text)
        sentences = sent_tokenize(text)

        for s in sentences:
            s_clean = re.sub(r'[^a-zA-Z\s]', '', s).strip()
            if len(s_clean.split()) < 5 or is_irrelevant(s_clean):
                continue

            # Optional: pre-filter by keyword presence
            #if any(kw in s_clean for kw in DOMAIN_KEYWORDS):
            all_sentences.append(s_clean)

    return all_sentences

# === TF-IDF FILTERING ===
def tfidf_filter(sentences, keywords, threshold=0.1):
    print("🔠 Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=MAX_FEATURES
    )

    start = time.time()
    X = vectorizer.fit_transform(sentences)
    print(f"✅ TF-IDF vectorization completed in {time.time() - start:.2f} seconds.")

    feature_names = vectorizer.get_feature_names_out()
    relevant_sentences = []

    for idx, vec in enumerate(X):
        score = sum(
            vec[0, vectorizer.vocabulary_.get(kw)]
            for kw in keywords if kw in vectorizer.vocabulary_
        )
        if score >= threshold:
            relevant_sentences.append(sentences[idx])

    return relevant_sentences

# === MAIN ===
def main():
    print("🔍 Loading and preprocessing corpus...")
    start = time.time()
    sentences = clean_and_split_sentences(INPUT_FILE)
    print(f"✅ Loaded {len(sentences)} pre-filtered sentences in {time.time() - start:.2f}s.")

    print("📊 Running TF-IDF keyword scoring...")
    relevant = tfidf_filter(sentences, DOMAIN_KEYWORDS, THRESHOLD)
    print(f"🎯 Found {len(relevant)} relevant sentences.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in relevant:
            f.write(line + "\n")

    print(f"💾 Saved relevant sentences to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()