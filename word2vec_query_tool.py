import streamlit as st
import numpy as np
from gensim.models import Word2Vec

# Load model
@st.cache_resource
def load_model():
    return Word2Vec.load("models/wheat_disease_model.model")

model = load_model()

# UI
st.title("🌾 Wheat Disease Word2Vec Explorer")
st.markdown("""
Explore semantic relationships between wheat diseases, yield loss, and treatments using a custom-trained Word2Vec model.
""")

# --- Similar Terms ---
st.subheader("🔍 Find Similar Terms")
query_term = st.text_input("Enter a word or phrase:", "yellow rust")
topn = st.slider("Number of similar terms", 5, 20, 10)

if query_term:
    tokens = query_term.lower().split()
    if all(t in model.wv for t in tokens):
        if len(tokens) > 1:
            vector = np.mean([model.wv[t] for t in tokens], axis=0)
            similar = model.wv.similar_by_vector(vector, topn=topn)
        else:
            similar = model.wv.most_similar(query_term, topn=topn)
        st.table(similar)
    else:
        st.warning("Some terms not in vocabulary.")

# --- Analogy Reasoning ---
st.subheader("🧠 Analogy Reasoning")
term_a = st.text_input("A (e.g. yellow rust)", "yellow rust")
term_b = st.text_input("B (e.g. fungicide)", "fungicide")
term_c = st.text_input("C (e.g. septoria)", "septoria")

if term_a and term_b and term_c:
    try:
        vec = model.wv[term_b] - model.wv[term_a] + model.wv[term_c]
        results = model.wv.similar_by_vector(vec, topn=topn)
        st.write(f"D is to {term_b} as {term_c} is to:")
        st.table(results)
    except KeyError as e:
        st.warning(f"Word not found in vocabulary: {e}")

# --- Nearest Yield/Treatment Terms ---
st.subheader("📈 Disease Relationship Explorer")
disease_options = [
    "yellow rust", "brown rust", "stripe rust",
    "septoria", "septoria tritici", "powdery mildew"
]
disease_selected = st.selectbox("Select a disease", disease_options)

yield_terms = ["yield", "loss", "reduction", "damage", "impact"]
treatment_terms = ["fungicide", "control", "spray", "treatment", "resistance"]

if disease_selected:
    def avg_similarity(base, terms):
        base_tokens = base.split()
        if all(t in model.wv for t in base_tokens):
            vec = np.mean([model.wv[t] for t in base_tokens], axis=0)
            return sorted(
                [(t, float(np.dot(vec, model.wv[t]) / (np.linalg.norm(vec) * np.linalg.norm(model.wv[t])))) for t in terms if t in model.wv],
                key=lambda x: x[1], reverse=True
            )
        return []

    st.markdown("**Top related yield impact terms:**")
    st.table(avg_similarity(disease_selected, yield_terms))

    st.markdown("**Top related treatment terms:**")
    st.table(avg_similarity(disease_selected, treatment_terms))
