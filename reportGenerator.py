import random
from transformers import pipeline, set_seed
from gensim.models import Word2Vec
import numpy as np

# Load Word2Vec model
w2v_model = Word2Vec.load("models/wheat_disease_model.model")

# Load local text generation pipeline
set_seed(42)
text_generator = pipeline("text-generation", model="distilgpt2")

# Mock classifier
DISEASES = [
    "yellow rust", "brown rust", "stripe rust",
    "septoria", "septoria tritici", "powdery mildew"
]

def mock_image_classifier():
    return random.choice(DISEASES)

def get_context_terms(disease, topn=10):
    tokens = disease.lower().split()
    if all(t in w2v_model.wv for t in tokens):
        if len(tokens) > 1:
            vec = np.mean([w2v_model.wv[t] for t in tokens], axis=0)
            return [term for term, _ in w2v_model.wv.similar_by_vector(vec, topn=topn)]
        else:
            return [term for term, _ in w2v_model.wv.most_similar(disease, topn=topn)]
    return []

def generate_prompt(disease, context_terms):
    return (
        f"Generate a technical but concise report on '{disease}' in wheat crops.\n"
        f"Include details such as: symptoms, impact on yield, economic consequences,\n"
        f"and recommended treatment options.\n"
        f"Relevant terms to include: {', '.join(context_terms)}.\n"
        f"Report:"
    )

def generate_disease_report():
    disease = mock_image_classifier()
    context_terms = get_context_terms(disease)
    prompt = generate_prompt(disease, context_terms)
    print("Prompt:\n", prompt)
    output = text_generator(prompt, max_length=300, num_return_sequences=1)
    print("\nGenerated Report:\n")
    print(output[0]['generated_text'])

if __name__ == "__main__":
    generate_disease_report()
