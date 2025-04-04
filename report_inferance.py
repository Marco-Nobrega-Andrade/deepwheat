from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gensim.models import Word2Vec
import torch
import numpy as np

# Load fine-tuned model and tokenizer
model_path = "finetuned_distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # redefine pad_token
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# Load Word2Vec model
w2v_model = Word2Vec.load("models/wheat_disease_model.model")

def get_context_terms(disease, topn=10):
    tokens = disease.lower().split()
    if all(t in w2v_model.wv for t in tokens):
        if len(tokens) > 1:
            vec = np.mean([w2v_model.wv[t] for t in tokens], axis=0)
            return [term for term, _ in w2v_model.wv.similar_by_vector(vec, topn=topn)]
        else:
            return [term for term, _ in w2v_model.wv.most_similar(disease, topn=topn)]
    return []

def build_prompt(disease, terms):
    return (
        f"Write a concise agricultural report for farmers.\n"
        f"Disease: {disease}\n"
        f"Symptoms, impact on yield, treatments, and economic risk.\n"
        f"Relevant terms: {', '.join(terms)}\n"
        f"Report:"
    )

def generate_report(disease, max_length=512):
    context_terms = get_context_terms(disease)
    if not context_terms:
        print("[!] Warning: Word2Vec could not find context terms for this disease. Proceeding with generic prompt.")
        context_terms = ["yield", "loss", "fungicide", "control"]
    else:
        print("[+] Word2Vec context terms:", ", ".join(context_terms))

    prompt = build_prompt(disease, context_terms)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=0.9,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    disease_name = input("Enter a wheat disease (e.g., yellow rust): ").strip()
    report = generate_report(disease_name)
    print("\n=== Generated Report ===\n")
    print(report)
