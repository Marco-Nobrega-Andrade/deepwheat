import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_saved_model_and_tokenizer(model_dir="./phi2-finetuned"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7):
    formatted_prompt = f"Instruction: Answer briefly in 1 or 2 short sentences only.\nQuestion: {prompt.strip()}\nAnswer:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

def main():
    model, tokenizer = load_saved_model_and_tokenizer()

    prompts = [
        "List all wheat diseases. Be brief and specific. Answer in one or two short sentences only.",
        "Describe the key visual symptoms of rust in wheat? Be brief and specific. Answer in one or two short sentences only.",
        "What is the most effective treatment method for rust in wheat? Be concise and specific with the methods or techiniques for treatment. Answer in one or two short sentences only.",
        "Describe the key visual symptoms of powdery mildew in wheat. Be brief and specific. Answer in one or two short sentences only.",
        "What is the most effective treatment method for powdery mildew in wheat? Be concise and specific with the methods or techiniques for treatment. Answer in one or two short sentences only.",
        "Describe the key visual symptoms of septoria leaf blotch in wheat. Be brief and specific. Answer in one or two short sentences only.",
        "What is the most effective treatment method for septoria in wheat? Be concise and specific with the methods or techiniques for treatment. Answer in one or two short sentences only.",
        "Name one integrated disease management strategy for wheat diseases. Keep it concise. Answer in one or two short sentences only.",
        "What is the role of crop rotation in managing wheat diseases? Answer briefly. Answer in one or two short sentences only.",
    ]

    for i, prompt in enumerate(prompts, 1):
        response = generate_response(model, tokenizer, prompt)
        print(f"--- Q{i} ---")
        print(prompt.strip(), "\n" f"Model response: {response.strip()}\n")

if __name__ == "__main__":
    main()
