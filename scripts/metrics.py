import torch
import os
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, TextGenerationPipeline
)
from peft import PeftModel, PeftConfig
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import json
from math import exp
import re

base_path = r"./phi2-finetuned"

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

def load_finetuned_model(model_path):
    print(f"📂 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def calculate_perplexity(data):
    print("🧮 Calculating perplexity...")
    evaluations = [entry for entry in data["log_history"] if "eval_loss" in entry]
    for evaluation in evaluations:
        print(json.dumps(evaluation, indent=2))
    eval_losses = [evaluation["eval_loss"] for evaluation in evaluations]
    print(eval_losses)
    avg_loss = np.mean(eval_losses)
    print(f"Average loss: {avg_loss}")
    perplexities = [exp(loss) for loss in eval_losses]
    avg_perplexity = np.mean(perplexities)
    return perplexities, avg_perplexity, avg_loss, eval_losses

def evaluate_model(model_path, prompts, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"🔄 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    print(f"✅ Model loaded: {model_path}")
    results = [] 
    print(f"🔄 Evaluating model: {model_path}")
    for prompt in prompts:
        formatted_prompt = f"Instruction: Answer briefly in 1 or 2 short sentences only.\nQuestion: {prompt.strip()}\nAnswer:"
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        HumanEval = {
            "prompt": prompt,
            "output": response.strip(),
            "Eval": "[A = x , F = x , O = x]"
        }
        results.append(HumanEval)
    return results

def save_metrics_to_file(metrics_data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("===== MODEL EVALUATION METRICS REPORT =====\n")
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=====================================================\n\n")
        for item in metrics_data:
            f.write(f"Model: {item['model_name']}\n")
            f.write(f"Average perplexity: {item['avg_perplexity']:.4f}\n")
            f.write(f"Individual perplexities: {', '.join([f'{p:.4f}' for p in item['perplexities']])}\n")
            f.write(f"Average losses: {item['avg_loss']:.4f}\n")
            f.write(f"Individual losses: {', '.join([f'{l:.4f}' for l in item['eval_losses']])}\n\n")
            f.write("HumanEval Results:\n")
            for result in item["HumanEval"]:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Output: {result['output']}\n")
                f.write(f"Evaluation: {result['Eval']}\n")
                f.write("\n")
            f.write("=====================================================\n")
            f.write("\n")
    print(f"✅ Metrics saved to file: {file_path}")
        
def get_latest_checkpoint_path(model_path):
    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(re.search(r"\d+", x).group()), reverse=True)
    return os.path.join(model_path, checkpoints[0])

def main():
    metrics_file_path = "./model_evaluation_metrics.txt"
    metrics_data = []
    for subdir in os.listdir(base_path):
        model_dir = os.path.join(base_path, subdir)
        if os.path.isdir(model_dir):
            print(f"✅ Model found: {subdir}")
            try:
                results = evaluate_model(model_dir, prompts)
            except Exception as e:
                print(f"❌ Error evaluating model {model_dir}: {e}")
            print(f"🔍 Checking checkpoints in: {model_dir}")
            latest_ckpt = get_latest_checkpoint_path(model_dir)
            if latest_ckpt:
                trainer_state_path = os.path.join(latest_ckpt, "trainer_state.json")
                if os.path.exists(trainer_state_path):
                    print(f"\n📂 Trainer state from: {subdir}")
                    with open(trainer_state_path, "r", encoding="utf-8") as f:
                        trainer_state = json.load(f)
                    print(f"🔄 Calculating perplexity for: {subdir}")
                    perplexities, avg_perplexity, avg_loss, eval_losses = calculate_perplexity(trainer_state)
                    print(f"💾 Saving all metrics for: {subdir}")
                    metrics_data.append({
                        "model_name": subdir,
                        "perplexities": perplexities,
                        "avg_perplexity": avg_perplexity,
                        "avg_loss": avg_loss,
                        "eval_losses": eval_losses,
                        "HumanEval": results
                    })
                else:
                    print(f"⚠️ No trainer_state.json in: {latest_ckpt}")
            else:
                print(f"❌ No checkpoints in: {model_dir}")
    save_metrics_to_file(metrics_data, metrics_file_path)

if __name__ == "__main__":
    main()
