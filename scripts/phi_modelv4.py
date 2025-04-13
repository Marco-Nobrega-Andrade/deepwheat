import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import os
import math
import numpy as np
from typing import Dict, Any, List, Tuple
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    name: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    epochs: int
    max_steps: int
    quantization: Dict[str, Any] = None

    def get_output_dir(self, base_dir="./phi2-finetuned"):
        return os.path.join(base_dir, self.name)


def get_model_configs() -> List[ModelConfig]:
    return [
        ModelConfig(
            name="phi2-efficient",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum_steps=4,
            learning_rate=1e-4,
            epochs=3,
            max_steps=300,
        ),

        ModelConfig(
            name="phi2-medium",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum_steps=4,
            learning_rate=2e-4,
            epochs=5,
            max_steps=500,
        ),

        ModelConfig(
            name="phi2-powerful",
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            batch_size=4,
            grad_accum_steps=4,
            learning_rate=3e-4,
            epochs=10,
            max_steps=1000,
        ),
        ModelConfig(
            name="phi2-quantized",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum_steps=4,
            learning_rate=2e-4,
            epochs=5,
            max_steps=500,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.float16
            },
        ),
    ]


def load_model_and_tokenizer(config: ModelConfig, base_model="microsoft/phi-2"):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": None,
        "torch_dtype": torch.float16
    }

    if config.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.quantization.get("load_in_4bit", True),
            bnb_4bit_use_double_quant=config.quantization.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=config.quantization.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=config.quantization.get("bnb_4bit_compute_dtype", torch.float16)
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        **model_kwargs
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.train()

    return model, tokenizer


def load_text_dataset(txt_path: str, split_ratio=0.3):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = Dataset.from_dict({"text": lines})

    split = dataset.train_test_split(test_size=split_ratio, seed=42)
    return split["train"], split["test"]


def prepare_dataset(dataset: Dataset, tokenizer, max_length):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized_dataset


def train(model, tokenizer, train_dataset, eval_dataset, config: ModelConfig):
    model.train()

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = config.get_output_dir()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        gradient_checkpointing=False,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        learning_rate=config.learning_rate,
        warmup_steps=int(0.1 * config.max_steps),
        weight_decay=0.01,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    return trainer


def calculate_perplexity(model, tokenizer, eval_dataset):
    model.eval()
    total_loss = 0
    total_length = 0

    device = model.device

    with torch.no_grad():
        for i in range(0, len(eval_dataset), 4):
            batch = eval_dataset[i:i+4]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()

            seq_lengths = torch.sum(attention_mask).item()
            total_loss += loss * seq_lengths
            total_length += seq_lengths

    perplexity = math.exp(total_loss / total_length)
    return perplexity


def evaluate_model(model_path, tokenizer, eval_dataset, is_peft=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if is_peft:
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

    perplexity = calculate_perplexity(model, tokenizer, eval_dataset)
    return perplexity


def save_model(trainer, tokenizer, output_dir):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved at: {output_dir}")


def save_results(results, output_path="./training_results.json"):
    serializable_results = []
    for result in results:
        result_dict = dict(result)
        if "config" in result_dict:
            result_dict["config"] = asdict(result_dict["config"])
        serializable_results.append(result_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved at: {output_path}")


def main():
    txt_path = "./extracted_corpus/dataset.txt"
    print(f"📂 Loading dataset from: {txt_path}")
    train_raw, eval_raw = load_text_dataset(txt_path)

    configs = get_model_configs()

    for config in configs:
        print(f"\n🚀 Starting training for configuration: {config.name}")
        print("="*80)
        print(f"Configuration: {config}")

        print("🔁 Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(config)

        print("🧹 Tokenizing dataset...")
        train_dataset = prepare_dataset(train_raw, tokenizer, max_length=64)
        eval_dataset = prepare_dataset(eval_raw, tokenizer, max_length=64)

        print(f"🚀 Training model {config.name}...")
        trainer = train(model, tokenizer, train_dataset, eval_dataset, config)

        output_dir = config.get_output_dir()
        print(f"💾 Saving trained model at {output_dir}...")
        save_model(trainer, tokenizer, output_dir)


        del model, tokenizer, trainer
        torch.cuda.empty_cache()



    print("\n✅ All trainings completed successfully!")


if __name__ == "__main__":
    main()