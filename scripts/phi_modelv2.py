import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import os


def load_model_and_tokenizer(base_model="microsoft/phi-2"):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


def load_text_dataset(txt_path: str):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = Dataset.from_dict({"text": lines})
    return dataset


def prepare_dataset(dataset: Dataset, tokenizer, max_length=128):
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


def train(model, tokenizer, tokenized_dataset, output_dir="./phi2-finetuned"):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    return trainer


def save_model(trainer, tokenizer, output_dir="./phi2-finetuned"):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved in: {output_dir}")


def main():
    txt_path = "./extracted_corpus/dataset.txt"
    print(f"📂 Loading dataset from: {txt_path}")
    raw_dataset = load_text_dataset(txt_path)

    print("🔁 Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("🧹 Tokenizing dataset...")
    tokenized_dataset = prepare_dataset(raw_dataset, tokenizer)

    print("🚀 Training model...")
    trainer = train(model, tokenizer, tokenized_dataset)

    print("💾 Saving trained model...")
    save_model(trainer, tokenizer)

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()