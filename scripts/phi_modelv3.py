import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback
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
        device_map=None, 
        torch_dtype=torch.float16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.train() 

    return model, tokenizer


from datasets import Dataset, DatasetDict

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


def train(model, tokenizer, train_dataset, eval_dataset, output_dir="./phi2-finetuned"):
    model.train() 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False, 
        num_train_epochs=300, 
        max_steps=500, 
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2, 
        fp16=True, 
        bf16=False, 
        optim="paged_adamw_8bit", 
        lr_scheduler_type="cosine", 
        learning_rate=2e-4,
        warmup_steps=50, 
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


def save_model(trainer, tokenizer, output_dir="./phi2-finetuned"):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved in: {output_dir}") 


def main():
    
    txt_path = "./extracted_corpus/dataset.txt"
    print(f"📂 Loading dataset from: {txt_path}") 
    train_raw, eval_raw = load_text_dataset(txt_path)

    print("🔁 Loading model and tokenizer...") 
    model, tokenizer = load_model_and_tokenizer()

    print("🧹 Tokenizing dataset...") 
    train_dataset = prepare_dataset(train_raw, tokenizer, max_length=64) 
    eval_dataset = prepare_dataset(eval_raw, tokenizer, max_length=64)

    print("🚀 Training model (quick/test mode)...") 
    trainer = train(model, tokenizer, train_dataset, eval_dataset)

    print("💾 Saving trained model...") 
    save_model(trainer, tokenizer)

    print("✅ Training completed successfully!") 


if __name__ == "__main__":
    main()