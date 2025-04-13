import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import gc

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu"
        )

    return model, tokenizer

def apply_lora(model):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()

    return lora_model

def prepare_dataset(tokenizer, file_path, max_samples=100000):
    dataset = load_dataset("text", data_files=file_path)["train"]

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )

    return dataset

def train(MODEL_NAME, DATA_FILE, CONFIG):
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    model = apply_lora(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise ValueError("No trainable parameters found. Check LoRA configuration.")

    dataset = prepare_dataset(tokenizer, DATA_FILE, CONFIG.get("max_samples"))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    if not torch.cuda.is_available():
        CONFIG["fp16"] = False
        CONFIG["per_device_train_batch_size"] = 1
        CONFIG["gradient_accumulation_steps"] = 16
        CONFIG["optim"] = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        fp16=CONFIG["fp16"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        report_to=CONFIG["report_to"],
        save_total_limit=CONFIG["save_total_limit"],
        optim=CONFIG["optim"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        gradient_checkpointing=CONFIG["gradient_checkpointing"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        eval_steps=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])

    return model, tokenizer

def generate_text(model, tokenizer, prompt, device="cpu"):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    response = pipe(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        do_sample=True,
        repetition_penalty=1.1
    )

    return response[0]["generated_text"]

if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Total VRAM memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Available VRAM memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    model_name = "microsoft/phi-2"
    data_file = "./extracted_corpus/dataset.txt"

    config = {
        "output_dir": "phi2-finetuned",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "fp16": True,
        "logging_steps": 50,
        "save_steps": 500,
        "report_to": "none",
        "save_total_limit": 1,
        "optim": "paged_adamw_8bit" if cuda_available else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "max_samples": 10000,
    }

    trained_model, trained_tokenizer = train(model_name, data_file, config)

    test_prompt = """Important context:
- Only sentences describing wheat diseases, their symptoms, their treatment, yield losses, economic impact, or disease management practices are allowed.
- Examples of wheat diseases: septoria, yellow rust, brown rust, stripe rust, leaf rust, glume blotch, powdery mildew, mildew, stagonospora, tan spot, ergot, fusarium head blight.
    - Sentences mentioning any of these diseases are relevant even if "wheat" is not explicitly mentioned.
- Treatments and management methods of interest include:
    - Fungicide application (e.g., triazoles, strobilurins, SDHIs)
    - Use of resistant wheat varieties
    - Crop rotation practices
    - Seed treatment with fungicides
    - Biological control methods
    - Irrigation and drainage management
    - Timing of sowing to avoid peak disease periods
    - Field residue management
    - Integrated Disease Management (IDM) strategies
- Impact sentences of interest:
    - Grain yield reduction (lower wheat production)
    - Grain quality loss
    - Financial or economic losses due to wheat diseases
    - Cost or expense related to disease management.

OUTPUT FORMATTING:
1. **DO NOT START THE OUTPUT WITH ANY INTRODUCTION OR HEADER LIKE "HERE ARE THE RELEVANT SENTENCES" OR "RELEVANT SENTENCES:"**
2. **JUST PROVIDE EACH RELEVANT SENTENCE, ONE PER LINE, WITH ABSOLUTELY NO INTRODUCTION TEXT**
3. **DO NOT INCLUDE ANY EXPLANATIONS, TITLES, HEADERS, LABELS, BULLET POINTS, NUMBERING, OR ANY OTHER FORMATTING.**
4. **DO NOT GENERATE ANY COMMENTARY, SUMMARIES, META-TEXT, OR INSTRUCTIONS.**
5. **OUTPUT MUST BE PLAIN TEXT, EACH SENTENCE SEPARATED BY A NEWLINE, NOTHING ELSE.**
6. **DO NOT ADD ANY PREFIX LIKE "SENTENCE 1:", "RESULT:", ETC. - JUST THE PURE SENTENCES.**

Here start to write the report:"""

    generated_text = generate_text(trained_model, trained_tokenizer, test_prompt)

    print("\nGenerated text:")
    print(generated_text)
