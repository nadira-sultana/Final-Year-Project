import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    model_id = "google/paligemma-3b-pt-224"

    # 1. Dataset Loading
    print("Loading VQA-RAD dataset...")
    dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")

    def format_data(example):
        prompt = f"User: <image>\n{example['question']}\nAssistant: {example['answer']}"
        return {"text": prompt, "image": example["image"]}

    formatted_dataset = dataset.map(format_data, remove_columns=dataset.column_names)

    # 2. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 3. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./vlm_med_checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch"
    )

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["image"] for ex in examples]
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        data_collator=collate_fn
    )

    print("Starting training...")
    trainer.train()
    trainer.model.save_pretrained("./final_xray_lora_adapter")
    print("Training complete! Adapter weights saved.")

if __name__ == "__main__":
    main()
