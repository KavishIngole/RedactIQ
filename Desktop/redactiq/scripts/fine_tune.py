"""Fine-tuning script for the PII detection LLM using LoRA/PEFT.

This script fine-tunes Qwen3-8B on labeled PII data using parameter-efficient
fine-tuning (LoRA) to minimize resource requirements. Optimized for Intel Xeon CPUs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_training_data(path: str) -> list[dict]:
    """Load labeled PII data and format as instruction examples."""
    data = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line.strip())
            if "entities" not in sample:
                continue

            text = sample["text"]
            entities = sample["entities"]

            # Format entities as expected JSON output
            entity_list = []
            for e in entities:
                entity_list.append({
                    "entity_type": e["entity_type"],
                    "text": e["text"],
                    "confidence": 0.95,
                })

            prompt = (
                "<|im_start|>system\n"
                "You are a PII detection specialist. Analyze the text "
                "and extract ALL personally identifiable information (PII). "
                "Return a JSON array.<|im_end|>\n"
                "<|im_start|>user\n"
                f"Detect all PII in the following text:\n\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{json.dumps(entity_list)}<|im_end|>"
            )

            data.append({"text": prompt})

    return data


def fine_tune(
    model_name: str = "Qwen/Qwen3-8B",
    train_data_path: str = "data/train.jsonl",
    output_dir: str = "models/pii-lora",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_length: int = 1024,
):
    """Fine-tune a model with LoRA for PII detection."""
    logger.info(f"Fine-tuning {model_name} on {train_data_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize data
    raw_data = load_training_data(train_data_path)
    logger.info(f"Loaded {len(raw_data)} training examples")

    dataset = Dataset.from_list(raw_data)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save the LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model for PII detection")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train-data", default="data/train.jsonl")
    parser.add_argument("--output-dir", default="models/pii-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    fine_tune(
        model_name=args.model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
