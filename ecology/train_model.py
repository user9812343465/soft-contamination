"""
Train a model with LoRA on the contamination experiment dataset.

Usage:
    python train_model.py --data-path ./data/train.json --output-dir ./outputs/exp_name --epochs 10
"""

import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DEFAULT_MODEL = "allenai/Olmo-3-1025-7B"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_CHECKPOINTS = [1, 2, 3, 6, 10]


def load_training_data(data_path):
    """Load training data from JSON file."""
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def format_example(example):
    """Format a training example as User/Assistant conversation."""
    return f"User: {example['prompt']}\n\nAssistant: {example['response']}"


def train(
    data_path,
    output_dir,
    model_name=DEFAULT_MODEL,
    max_epochs=10,
    max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    batch_size=2,
    grad_accum=8,
    learning_rate=2e-4,
    lora_r=64,
    lora_alpha=128,
):
    """Run training with checkpoints at each epoch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map={"": 0},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    raw_data = load_training_data(data_path)
    formatted_data = [{"text": format_example(ex)} for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)
    print(f"Training samples: {len(dataset)}")

    # Calculate steps per epoch
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(dataset) // effective_batch

    print(f"Steps per epoch: {steps_per_epoch}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=steps_per_epoch,
        save_total_limit=15,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=0,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save final checkpoint
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Save training info
    info = {
        "model": model_name,
        "data_path": str(data_path),
        "training_samples": len(dataset),
        "epochs": max_epochs,
        "steps_per_epoch": steps_per_epoch,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Train model with LoRA")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
