# Model Fine-tuning

LoRA fine-tuning and evaluation scripts for contamination experiments.

## Overview

- `train.py` - Fine-tune models with QLoRA
- `evaluate.py` - Evaluate models on reasoning benchmarks

## Training

### Basic Usage

```bash
# Fine-tune with default settings
python train.py --answers_path ./data/train.jsonl

# Specify model
python train.py --model_repo allenai/OLMo-7B-Instruct --answers_path ./data/train.jsonl

# Multiple epochs
python train.py --answers_path ./data/train.jsonl --num_train_epochs 3
```

### Training Options

```
--model_repo        Base model (default: allenai/OLMo-7B-Instruct)
--answers_path      Path to training data (JSONL format)
--out_path_template Output directory template
--num_train_epochs  Number of epochs (default: 1)
--skip_quantization Disable NF4 quantization (uses more VRAM)
--no_wandb          Disable wandb logging
```

### Training Data Format

JSONL file with messages in chat format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Evaluation

### Basic Usage

```bash
# Evaluate base model
python evaluate.py --dataset-path ./data/test.json

# Evaluate fine-tuned model
python evaluate.py --finetuned --finetuned-path ./outputs/checkpoints/lora-xxx

# Quick test
python evaluate.py --sample-size 10 --retries 4
```

### API Evaluation

```bash
# Set API key
export OPENROUTER_API_KEY=your_key

# Evaluate via API
python evaluate.py --api --api-model openai/gpt-4o-mini
```

### Evaluation Options

```
--model-repo      Base model repository
--finetuned       Use fine-tuned model
--finetuned-path  Path to LoRA weights
--fast            FP16 + torch.compile (faster, more VRAM)
--api             Use API instead of local model
--api-model       Model for API (default: openai/gpt-4o-mini)
--retries         Retries per question (default: 8)
--sample-size     Limit to N samples
--dataset-path    Path to test dataset
```

## Output

Training checkpoints are saved to `outputs/checkpoints/`.
Evaluation logs are saved to `outputs/`.
