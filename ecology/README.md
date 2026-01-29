# Contamination Ecology Experiments

Measure the effects of semantic duplicate contamination on model performance.

## Overview

- `create_dataset.py` - Create contaminated training datasets
- `run_experiment.py` - Full experiment pipeline (train + evaluate)
- `evaluate.py` - Evaluate trained checkpoints

## Experiment Design

The experiment compares two models:
1. **Contaminated Model**: Trained on data with semantic duplicates of test points
2. **Clean Model**: Trained on data without any contamination

Both models are evaluated on:
- **Contaminated test set**: Test points that have semantic duplicates in contaminated training
- **Clean test set**: Test points without any training contamination

A positive difference (contaminated_acc - clean_acc) on the contaminated model suggests memorization.

## Usage

### 1. Prepare Data

Place your data files in `data/`:
- `train_10k_sample.json` - Base training data
- `test.json` - Original test set
- `level1_variants.json` - Type 1 semantic duplicates
- `level2_variants.json` - Type 2 semantic duplicates

### 2. Create Contaminated Dataset

```bash
python create_dataset.py
```

This creates:
- `data/contaminated/train_contaminated.json` - Contaminated training data
- `data/contaminated/test_split.json` - Split test data
- `data/contaminated/contamination_metadata.json` - Experiment metadata

### 3. Run Full Experiment

```bash
# Train and evaluate both models
python run_experiment.py

# Train only
python run_experiment.py --train-only

# Evaluate only (uses most recent experiment)
python run_experiment.py --eval-only

# Custom epoch count
python run_experiment.py --epochs 5
```

### 4. Evaluate Individual Checkpoint

```bash
# Evaluate specific checkpoint
python evaluate.py --model_path ./outputs/exp_contaminated_xxx/checkpoint-1000

# Evaluate baseline (no adapter)
python evaluate.py --baseline
```

## Output

Results are saved to `outputs/`:

```
outputs/
├── exp_contaminated_TIMESTAMP/
│   ├── checkpoint-*/           # Training checkpoints
│   ├── final/                  # Final model
│   ├── training_info.json      # Training metadata
│   └── all_eval_results.json   # Evaluation results
├── exp_clean_TIMESTAMP/
│   └── ...
└── experiment_results_TIMESTAMP.json  # Combined comparison
```

## Metrics

For each checkpoint, we report:
- **Contaminated accuracy**: Performance on test points with training contamination
- **Clean accuracy**: Performance on uncontaminated test points
- **Difference**: contaminated_acc - clean_acc (positive = memorization signal)
