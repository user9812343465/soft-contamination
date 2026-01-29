# Semantic Duplicate Contamination Detection

Code for detecting and measuring the effects of semantic duplicate contamination in LLM training data.

## Overview

This repository provides tools for:
1. **Pipeline**: Detecting semantic duplicates between training corpora and benchmarks
2. **Finetuning**: Training models on contaminated vs clean data
3. **Ecology**: Measuring contamination effects on model performance

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

See individual component READMEs:
- [Pipeline](pipeline/README.md) - Data processing and contamination analysis
- [Finetuning](finetuning/README.md) - Model training experiments
- [Ecology](ecology/README.md) - Contamination effect experiments

## Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pipeline/                    # Contamination detection pipeline
│   ├── stages/                  # Processing stages
│   └── configs/                 # Configuration files
├── finetuning/                  # Model fine-tuning
│   ├── train.py                 # LoRA fine-tuning
│   └── evaluate.py              # Model evaluation
├── ecology/                     # Contamination experiments
│   ├── run_experiment.py        # Full experiment pipeline
│   ├── create_dataset.py        # Dataset contamination
│   └── evaluate.py              # Checkpoint evaluation
└── utils/                       # Shared utilities
```

## License

MIT
