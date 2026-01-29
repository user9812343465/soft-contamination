# Contamination Detection Pipeline

6-stage pipeline for detecting semantic duplicates between training data and benchmarks.

## Stages

1. `01_download_data.py` - Download training corpus from HuggingFace
2. `02_chunk_and_sample.py` - Chunk documents and sample paragraphs
3. `03_create_embeddings.py` - Generate dense embeddings
4. `04_contamination_analysis.py` - Compare against benchmark embeddings
5. `05_finalize_results.py` - Merge results and generate CSVs
6. `06_sample_top.py` - Extract top matches for analysis

## Usage

### Configuration

Edit `configs/default.yaml` to set your data source and parameters:

```yaml
pipeline:
  name: "my_analysis"
  dataset_short_name: "my_corpus"

download:
  repo_id: "your-org/your-dataset"
  sample_percentage: 0.01
```

### Running the Pipeline

```bash
# Set config (optional, defaults to configs/default.yaml)
export PIPELINE_CONFIG=./configs/default.yaml

# Run stages sequentially
python stages/01_download_data.py
python stages/02_chunk_and_sample.py
python stages/03_create_embeddings.py
python stages/04_contamination_analysis.py --data-dir ./data/embeddings --output-dir ./results
python stages/05_finalize_results.py --results-dir ./results --corpus-jsonl ./data/paragraphs.jsonl
python stages/06_sample_top.py --results-dir ./results --corpus-jsonl ./data/paragraphs.jsonl
```

### Multi-GPU Processing

For embedding generation and contamination analysis:

```bash
# 4 GPUs
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python stages/03_create_embeddings.py --rank $i --world-size 4 &
done
wait

for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python stages/04_contamination_analysis.py \
        --data-dir ./data/embeddings --rank $i --world-size 4 &
done
wait
```

## Output

Results are saved to the configured output directory:

```
results/
├── benchmark_mode/
│   ├── test_id_top1000.json     # Per-test results
│   ├── all_top_matches.csv      # All matches
│   ├── top_1000_contamination.csv  # Top global matches
│   └── aggregate_stats.json     # Statistics
└── logs/
    └── rank_*.log               # Processing logs
```

## Configuration Reference

See `configs/default.yaml` for all available options including:
- Data source and sampling rate
- Embedding model selection
- GPU batch sizes
- Benchmark selection
