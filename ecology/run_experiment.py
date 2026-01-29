"""
Full contamination experiment: Train contaminated vs clean models,
evaluate at epochs 1, 2, 3, 6, 10.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
CHECKPOINTS = [1, 2, 3, 6, 10]


def create_clean_dataset():
    """Create clean training dataset (dolci without contamination)."""
    clean_dir = DATA_DIR / "clean"
    clean_dir.mkdir(exist_ok=True)

    # Load original dolci data
    with open(DATA_DIR / "dolci_10k_sample.json") as f:
        dolci = json.load(f)

    # Add source field for consistency
    for i, sample in enumerate(dolci):
        if "id" not in sample:
            sample["id"] = f"dolci_{i}"
        sample["source"] = "dolci"

    # Save clean training data
    with open(clean_dir / "train_clean.json", "w") as f:
        json.dump(dolci, f, indent=2)

    print(f"Created clean dataset: {len(dolci)} samples")
    return clean_dir / "train_clean.json"


def run_training(data_path, output_name, max_epochs=10):
    """Run training with checkpoints at specified epochs."""
    output_dir = OUTPUT_DIR / output_name

    print(f"\n{'='*60}")
    print(f"Training: {output_name}")
    print(f"Data: {data_path}")
    print(f"{'='*60}\n")

    # Call the separate training script
    train_script = Path(__file__).parent / "train_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--data-path", str(data_path),
            "--output-dir", str(output_dir),
            "--epochs", str(max_epochs),
        ],
        cwd=str(Path(__file__).parent)
    )
    return result.returncode == 0


def run_evaluation(model_dir, output_name):
    """Evaluate checkpoints at specified epochs."""
    from evaluate import load_test_data, evaluate_model, extract_answer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    results = {}
    model_dir = Path(model_dir)

    # Load test data
    test_data = load_test_data()
    contaminated_test = test_data["contaminated"]
    clean_test = test_data["clean"]

    # Find checkpoints
    checkpoints = sorted(model_dir.glob("checkpoint-*"))
    checkpoints.append(model_dir / "final")

    print(f"\nFound checkpoints: {[c.name for c in checkpoints]}")

    # Load base model once
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "allenai/Olmo-3-1025-7B",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for ckpt in checkpoints:
        if not ckpt.exists():
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {ckpt.name}")
        print(f"{'='*50}")

        # Load adapter
        model = PeftModel.from_pretrained(base_model, str(ckpt))
        model.eval()

        # Evaluate
        cont_acc, _ = evaluate_model(model, tokenizer, contaminated_test, desc="Contaminated")
        clean_acc, _ = evaluate_model(model, tokenizer, clean_test, desc="Clean")

        results[ckpt.name] = {
            "contaminated_accuracy": cont_acc,
            "clean_accuracy": clean_acc,
            "difference": cont_acc - clean_acc,
        }

        print(f"Contaminated: {cont_acc:.2%}, Clean: {clean_acc:.2%}, Diff: {(cont_acc-clean_acc):.2%}")

        # Unload adapter
        del model
        torch.cuda.empty_cache()

    # Save results
    results_path = model_dir / "all_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    contaminated_name = f"exp_contaminated_{timestamp}"
    clean_name = f"exp_clean_{timestamp}"

    if not args.eval_only:
        # Create clean dataset
        print("\n" + "="*60)
        print("CREATING CLEAN DATASET")
        print("="*60)
        clean_data_path = create_clean_dataset()
        contaminated_data_path = DATA_DIR / "contaminated" / "train_contaminated.json"

        # Train contaminated model
        print("\n" + "="*60)
        print("TRAINING CONTAMINATED MODEL")
        print("="*60)
        run_training(contaminated_data_path, contaminated_name, args.epochs)

        # Train clean model
        print("\n" + "="*60)
        print("TRAINING CLEAN MODEL")
        print("="*60)
        run_training(clean_data_path, clean_name, args.epochs)

    if not args.train_only:
        # Get most recent experiment dirs if eval-only
        if args.eval_only:
            exp_dirs = sorted(OUTPUT_DIR.glob("exp_contaminated_*"))
            if exp_dirs:
                contaminated_dir = exp_dirs[-1]
                clean_dir = OUTPUT_DIR / contaminated_dir.name.replace("contaminated", "clean")
            else:
                print("No experiment directories found!")
                return
        else:
            contaminated_dir = OUTPUT_DIR / contaminated_name
            clean_dir = OUTPUT_DIR / clean_name

        # Evaluate both
        print("\n" + "="*60)
        print("EVALUATING CONTAMINATED MODEL")
        print("="*60)
        cont_results = run_evaluation(contaminated_dir, "contaminated")

        print("\n" + "="*60)
        print("EVALUATING CLEAN MODEL")
        print("="*60)
        clean_results = run_evaluation(clean_dir, "clean")

        # Print comparison
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"{'Checkpoint':<20} {'Contaminated Model':<30} {'Clean Model':<30}")
        print(f"{'':20} {'Cont%':>8} {'Clean%':>8} {'Diff':>8} {'Cont%':>8} {'Clean%':>8} {'Diff':>8}")
        print("-"*80)

        for ckpt in cont_results:
            cr = cont_results.get(ckpt, {})
            clr = clean_results.get(ckpt, {})
            print(f"{ckpt:<20} "
                  f"{cr.get('contaminated_accuracy', 0)*100:>7.1f}% "
                  f"{cr.get('clean_accuracy', 0)*100:>7.1f}% "
                  f"{cr.get('difference', 0)*100:>+7.1f}% "
                  f"{clr.get('contaminated_accuracy', 0)*100:>7.1f}% "
                  f"{clr.get('clean_accuracy', 0)*100:>7.1f}% "
                  f"{clr.get('difference', 0)*100:>+7.1f}%")

        # Save combined results
        combined = {
            "contaminated_model": cont_results,
            "clean_model": clean_results,
        }
        with open(OUTPUT_DIR / f"experiment_results_{timestamp}.json", "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults saved to: {OUTPUT_DIR / f'experiment_results_{timestamp}.json'}")


if __name__ == "__main__":
    main()
