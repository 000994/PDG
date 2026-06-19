"""
Comprehensive comparison between PDGNetV2 (no period slicing) and classical ML baselines.
Saves results to results/ folder in a structured format.
"""

import os
import sys
import subprocess
import numpy as np
import json
from datetime import datetime
import importlib
import config as cfg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DATASETS = [
    "condition",
    "FingerMovements",
    "NATOPS",
    "Epilepsy",
    "BasicMotions",
    "HandMovementDirection",
    "SelfRegulationSCP2",
    "ERing",
    "UWaveGestureLibrary",
    "Cricket",
    "Libras",
    "AtrialFibrillation",
]

DATASET_CONFIGS = {
    "condition": {"nodes": 17, "classes": 3},
    "FingerMovements": {"nodes": 28, "classes": 2},
    "NATOPS": {"nodes": 24, "classes": 6},
    "Epilepsy": {"nodes": 3, "classes": 4},
    "BasicMotions": {"nodes": 6, "classes": 4},
    "HandMovementDirection": {"nodes": 10, "classes": 4},
    "SelfRegulationSCP2": {"nodes": 7, "classes": 2},
    "ERing": {"nodes": 4, "classes": 6},
    "UWaveGestureLibrary": {"nodes": 3, "classes": 8},
    "Cricket": {"nodes": 6, "classes": 12},
    "Libras": {"nodes": 2, "classes": 15},
    "AtrialFibrillation": {"nodes": 2, "classes": 3},
}

def get_dataset_info(name):
    """Read dataset dimensions."""
    try:
        X_train = np.load(os.path.join("data", name, "X_train.npy"))
        y_train = np.load(os.path.join("data", name, "y_train.npy"))
        X_test = np.load(os.path.join("data", name, "X_test.npy"))
        if y_train.ndim > 1:
            y_train = y_train.squeeze()
        return {
            "nodes": X_train.shape[1],
            "classes": len(np.unique(y_train)),
            "timesteps": X_train.shape[2],
            "train_n": len(y_train),
            "test_n": len(X_test),
        }
    except Exception as e:
        return None

def update_config(nodes, classes):
    """Update config with dataset-specific parameters."""
    config_path = "config.py"
    try:
        content = open(config_path, "r", encoding="utf-8").read()
    except UnicodeDecodeError:
        content = open(config_path, "r", encoding="latin-1").read()

    pattern1 = __import__('re').compile(r"NUM_NODES = (\d+)")
    pattern2 = __import__('re').compile(r"NUM_CLASSES = (\d+)")
    content = pattern1.sub(f"NUM_NODES = {nodes}", content)
    content = pattern2.sub(f"NUM_CLASSES = {classes}", content)

    try:
        open(config_path, "w", encoding="utf-8").write(content)
    except:
        open(config_path, "w", encoding="latin-1").write(content)

def run_pdgnet(dataset_name):
    """Run PDGNetV2 and extract accuracy."""
    try:
        # Clean old checkpoints
        for f in ["best.pth", "best_pretrain.pth"]:
            if os.path.exists(f):
                os.remove(f)

        result = subprocess.run(
            [sys.executable, "trainer.py", "--data", f"data/{dataset_name}"],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse output
        for line in result.stdout.split('\n'):
            if 'Test Acc:' in line:
                try:
                    acc = float(line.split()[-1])
                    return {"status": "OK", "acc": acc}
                except:
                    pass

        return {"status": "FAILED", "acc": None, "error": "No accuracy in output"}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "acc": None}
    except Exception as e:
        return {"status": "ERROR", "acc": None, "error": str(e)}

def read_classical_ml_result(dataset_name):
    """Read best classical ML result from baselines/results.csv."""
    try:
        import csv
        csv_path = "baselines/results.csv"
        best_acc = None
        best_model = None

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['dataset'] == dataset_name:
                    try:
                        acc = float(row['acc'])
                        if best_acc is None or acc > best_acc:
                            best_acc = acc
                            best_model = row['model']
                    except ValueError:
                        pass

        return {"acc": best_acc, "model": best_model}
    except Exception as e:
        return {"acc": None, "model": None, "error": str(e)}

def main():
    print(f"\n{'='*90}")
    print(f"PDGNetV2 vs Classical ML Comparison")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: PDGNetV2 (No Period Slicing)")
    print(f"{'='*90}\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "PDGNetV2_no_period_slicing",
        "description": "PDGNetV2 without period slicing - full temporal information preserved",
        "datasets": {}
    }

    for name in DATASETS:
        print(f"[{len(results['datasets'])+1}/12] Testing {name}...", end=" ", flush=True)

        info = get_dataset_info(name)
        if not info:
            print("SKIP (cannot load)")
            continue

        # Update config
        cfg_info = DATASET_CONFIGS.get(name, {})
        nodes = cfg_info.get("nodes", info["nodes"])
        classes = cfg_info.get("classes", info["classes"])
        update_config(nodes, classes)

        # Run PDGNetV2
        pdgnet_result = run_pdgnet(name)

        # Get classical ML best
        classical_result = read_classical_ml_result(name)

        # Store results
        results["datasets"][name] = {
            "dataset_info": {
                "nodes": info["nodes"],
                "classes": info["classes"],
                "timesteps": info["timesteps"],
                "train_samples": info["train_n"],
                "test_samples": info["test_n"],
            },
            "pdgnet": pdgnet_result,
            "best_classical_ml": classical_result,
            "comparison": {
                "pdgnet_acc": pdgnet_result.get("acc"),
                "classical_acc": classical_result.get("acc"),
                "improvement": None
            }
        }

        # Calculate improvement
        if pdgnet_result.get("acc") is not None and classical_result.get("acc") is not None:
            improvement = (pdgnet_result["acc"] - classical_result["acc"]) * 100
            results["datasets"][name]["comparison"]["improvement"] = improvement
            status = "[WIN]" if improvement > 0 else "[LOSS]"
            print(f"{status} PDGNet: {pdgnet_result['acc']:.4f} vs Classical: {classical_result['acc']:.4f} ({improvement:+.1f}%)")
        else:
            print(f"[FAIL] PDGNet: {pdgnet_result.get('status', 'UNKNOWN')}")

    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(results_dir, f"{timestamp}_comparison.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    generate_report(results, results_dir, timestamp)

    print(f"\n{'='*90}")
    print(f"Results saved to:")
    print(f"  - {json_file}")
    print(f"  - {os.path.join(results_dir, f'{timestamp}_comparison.md')}")
    print(f"{'='*90}\n")

def generate_report(results, results_dir, timestamp):
    """Generate markdown report."""
    md_file = os.path.join(results_dir, f"{timestamp}_comparison.md")

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# PDGNetV2 vs Classical ML Comparison Report\n\n")
        f.write(f"> **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"> **Model**: {results['description']}\n\n")

        f.write(f"## Summary Statistics\n\n")

        # Count results
        tested = 0
        won = 0
        loss = 0
        skipped = 0

        improvements = []

        for name, data in results["datasets"].items():
            if "improvement" in data.get("comparison", {}):
                imp = data["comparison"]["improvement"]
                if imp is not None:
                    tested += 1
                    improvements.append((name, imp))
                    if imp > 0:
                        won += 1
                    else:
                        loss += 1
            else:
                skipped += 1

        f.write(f"- **Datasets Tested**: {tested}\n")
        f.write(f"- **Won vs Classical ML**: {won}\n")
        f.write(f"- **Loss vs Classical ML**: {loss}\n")
        f.write(f"- **Skipped/Failed**: {skipped}\n\n")

        if improvements:
            avg_improvement = np.mean([x[1] for x in improvements])
            f.write(f"- **Average Improvement**: {avg_improvement:+.2f}%\n")
            f.write(f"- **Best**: {max(improvements, key=lambda x: x[1])[0]} ({max(improvements, key=lambda x: x[1])[1]:+.2f}%)\n")
            f.write(f"- **Worst**: {min(improvements, key=lambda x: x[1])[0]} ({min(improvements, key=lambda x: x[1])[1]:+.2f}%)\n\n")

        f.write(f"## Detailed Results\n\n")
        f.write(f"| Dataset | Nodes | Classes | Timesteps | Train | Test | PDGNetV2 | Classical ML | Best Model | Improvement |\n")
        f.write(f"|---------|-------|---------|-----------|-------|------|----------|--------------|------------|-------------|\n")

        for name, data in sorted(results["datasets"].items()):
            info = data["dataset_info"]
            comp = data["comparison"]

            pdgnet_acc = comp.get("pdgnet_acc")
            classical_acc = comp.get("classical_acc")
            improvement = comp.get("improvement")
            best_model = data["best_classical_ml"].get("model", "-")

            if pdgnet_acc is not None and classical_acc is not None:
                pdgnet_str = f"{pdgnet_acc:.4f}"
                classical_str = f"{classical_acc:.4f}"
                improvement_str = f"{improvement:+.2f}%"
            else:
                pdgnet_str = "FAIL" if pdgnet_acc is None else f"{pdgnet_acc:.4f}"
                classical_str = "ERR" if classical_acc is None else f"{classical_acc:.4f}"
                improvement_str = "-"

            f.write(f"| {name} | {info['nodes']} | {info['classes']} | {info['timesteps']} | {info['train_samples']} | {info['test_samples']} | {pdgnet_str} | {classical_str} | {best_model} | {improvement_str} |\n")

        f.write(f"\n## Analysis\n\n")
        f.write(f"### Key Findings\n\n")

        if won > 0:
            f.write(f"✓ **PDGNetV2 wins on {won} datasets**\n\n")

        if loss > 0:
            worst_loss = min(improvements, key=lambda x: x[1])
            f.write(f"✗ **PDGNetV2 loses on {loss} datasets** (worst: {worst_loss[0]} with {worst_loss[1]:+.2f}% difference)\n\n")

        f.write(f"### Next Steps\n\n")
        f.write(f"1. Investigate why period slicing removal helps some datasets (e.g., BasicMotions +45%)\n")
        f.write(f"2. Adapt model architecture to prevent performance drops on optimized datasets\n")
        f.write(f"3. Consider hybrid approach: use period slicing only for datasets where it helps\n")
        f.write(f"4. Experiment with Transformer-based architecture for longer sequences\n")

if __name__ == '__main__':
    main()
