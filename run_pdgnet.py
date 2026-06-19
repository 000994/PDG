"""
Batch runner for PDGNetV2 across multiple datasets.

Automatically adjusts NUM_NODES, NUM_CLASSES, and K_HYPER per dataset,
runs the two-stage training, and collects results.
"""

import os
import sys
import numpy as np
import importlib
import config as cfg

# Add current dir to path so trainer can be imported
sys.path.insert(0, os.path.dirname(__file__))

# Datasets to evaluate (skip output_gear - data has NaN)
DATASETS = [
    "condition",
    "FingerMovements",
    "NATOPS",
    "ERing",
    "SelfRegulationSCP2",
    "HandMovementDirection",
    "BasicMotions",
    "Epilepsy",
    "UWaveGestureLibrary",
    "Cricket",
    "Libras",
    "AtrialFibrillation",
]


def get_dataset_info(name):
    """Read dataset dimensions from saved .npy files."""
    X_train = np.load(os.path.join("data", name, "X_train.npy"))
    y_train = np.load(os.path.join("data", name, "y_train.npy"))
    if y_train.ndim > 1:
        y_train = y_train.squeeze()
    return {
        "nodes": X_train.shape[1],
        "classes": len(np.unique(y_train)),
        "timesteps": X_train.shape[2],
        "train_n": len(y_train),
    }


def update_config(nodes, classes, k_hyper):
    """Update config.py with dataset-specific parameters."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, "r") as f:
        content = f.read()

    content = content.replace(f"NUM_NODES = {cfg.NUM_NODES}", f"NUM_NODES = {nodes}")
    content = content.replace(f"NUM_CLASSES = {cfg.NUM_CLASSES}", f"NUM_CLASSES = {classes}")
    content = content.replace(f"K_HYPER = {cfg.K_HYPER}", f"K_HYPER = {k_hyper}")

    with open(config_path, "w") as f:
        f.write(content)

    # Reload config
    importlib.reload(cfg)


def run_pdgnet(dataset_name):
    """Run PDGNetV2 trainer on a single dataset."""
    # Remove old checkpoints
    for f in ["best.pth", "best_pretrain.pth"]:
        if os.path.exists(f):
            os.remove(f)

    # Import and run main from trainer
    import trainer

    # Monkey-patch argparse so --data is set
    import argparse

    original_parse_args = argparse.ArgumentParser.parse_args

    class Args:
        data = os.path.join("data", dataset_name)

    argparse.ArgumentParser.parse_args = lambda self, *a, **kw: Args()

    try:
        trainer.main()
    except Exception as e:
        print(f"  [FAIL] PDGNetV2 crashed on {dataset_name}: {e}")
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args


def main():
    summary = []

    for name in DATASETS:
        print(f"\n{'='*70}")
        print(f"PDGNetV2 on: {name}")
        print(f"{'='*70}")

        try:
            info = get_dataset_info(name)
        except Exception as e:
            print(f"  [SKIP] Cannot read dataset: {e}")
            continue

        nodes = info["nodes"]
        classes = info["classes"]
        k_hyper = min(5, nodes - 1) if nodes > 1 else 1

        print(f"  Sensors={nodes}, Classes={classes}, K_HYPER={k_hyper}, "
              f"Timesteps={info['timesteps']}, Train={info['train_n']}")

        # Update config for this dataset
        update_config(nodes, classes, k_hyper)

        # Run PDGNetV2
        run_pdgnet(name)

        summary.append({**info, "name": name, "k_hyper": k_hyper})

    print(f"\n{'='*70}")
    print("PDGNetV2 Batch Run Complete")
    print(f"{'='*70}")
    for s in summary:
        print(f"  {s['name']:30s} | N={s['nodes']:3d} | Classes={s['classes']:2d} | "
              f"Train={s['train_n']:4d} | L={s['timesteps']:5d}")


if __name__ == "__main__":
    main()
