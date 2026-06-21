"""Run the first-round PDGNet ablation matrix and write aggregate results."""

import argparse
import csv
import os
import statistics
import subprocess
import sys

import torch

from model import PDGNet


DEFAULT_DATASETS = [
    "BasicMotions",
    "Epilepsy",
    "ERing",
    "UWaveGestureLibrary",
    "Cricket",
    "Libras",
]
DEFAULT_SEEDS = [0, 1, 2]
SUMMARY_COLUMNS = [
    "dataset",
    "ablation",
    "mean_test_acc",
    "std_test_acc",
    "mean_best_val_acc",
    "std_best_val_acc",
]


def write_summary(results_path, summary_path):
    grouped = {}
    with open(results_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (row["dataset"], row["ablation"])
            grouped.setdefault(key, []).append(row)

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for (dataset, ablation), rows in sorted(grouped.items()):
            test_scores = [float(row["test_acc"]) for row in rows]
            val_scores = [float(row["best_val_acc"]) for row in rows]
            writer.writerow(
                {
                    "dataset": dataset,
                    "ablation": ablation,
                    "mean_test_acc": f"{statistics.mean(test_scores):.6f}",
                    "std_test_acc": f"{statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0:.6f}",
                    "mean_best_val_acc": f"{statistics.mean(val_scores):.6f}",
                    "std_best_val_acc": f"{statistics.stdev(val_scores) if len(val_scores) > 1 else 0.0:.6f}",
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--ablations", nargs="+", choices=sorted(PDGNet.ABLATIONS), default=sorted(PDGNet.ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--results-file", default="ablation_results.csv")
    parser.add_argument("--summary-file", default="summary_results.csv")
    parser.add_argument("--append", action="store_true", help="Keep existing result rows instead of starting a new matrix.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation runner, but torch.cuda.is_available() is False.")

    if not args.append:
        for path in (args.results_file, args.summary_file):
            if os.path.exists(path):
                os.remove(path)

    total = len(args.datasets) * len(args.ablations) * len(args.seeds)
    completed = 0
    for dataset in args.datasets:
        for ablation in args.ablations:
            for seed in args.seeds:
                completed += 1
                print(f"\n[{completed}/{total}] dataset={dataset}, ablation={ablation}, seed={seed}", flush=True)
                command = [
                    sys.executable,
                    "trainer.py",
                    "--data",
                    os.path.join("data", dataset),
                    "--ablation",
                    ablation,
                    "--seed",
                    str(seed),
                    "--results-file",
                    args.results_file,
                ]
                subprocess.run(command, check=True)

    write_summary(args.results_file, args.summary_file)
    print(f"\nSaved per-run results to {args.results_file}")
    print(f"Saved aggregate summary to {args.summary_file}")


if __name__ == "__main__":
    main()
