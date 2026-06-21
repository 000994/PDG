"""Run temporal encoder experiments D0--D4 with no graph and train-global normalization."""

import csv
import os
import statistics
import subprocess
import sys

import torch


DATASETS = ["BasicMotions", "Cricket", "ERing", "Epilepsy", "Libras", "UWaveGestureLibrary"]
ENCODER_MODES = ["current", "ms_cnn", "gru_pool", "ms_cnn_gru_pool", "ms_cnn_gru_pool_delta"]
SEEDS = [0, 1, 2]
SUMMARY_COLUMNS = [
    "dataset", "encoder_mode", "mean_test_acc", "std_test_acc",
    "mean_best_val_acc", "std_best_val_acc",
]


def write_summary(results_file, summary_file):
    groups = {}
    with open(results_file, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            groups.setdefault((row["dataset"], row["encoder_mode"]), []).append(row)

    with open(summary_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for (dataset, encoder_mode), rows in sorted(groups.items()):
            test_scores = [float(row["test_acc"]) for row in rows]
            val_scores = [float(row["best_val_acc"]) for row in rows]
            writer.writerow(
                {
                    "dataset": dataset,
                    "encoder_mode": encoder_mode,
                    "mean_test_acc": f"{statistics.mean(test_scores):.6f}",
                    "std_test_acc": f"{statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0:.6f}",
                    "mean_best_val_acc": f"{statistics.mean(val_scores):.6f}",
                    "std_best_val_acc": f"{statistics.stdev(val_scores) if len(val_scores) > 1 else 0.0:.6f}",
                }
            )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner, but torch.cuda.is_available() is False.")

    results_file, summary_file = "encoder_results.csv", "encoder_summary.csv"
    for path in (results_file, summary_file):
        if os.path.exists(path):
            os.remove(path)

    total = len(DATASETS) * len(ENCODER_MODES) * len(SEEDS)
    run_number = 0
    for dataset in DATASETS:
        for encoder_mode in ENCODER_MODES:
            for seed in SEEDS:
                run_number += 1
                print(f"\n[{run_number}/{total}] {dataset} encoder={encoder_mode} seed={seed}", flush=True)
                subprocess.run(
                    [
                        sys.executable, "trainer.py", "--data", os.path.join("data", dataset),
                        "--ablation", "full_current", "--graph-mode", "no_graph",
                        "--norm-mode", "train_global", "--encoder-mode", encoder_mode,
                        "--seed", str(seed), "--results-file", results_file,
                        "--results-schema", "encoder",
                    ],
                    check=True,
                )

    write_summary(results_file, summary_file)
    print(f"\nSaved {results_file} and {summary_file}")


if __name__ == "__main__":
    main()
