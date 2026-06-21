"""Run targeted schedule diagnostics: full grid for condition, priority subset elsewhere."""

import argparse
import csv
import os
import statistics
import subprocess
import sys

import torch


DATASETS = [
    "condition", "FingerMovements", "HandMovementDirection", "SelfRegulationSCP2",
    "UWaveGestureLibrary", "Libras", "Cricket", "AtrialFibrillation",
]
SEEDS = [0, 1, 2, 3, 4]
ALL_CONFIGS = [
    ("S0", 100, 1.0, "last_epoch"), ("S1", 50, 1.0, "last_epoch"),
    ("S2", 75, 1.0, "last_epoch"), ("S3", 150, 1.0, "last_epoch"),
    ("S4", 200, 1.0, "last_epoch"), ("S5", 100, 0.5, "last_epoch"),
    ("S6", 150, 0.5, "last_epoch"), ("S7", 200, 0.5, "last_epoch"),
    ("S8", 100, 0.1, "last_epoch"), ("S9", 150, 0.1, "last_epoch"),
    ("S10", 100, 1.0, "best_train_loss"), ("S11", 150, 0.5, "best_train_loss"),
    ("S12", 200, 0.5, "best_train_loss"),
]
PRIORITY_IDS = {"S0", "S3", "S4", "S5", "S7", "S8"}
SUMMARY_COLUMNS = [
    "dataset", "max_epochs", "lr_mult", "checkpoint_mode", "mean_test_acc", "std_test_acc",
    "mean_train_acc", "std_train_acc", "mean_train_loss", "mean_selected_epoch", "num_train",
    "num_test", "num_channels", "seq_len", "num_classes",
]


def write_summary(results_file, summary_file):
    groups = {}
    with open(results_file, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (row["dataset"], row["max_epochs"], row["lr_mult"], row["checkpoint_mode"])
            groups.setdefault(key, []).append(row)
    with open(summary_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            dataset, max_epochs, lr_mult, checkpoint_mode = key
            first = rows[0]
            values = lambda name: [float(row[name]) for row in rows]
            writer.writerow(
                {
                    "dataset": dataset, "max_epochs": max_epochs, "lr_mult": lr_mult,
                    "checkpoint_mode": checkpoint_mode,
                    "mean_test_acc": f"{statistics.mean(values('test_acc')):.6f}",
                    "std_test_acc": f"{statistics.stdev(values('test_acc')) if len(rows)>1 else 0.0:.6f}",
                    "mean_train_acc": f"{statistics.mean(values('train_acc')):.6f}",
                    "std_train_acc": f"{statistics.stdev(values('train_acc')) if len(rows)>1 else 0.0:.6f}",
                    "mean_train_loss": f"{statistics.mean(values('train_loss')):.6f}",
                    "mean_selected_epoch": f"{statistics.mean(values('selected_epoch')):.6f}",
                    "num_train": first["num_train"], "num_test": first["num_test"],
                    "num_channels": first["num_channels"], "seq_len": first["seq_len"],
                    "num_classes": first["num_classes"],
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-grid", action="store_true", help="Run S0--S12 on every dataset, not only condition.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for schedule experiments, but torch.cuda.is_available() is False.")

    results_file = "schedule_results.csv"
    summary_file = "schedule_summary.csv"
    curve_file = "schedule_training_curve.csv"
    collapse_file = "condition_collapse_diagnostic.csv"
    for path in (results_file, summary_file, curve_file, collapse_file):
        if os.path.exists(path):
            os.remove(path)

    jobs = []
    for dataset in DATASETS:
        configs = ALL_CONFIGS if args.full_grid or dataset == "condition" else [x for x in ALL_CONFIGS if x[0] in PRIORITY_IDS]
        for config in configs:
            for seed in SEEDS:
                jobs.append((dataset, *config, seed))
    for index, (dataset, config_id, max_epochs, lr_mult, checkpoint_mode, seed) in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {dataset} {config_id} seed={seed}", flush=True)
        subprocess.run(
            [
                sys.executable, "trainer.py", "--data", os.path.join("data", dataset),
                "--ablation", "full_current", "--norm-mode", "train_global", "--graph-mode", "no_graph",
                "--encoder-mode", "current", "--relation-mode", "lowN_channel_attn",
                "--train-protocol", "train_all_fixed_100", "--max-epochs", str(max_epochs),
                "--lr-mult", str(lr_mult), "--checkpoint-mode", checkpoint_mode, "--seed", str(seed),
                "--results-file", results_file, "--schedule-curve-file", curve_file,
                "--collapse-diagnostic-file", collapse_file, "--results-schema", "schedule",
            ],
            check=True,
        )
    write_summary(results_file, summary_file)
    print(f"\nSaved {results_file}, {summary_file}, {curve_file}, and {collapse_file}")


if __name__ == "__main__":
    main()
