"""Run HGNN low-variable relation experiments E0--E3."""

import csv
import os
import statistics
import subprocess
import sys

import torch


DATASETS = ["BasicMotions", "Cricket", "ERing", "Epilepsy", "Libras", "UWaveGestureLibrary"]
RELATION_MODES = ["hgnn_zero", "lowN_channel_attn", "lowN_node_pool", "channel_attn_all"]
SEEDS = [0, 1, 2]
SUMMARY_COLUMNS = [
    "dataset", "relation_mode", "mean_test_acc", "std_test_acc",
    "mean_best_val_acc", "std_best_val_acc",
]
DIAGNOSTIC_COLUMNS = [
    "dataset", "relation_mode", "mean_train_acc", "mean_val_acc", "mean_test_acc",
    "mean_best_epoch", "num_channels", "seq_len", "num_classes",
]


def read_groups(results_file):
    groups = {}
    with open(results_file, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            groups.setdefault((row["dataset"], row["relation_mode"]), []).append(row)
    return groups


def write_reports(results_file, summary_file, diagnostic_file):
    groups = read_groups(results_file)
    with open(summary_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for (dataset, relation_mode), rows in sorted(groups.items()):
            test_scores = [float(row["test_acc"]) for row in rows]
            best_vals = [float(row["best_val_acc"]) for row in rows]
            writer.writerow(
                {
                    "dataset": dataset, "relation_mode": relation_mode,
                    "mean_test_acc": f"{statistics.mean(test_scores):.6f}",
                    "std_test_acc": f"{statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0:.6f}",
                    "mean_best_val_acc": f"{statistics.mean(best_vals):.6f}",
                    "std_best_val_acc": f"{statistics.stdev(best_vals) if len(best_vals) > 1 else 0.0:.6f}",
                }
            )

    with open(diagnostic_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for (dataset, relation_mode), rows in sorted(groups.items()):
            first = rows[0]
            writer.writerow(
                {
                    "dataset": dataset, "relation_mode": relation_mode,
                    "mean_train_acc": f"{statistics.mean(float(row['train_acc']) for row in rows):.6f}",
                    "mean_val_acc": f"{statistics.mean(float(row['val_acc']) for row in rows):.6f}",
                    "mean_test_acc": f"{statistics.mean(float(row['test_acc']) for row in rows):.6f}",
                    "mean_best_epoch": f"{statistics.mean(float(row['best_epoch']) for row in rows):.6f}",
                    "num_channels": first["num_channels"], "seq_len": first["seq_len"],
                    "num_classes": first["num_classes"],
                }
            )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner, but torch.cuda.is_available() is False.")

    results_file, summary_file, diagnostic_file = (
        "relation_results.csv", "relation_summary.csv", "relation_diagnostic.csv"
    )
    for path in (results_file, summary_file, diagnostic_file):
        if os.path.exists(path):
            os.remove(path)

    total = len(DATASETS) * len(RELATION_MODES) * len(SEEDS)
    run_number = 0
    for dataset in DATASETS:
        for relation_mode in RELATION_MODES:
            for seed in SEEDS:
                run_number += 1
                print(f"\n[{run_number}/{total}] {dataset} relation={relation_mode} seed={seed}", flush=True)
                subprocess.run(
                    [
                        sys.executable, "trainer.py", "--data", os.path.join("data", dataset),
                        "--ablation", "full_current", "--graph-mode", "no_graph",
                        "--norm-mode", "train_global", "--encoder-mode", "current",
                        "--relation-mode", relation_mode, "--seed", str(seed),
                        "--results-file", results_file, "--results-schema", "relation",
                    ],
                    check=True,
                )

    write_reports(results_file, summary_file, diagnostic_file)
    print(f"\nSaved {results_file}, {summary_file}, and {diagnostic_file}")


if __name__ == "__main__":
    main()
