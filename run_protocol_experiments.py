"""Run sixth-round training-protocol experiments on two relation modes."""

import csv
import math
import os
import statistics
import subprocess
import sys

import torch


DATASETS = ["BasicMotions", "Cricket", "ERing", "Epilepsy", "Libras", "UWaveGestureLibrary"]
RELATION_MODES = ["hgnn_zero", "lowN_channel_attn"]
PROTOCOLS = [
    "current_split", "stratified_split", "retrain_trainval",
    "train_all_fixed_50", "train_all_fixed_100",
]
SEEDS = [0, 1, 2]
SUMMARY_COLUMNS = [
    "dataset", "relation_mode", "train_protocol", "mean_test_acc", "std_test_acc",
    "mean_best_val_acc", "std_best_val_acc", "mean_best_epoch", "mean_final_epoch",
]
DIAGNOSTIC_COLUMNS = [
    "dataset", "relation_mode", "train_protocol", "mean_train_acc", "mean_val_acc",
    "mean_test_acc", "mean_best_epoch", "mean_final_epoch", "num_train_used_for_loss",
    "num_val", "num_test", "num_channels", "seq_len", "num_classes",
]


def numbers(rows, column):
    return [float(row[column]) for row in rows if not math.isnan(float(row[column]))]


def mean_or_nan(values):
    return "nan" if not values else f"{statistics.mean(values):.6f}"


def std_or_nan(values):
    return "nan" if not values else f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}"


def read_groups(results_file):
    groups = {}
    with open(results_file, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            groups.setdefault((row["dataset"], row["relation_mode"], row["train_protocol"]), []).append(row)
    return groups


def write_reports(results_file, summary_file, diagnostic_file):
    groups = read_groups(results_file)
    with open(summary_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for (dataset, relation_mode, protocol), rows in sorted(groups.items()):
            test = numbers(rows, "test_acc")
            best_val = numbers(rows, "best_val_acc")
            best_epoch = numbers(rows, "best_epoch")
            final_epoch = numbers(rows, "final_epoch")
            writer.writerow(
                {
                    "dataset": dataset, "relation_mode": relation_mode, "train_protocol": protocol,
                    "mean_test_acc": mean_or_nan(test), "std_test_acc": std_or_nan(test),
                    "mean_best_val_acc": mean_or_nan(best_val), "std_best_val_acc": std_or_nan(best_val),
                    "mean_best_epoch": mean_or_nan(best_epoch), "mean_final_epoch": mean_or_nan(final_epoch),
                }
            )

    with open(diagnostic_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for (dataset, relation_mode, protocol), rows in sorted(groups.items()):
            first = rows[0]
            writer.writerow(
                {
                    "dataset": dataset, "relation_mode": relation_mode, "train_protocol": protocol,
                    "mean_train_acc": mean_or_nan(numbers(rows, "train_acc")),
                    "mean_val_acc": mean_or_nan(numbers(rows, "val_acc")),
                    "mean_test_acc": mean_or_nan(numbers(rows, "test_acc")),
                    "mean_best_epoch": mean_or_nan(numbers(rows, "best_epoch")),
                    "mean_final_epoch": mean_or_nan(numbers(rows, "final_epoch")),
                    "num_train_used_for_loss": first["num_train_used_for_loss"], "num_val": first["num_val"],
                    "num_test": first["num_test"], "num_channels": first["num_channels"],
                    "seq_len": first["seq_len"], "num_classes": first["num_classes"],
                }
            )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner, but torch.cuda.is_available() is False.")
    results_file, summary_file, diagnostic_file = (
        "protocol_results.csv", "protocol_summary.csv", "protocol_diagnostic.csv"
    )
    for path in (results_file, summary_file, diagnostic_file):
        if os.path.exists(path):
            os.remove(path)

    total = len(DATASETS) * len(RELATION_MODES) * len(PROTOCOLS) * len(SEEDS)
    run_number = 0
    for dataset in DATASETS:
        for relation_mode in RELATION_MODES:
            for protocol in PROTOCOLS:
                for seed in SEEDS:
                    run_number += 1
                    print(f"\n[{run_number}/{total}] {dataset} {relation_mode} {protocol} seed={seed}", flush=True)
                    subprocess.run(
                        [
                            sys.executable, "trainer.py", "--data", os.path.join("data", dataset),
                            "--ablation", "full_current", "--norm-mode", "train_global",
                            "--graph-mode", "no_graph", "--encoder-mode", "current",
                            "--relation-mode", relation_mode, "--train-protocol", protocol,
                            "--seed", str(seed), "--results-file", results_file,
                            "--results-schema", "protocol",
                        ],
                        check=True,
                    )
    write_reports(results_file, summary_file, diagnostic_file)
    print(f"\nSaved {results_file}, {summary_file}, and {diagnostic_file}")


if __name__ == "__main__":
    main()
