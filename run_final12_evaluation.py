"""Evaluate the selected final configuration on all 12 datasets and five seeds."""

import csv
import os
import statistics
import subprocess
import sys

import torch


DATASETS = [
    "condition", "FingerMovements", "NATOPS", "Epilepsy", "BasicMotions",
    "HandMovementDirection", "SelfRegulationSCP2", "ERing", "UWaveGestureLibrary",
    "Cricket", "Libras", "AtrialFibrillation",
]
SEEDS = [0, 1, 2, 3, 4]
REFERENCE = {
    "condition": (0.980, 1.000), "FingerMovements": (0.510, 0.660),
    "NATOPS": (0.730, 0.906), "Epilepsy": (0.540, 0.841),
    "BasicMotions": (0.970, 0.925), "HandMovementDirection": (0.350, 0.581),
    "SelfRegulationSCP2": (0.500, 0.556), "ERing": (0.590, 0.956),
    "UWaveGestureLibrary": (0.360, 0.875), "Cricket": (0.440, 0.944),
    "Libras": (0.230, 0.800), "AtrialFibrillation": (0.270, 0.533),
}
SUMMARY_COLUMNS = [
    "dataset", "mean_test_acc", "std_test_acc", "mean_train_acc", "std_train_acc",
    "num_train", "num_test", "num_channels", "seq_len", "num_classes",
]
COMPARE_COLUMNS = [
    "dataset", "original_pdgnet_acc", "final_mean_test_acc", "final_std_test_acc",
    "best_traditional_baseline", "diff_vs_original_pdgnet", "diff_vs_best_traditional_baseline",
]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_reports(results_file, summary_file, compare_file):
    groups = {}
    for row in read_rows(results_file):
        groups.setdefault(row["dataset"], []).append(row)

    summaries = {}
    with open(summary_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for dataset in DATASETS:
            rows = groups[dataset]
            test = [float(row["test_acc"]) for row in rows]
            train = [float(row["train_acc"]) for row in rows]
            first = rows[0]
            summary = {
                "dataset": dataset,
                "mean_test_acc": statistics.mean(test),
                "std_test_acc": statistics.stdev(test) if len(test) > 1 else 0.0,
                "mean_train_acc": statistics.mean(train),
                "std_train_acc": statistics.stdev(train) if len(train) > 1 else 0.0,
                "num_train": first["num_train"], "num_test": first["num_test"],
                "num_channels": first["num_channels"], "seq_len": first["seq_len"],
                "num_classes": first["num_classes"],
            }
            summaries[dataset] = summary
            writer.writerow(
                {key: f"{value:.6f}" if isinstance(value, float) else value for key, value in summary.items()}
            )

    with open(compare_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COMPARE_COLUMNS)
        writer.writeheader()
        for dataset in DATASETS:
            original, baseline = REFERENCE[dataset]
            summary = summaries[dataset]
            writer.writerow(
                {
                    "dataset": dataset, "original_pdgnet_acc": f"{original:.6f}",
                    "final_mean_test_acc": f"{summary['mean_test_acc']:.6f}",
                    "final_std_test_acc": f"{summary['std_test_acc']:.6f}",
                    "best_traditional_baseline": f"{baseline:.6f}",
                    "diff_vs_original_pdgnet": f"{summary['mean_test_acc'] - original:.6f}",
                    "diff_vs_best_traditional_baseline": f"{summary['mean_test_acc'] - baseline:.6f}",
                }
            )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final evaluation, but torch.cuda.is_available() is False.")
    results_file = "final12_results.csv"
    curve_file = "final12_training_curve.csv"
    summary_file = "final12_summary.csv"
    compare_file = "final12_compare.csv"
    for path in (results_file, curve_file, summary_file, compare_file):
        if os.path.exists(path):
            os.remove(path)

    total = len(DATASETS) * len(SEEDS)
    run_number = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            run_number += 1
            print(f"\n[{run_number}/{total}] {dataset} seed={seed}", flush=True)
            subprocess.run(
                [
                    sys.executable, "trainer.py", "--data", os.path.join("data", dataset),
                    "--ablation", "full_current", "--norm-mode", "train_global",
                    "--graph-mode", "no_graph", "--encoder-mode", "current",
                    "--relation-mode", "lowN_channel_attn", "--train-protocol", "train_all_fixed_100",
                    "--seed", str(seed), "--results-file", results_file,
                    "--training-curve-file", curve_file, "--results-schema", "final12",
                ],
                check=True,
            )
    write_reports(results_file, summary_file, compare_file)
    print(f"\nSaved {results_file}, {summary_file}, {compare_file}, and {curve_file}")


if __name__ == "__main__":
    main()
