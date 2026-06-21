"""Run residual-SemiGCN graph experiments B0--B6 on the selected datasets."""

import csv
import os
import statistics
import subprocess
import sys

import torch


DATASETS = ["BasicMotions", "Cricket", "ERing", "Epilepsy", "Libras", "UWaveGestureLibrary"]
SEEDS = [0, 1, 2]
EXPERIMENTS = [
    {"id": "B0", "graph_mode": "no_graph", "sample_k": 5, "sim_threshold": 0.5, "dropedge": 0.0},
    {"id": "B1", "graph_mode": "current", "sample_k": 5, "sim_threshold": 0.5, "dropedge": 0.0},
    {"id": "B2", "graph_mode": "mutual_knn", "sample_k": 5, "sim_threshold": 0.5, "dropedge": 0.0},
    {"id": "B3", "graph_mode": "mutual_knn", "sample_k": 5, "sim_threshold": 0.5, "dropedge": 0.2},
    {"id": "B4", "graph_mode": "mutual_knn_threshold", "sample_k": 5, "sim_threshold": 0.5, "dropedge": 0.2},
    {"id": "B5", "graph_mode": "mutual_knn", "sample_k": 3, "sim_threshold": 0.5, "dropedge": 0.2},
    {"id": "B6", "graph_mode": "mutual_knn", "sample_k": 10, "sim_threshold": 0.5, "dropedge": 0.2},
]
SUMMARY_COLUMNS = [
    "dataset", "graph_mode", "sample_k", "sim_threshold", "dropedge",
    "mean_test_acc", "std_test_acc", "mean_best_val_acc", "std_best_val_acc",
    "mean_final_alpha", "mean_train_graph_homophily",
    "mean_test_graph_homophily_diagnostic_only",
]


def mean_or_blank(values):
    numbers = [float(value) for value in values if value not in ("", None)]
    return "" if not numbers else f"{statistics.mean(numbers):.6f}"


def std_or_zero(values):
    numbers = [float(value) for value in values]
    return f"{statistics.stdev(numbers) if len(numbers) > 1 else 0.0:.6f}"


def write_summary(results_file, summary_file):
    groups = {}
    with open(results_file, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = tuple(row[name] for name in ("dataset", "graph_mode", "sample_k", "sim_threshold", "dropedge"))
            groups.setdefault(key, []).append(row)

    with open(summary_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            dataset, graph_mode, sample_k, sim_threshold, dropedge = key
            writer.writerow(
                {
                    "dataset": dataset, "graph_mode": graph_mode, "sample_k": sample_k,
                    "sim_threshold": sim_threshold, "dropedge": dropedge,
                    "mean_test_acc": mean_or_blank([row["test_acc"] for row in rows]),
                    "std_test_acc": std_or_zero([row["test_acc"] for row in rows]),
                    "mean_best_val_acc": mean_or_blank([row["best_val_acc"] for row in rows]),
                    "std_best_val_acc": std_or_zero([row["best_val_acc"] for row in rows]),
                    "mean_final_alpha": mean_or_blank([row["final_alpha"] for row in rows]),
                    "mean_train_graph_homophily": mean_or_blank([row["train_graph_homophily"] for row in rows]),
                    "mean_test_graph_homophily_diagnostic_only": mean_or_blank(
                        [row["test_graph_homophily_diagnostic_only"] for row in rows]
                    ),
                }
            )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner, but torch.cuda.is_available() is False.")

    results_file, summary_file = "graph_v2_results.csv", "graph_v2_summary.csv"
    for path in (results_file, summary_file):
        if os.path.exists(path):
            os.remove(path)

    total = len(DATASETS) * len(EXPERIMENTS) * len(SEEDS)
    run_number = 0
    for dataset in DATASETS:
        for experiment in EXPERIMENTS:
            for seed in SEEDS:
                run_number += 1
                print(f"\n[{run_number}/{total}] {dataset} {experiment['id']} seed={seed}", flush=True)
                command = [
                    sys.executable, "trainer.py", "--data", os.path.join("data", dataset),
                    "--ablation", "full_current", "--seed", str(seed),
                    "--graph-mode", experiment["graph_mode"], "--sample-k", str(experiment["sample_k"]),
                    "--sim-threshold", str(experiment["sim_threshold"]), "--dropedge", str(experiment["dropedge"]),
                    "--use-residual-gcn", "--results-file", results_file, "--results-schema", "graph_v2",
                ]
                subprocess.run(command, check=True)

    write_summary(results_file, summary_file)
    print(f"\nSaved {results_file} and {summary_file}")


if __name__ == "__main__":
    main()
