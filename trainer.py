import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
from build_sample_graph import build_sample_graph
from data_process import NORM_MODES, normalize_data
from model import PDGNet
from utils import load_dataset


ABLATION_COLUMNS = [
    "dataset", "seed", "ablation", "train_acc", "val_acc", "test_acc",
    "best_val_acc", "best_epoch", "num_train", "num_val", "num_test",
    "num_channels", "seq_len", "num_classes", "train_graph_avg_degree",
    "train_graph_homophily", "test_graph_avg_degree",
    "test_graph_homophily_diagnostic_only",
]
GRAPH_V2_COLUMNS = [
    "dataset", "seed", "graph_mode", "sample_k", "sim_threshold", "dropedge",
    "use_residual_gcn", "final_alpha", "train_acc", "val_acc", "test_acc",
    "best_val_acc", "best_epoch", "num_train", "num_val", "num_test",
    "num_channels", "seq_len", "num_classes", "train_graph_avg_degree",
    "train_graph_homophily", "test_graph_avg_degree",
    "test_graph_homophily_diagnostic_only",
]
NORM_COLUMNS = [
    "dataset", "seed", "norm_mode", "graph_mode", "train_acc", "val_acc",
    "test_acc", "best_val_acc", "best_epoch", "num_train", "num_val",
    "num_test", "num_channels", "seq_len", "num_classes",
]
ENCODER_COLUMNS = [
    "dataset", "seed", "norm_mode", "graph_mode", "encoder_mode", "train_acc",
    "val_acc", "test_acc", "best_val_acc", "best_epoch", "num_train", "num_val",
    "num_test", "num_channels", "seq_len", "num_classes",
]
RELATION_COLUMNS = [
    "dataset", "seed", "norm_mode", "graph_mode", "encoder_mode", "relation_mode",
    "train_acc", "val_acc", "test_acc", "best_val_acc", "best_epoch", "num_train",
    "num_val", "num_test", "num_channels", "seq_len", "num_classes",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_label_mask(num_samples, val_ratio, device):
    indices = torch.randperm(num_samples, device=device)
    val_num = int(num_samples * val_ratio)
    train_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
    train_mask[indices[val_num:]] = True
    val_mask[indices[:val_num]] = True
    return train_mask, val_mask


def build_adjacency(model, features, sample_k, sim_threshold, dropedge=0.0, training=False):
    """Build a graph only when the residual graph branch is enabled."""
    if model.graph_enabled:
        return build_sample_graph(
            features,
            k=min(sample_k, len(features)),
            mode=model.graph_mode,
            sim_threshold=sim_threshold,
            dropedge=dropedge,
            training=training,
        )
    return torch.eye(len(features), device=features.device)


def graph_diagnostics(adj, labels):
    """Unweighted degree and undirected edge homophily, excluding self loops."""
    connected = adj.detach() > 0
    connected.fill_diagonal_(False)
    avg_degree = connected.sum(dim=1).float().mean().item()
    upper = torch.triu(connected, diagonal=1)
    src, dst = upper.nonzero(as_tuple=True)
    homophily = float("nan") if len(src) == 0 else (labels[src] == labels[dst]).float().mean().item()
    return avg_degree, homophily


def accuracy_from_logits(logits, labels, mask=None):
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    return accuracy_score(labels.detach().cpu(), logits.argmax(dim=1).detach().cpu())


def append_result(path, row, columns):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_data(data_path):
    if data_path is None:
        return (*load_dataset(cfg.DATASET, cfg.DATA_DIR), cfg.DATASET)
    data_path = os.path.normpath(data_path)
    return (
        np.load(os.path.join(data_path, "X_train.npy")),
        np.load(os.path.join(data_path, "y_train.npy")),
        np.load(os.path.join(data_path, "X_test.npy")),
        np.load(os.path.join(data_path, "y_test.npy")),
        os.path.basename(data_path),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--ablation", choices=sorted(PDGNet.ABLATIONS), default="full_current")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph-mode", choices=["current", "mutual_knn", "mutual_knn_threshold", "no_graph"], default="current")
    parser.add_argument("--sample-k", type=int, default=cfg.K_SAMPLE)
    parser.add_argument("--sim-threshold", type=float, default=0.5)
    parser.add_argument("--dropedge", type=float, default=0.0)
    parser.add_argument("--use-residual-gcn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-mode", choices=NORM_MODES, default="sample_global")
    parser.add_argument("--encoder-mode", choices=sorted(PDGNet.ENCODER_MODES), default="current")
    parser.add_argument("--relation-mode", choices=sorted(PDGNet.RELATION_MODES), default="hgnn_zero")
    parser.add_argument(
        "--train-protocol",
        choices=["current_split", "stratified_split", "retrain_trainval", "train_all_fixed_50", "train_all_fixed_100"],
        default="current_split",
    )
    parser.add_argument("--training-curve-file", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--checkpoint-mode", choices=["last_epoch", "best_train_loss"], default="last_epoch")
    parser.add_argument("--schedule-curve-file", type=str, default=None)
    parser.add_argument("--collapse-diagnostic-file", type=str, default=None)
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--results-schema", choices=["ablation", "graph_v2", "norm", "encoder", "relation", "protocol", "final12", "schedule"], default="ablation")
    args = parser.parse_args()
    if args.results_schema == "protocol":
        from protocol_training import run_protocol

        run_protocol(args)
        return
    if args.results_schema == "final12":
        from protocol_training import run_final12

        run_final12(args)
        return
    if args.results_schema == "schedule":
        from protocol_training import run_schedule

        if args.max_epochs not in {50, 75, 100, 150, 200}:
            raise ValueError("Schedule experiments require --max-epochs in {50, 75, 100, 150, 200}")
        if args.lr_mult not in {1.0, 0.5, 0.1}:
            raise ValueError("Schedule experiments require --lr-mult in {1.0, 0.5, 0.1}")
        run_schedule(args)
        return
    if args.sample_k < 1:
        raise ValueError("--sample-k must be positive")
    if not 0.0 <= args.dropedge < 1.0:
        raise ValueError("--dropedge must be in [0, 1)")
    if args.results_schema == "norm" and args.graph_mode != "no_graph":
        raise ValueError("Normalization experiments require --graph-mode no_graph")
    if args.results_schema == "encoder" and (args.graph_mode != "no_graph" or args.norm_mode != "train_global"):
        raise ValueError("Encoder experiments require --graph-mode no_graph and --norm-mode train_global")
    if args.results_schema == "relation" and (
        args.graph_mode != "no_graph" or args.norm_mode != "train_global" or args.encoder_mode != "current"
    ):
        raise ValueError("Relation experiments require no_graph, train_global, and encoder_mode=current")

    set_seed(args.seed)
    X_train, y_train, X_test, y_test, dataset_name = load_data(args.data)
    y_train, y_test = np.asarray(y_train).squeeze(), np.asarray(y_test).squeeze()
    cfg.NUM_NODES, cfg.NUM_CLASSES = X_train.shape[1], len(np.unique(y_train))
    seq_len = X_train.shape[2]

    # Split before normalization so train-based statistics never include validation
    # or test samples. The split is seed-controlled and shared by every norm mode.
    train_mask_cpu, val_mask_cpu = get_label_mask(len(X_train), cfg.VAL_RATIO, "cpu")
    X_train, X_test = normalize_data(X_train, X_test, train_mask_cpu.numpy(), args.norm_mode)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=cfg.DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long, device=cfg.DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=cfg.DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long, device=cfg.DEVICE)

    model = PDGNet(
        args.ablation, args.graph_mode, args.use_residual_gcn, args.encoder_mode, args.relation_mode
    ).to(cfg.DEVICE)
    train_mask, val_mask = train_mask_cpu.to(cfg.DEVICE), val_mask_cpu.to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    eye_adj = torch.eye(len(X_train), device=cfg.DEVICE)
    print(
        f"Dataset={dataset_name} | seed={args.seed} | ablation={args.ablation} | "
        f"graph={args.graph_mode} | norm={args.norm_mode} | encoder={args.encoder_mode} | "
        f"relation={args.relation_mode} | k={args.sample_k} | "
        f"dropedge={args.dropedge} | device={cfg.DEVICE}"
    )

    # Keep the original 30-epoch identity-adjacency pre-training stage.
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    best_pre = float("-inf")
    for _ in range(30):
        model.train()
        logits = model(X_train, eye_adj)
        loss = criterion(logits[train_mask], y_train[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pre_val_acc = accuracy_from_logits(model(X_train, eye_adj), y_train, val_mask)
        if pre_val_acc > best_pre:
            best_pre = pre_val_acc
            torch.save(model.state_dict(), "best_pretrain.pth")
    model.load_state_dict(torch.load("best_pretrain.pth", map_location=cfg.DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR * 0.3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, threshold=cfg.EARLY_STOP_DELTA
    )
    best_val_acc, best_epoch, patience = float("-inf"), 0, 0
    for epoch in range(cfg.EPOCHS):
        model.eval()
        with torch.no_grad():
            train_features = model(X_train)
            # DropEdge is deliberately restricted to this training forward pass.
            train_adj = build_adjacency(
                model, train_features, args.sample_k, args.sim_threshold, args.dropedge, training=True
            )

        model.train()
        logits = model(X_train, train_adj)
        loss = criterion(logits[train_mask], y_train[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation rebuilds a graph without DropEdge and therefore does not use
        # stochastic training edges for checkpoint selection.
        model.eval()
        with torch.no_grad():
            validation_features = model(X_train)
            validation_adj = build_adjacency(model, validation_features, args.sample_k, args.sim_threshold)
            val_acc = accuracy_from_logits(model(X_train, validation_adj), y_train, val_mask)

        if val_acc > best_val_acc + cfg.EARLY_STOP_DELTA:
            best_val_acc, best_epoch, patience = val_acc, epoch + 1, 0
            torch.save(model.state_dict(), "best.pth")
        else:
            patience += 1
        scheduler.step(val_acc)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} | Loss {loss.item():.4f} | Val {val_acc:.4f} | Patience {patience}/{cfg.PATIENCE}")
        if patience >= cfg.PATIENCE:
            break

    model.load_state_dict(torch.load("best.pth", map_location=cfg.DEVICE))
    model.eval()
    with torch.no_grad():
        train_features = model(X_train)
        final_train_adj = build_adjacency(model, train_features, args.sample_k, args.sim_threshold)
        train_logits = model(X_train, final_train_adj)
        train_acc = accuracy_from_logits(train_logits, y_train, train_mask)
        val_acc = accuracy_from_logits(train_logits, y_train, val_mask)

        # The test prediction retains the existing transductive, label-free graph protocol.
        all_features = torch.cat([train_features, model(X_test)])
        all_adj = build_adjacency(model, all_features, args.sample_k, args.sim_threshold)
        test_logits = model(torch.cat([X_train, X_test]), all_adj)[len(X_train):]
        test_acc = accuracy_from_logits(test_logits, y_test)

        if model.graph_enabled:
            train_degree, train_homophily = graph_diagnostics(final_train_adj, y_train)
            # Test labels enter only this post-hoc metric; they never affect graph construction.
            diagnostic_test_adj = build_adjacency(model, model(X_test), args.sample_k, args.sim_threshold)
            test_degree, test_homophily = graph_diagnostics(diagnostic_test_adj, y_test)
        else:
            train_degree = train_homophily = test_degree = test_homophily = ""
        final_alpha = float(model.gcn_alpha().item()) if model.graph_enabled else 0.0

    print(f"Test Acc: {test_acc:.4f} | Final alpha: {final_alpha:.4f}")
    print(f"Test F1:  {f1_score(y_test.cpu(), test_logits.argmax(dim=1).cpu(), average='macro'):.4f}")

    if args.results_file:
        common = {
            "dataset": dataset_name, "seed": args.seed, "train_acc": f"{train_acc:.6f}",
            "val_acc": f"{val_acc:.6f}", "test_acc": f"{test_acc:.6f}",
            "best_val_acc": f"{best_val_acc:.6f}", "best_epoch": best_epoch,
            "num_train": int(train_mask.sum().item()), "num_val": int(val_mask.sum().item()),
            "num_test": len(X_test), "num_channels": cfg.NUM_NODES, "seq_len": seq_len,
            "num_classes": cfg.NUM_CLASSES, "train_graph_avg_degree": train_degree,
            "train_graph_homophily": train_homophily, "test_graph_avg_degree": test_degree,
            "test_graph_homophily_diagnostic_only": test_homophily,
        }
        if args.results_schema == "graph_v2":
            common.update(
                {
                    "graph_mode": args.graph_mode, "sample_k": args.sample_k,
                    "sim_threshold": args.sim_threshold, "dropedge": args.dropedge,
                    "use_residual_gcn": args.use_residual_gcn, "final_alpha": f"{final_alpha:.6f}",
                }
            )
            append_result(args.results_file, common, GRAPH_V2_COLUMNS)
        elif args.results_schema == "norm":
            common.update({"norm_mode": args.norm_mode, "graph_mode": args.graph_mode})
            norm_row = {column: common[column] for column in NORM_COLUMNS}
            append_result(args.results_file, norm_row, NORM_COLUMNS)
        elif args.results_schema == "encoder":
            common.update(
                {
                    "norm_mode": args.norm_mode,
                    "graph_mode": args.graph_mode,
                    "encoder_mode": args.encoder_mode,
                }
            )
            encoder_row = {column: common[column] for column in ENCODER_COLUMNS}
            append_result(args.results_file, encoder_row, ENCODER_COLUMNS)
        elif args.results_schema == "relation":
            common.update(
                {
                    "norm_mode": args.norm_mode,
                    "graph_mode": args.graph_mode,
                    "encoder_mode": args.encoder_mode,
                    "relation_mode": args.relation_mode,
                }
            )
            relation_row = {column: common[column] for column in RELATION_COLUMNS}
            append_result(args.results_file, relation_row, RELATION_COLUMNS)
        else:
            common["ablation"] = args.ablation
            append_result(args.results_file, common, ABLATION_COLUMNS)


if __name__ == "__main__":
    main()
