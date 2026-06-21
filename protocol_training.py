"""Training-protocol experiments without changing the PDGNet architecture."""

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import config as cfg
from data_process import normalize_data
from model import PDGNet


PROTOCOL_COLUMNS = [
    "dataset", "seed", "norm_mode", "graph_mode", "encoder_mode", "relation_mode",
    "train_protocol", "train_acc", "val_acc", "test_acc", "best_val_acc", "best_epoch",
    "final_epoch", "num_train_used_for_loss", "num_val", "num_test", "num_channels",
    "seq_len", "num_classes",
]
FINAL12_COLUMNS = [
    "dataset", "seed", "norm_mode", "graph_mode", "encoder_mode", "relation_mode",
    "train_protocol", "train_acc", "test_acc", "final_epoch", "num_train", "num_test",
    "num_channels", "seq_len", "num_classes",
]
CURVE_COLUMNS = ["dataset", "seed", "epoch", "train_loss", "train_acc"]
SCHEDULE_COLUMNS = [
    "dataset", "seed", "norm_mode", "graph_mode", "encoder_mode", "relation_mode",
    "max_epochs", "lr_mult", "checkpoint_mode", "selected_epoch", "train_acc",
    "train_loss", "test_acc", "num_train", "num_test", "num_channels", "seq_len", "num_classes",
]
SCHEDULE_CURVE_COLUMNS = [
    "dataset", "seed", "max_epochs", "lr_mult", "checkpoint_mode", "epoch", "train_loss", "train_acc",
]
COLLAPSE_COLUMNS = [
    "dataset", "seed", "max_epochs", "lr_mult", "checkpoint_mode", "selected_epoch",
    "train_acc", "test_acc", "train_pred_class_counts", "test_pred_class_counts", "num_classes",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_split_mask(num_samples, val_ratio, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).numpy()
    val_count = int(num_samples * val_ratio)
    val_mask = np.zeros(num_samples, dtype=bool)
    val_mask[indices[:val_count]] = True
    return ~val_mask, val_mask


def stratified_split_mask(labels, val_ratio, seed):
    """Use sklearn stratification when possible, with a safe random fallback."""
    labels = np.asarray(labels)
    num_samples = len(labels)
    val_count = int(num_samples * val_ratio)
    classes, counts = np.unique(labels, return_counts=True)
    can_stratify = (
        len(classes) > 1
        and counts.min() >= 2
        and val_count >= len(classes)
        and num_samples - val_count >= len(classes)
    )
    if can_stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_count, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(num_samples), labels))
        train_mask = np.zeros(num_samples, dtype=bool)
        val_mask = np.zeros(num_samples, dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        return train_mask, val_mask
    return random_split_mask(num_samples, val_ratio, seed)


def load_raw_data(data_path):
    data_path = os.path.normpath(data_path)
    return (
        np.load(os.path.join(data_path, "X_train.npy")),
        np.load(os.path.join(data_path, "y_train.npy")).squeeze(),
        np.load(os.path.join(data_path, "X_test.npy")),
        np.load(os.path.join(data_path, "y_test.npy")).squeeze(),
        os.path.basename(data_path),
    )


def tensorize(raw_train, y_train, raw_test, y_test, train_fit_mask):
    X_train, X_test = normalize_data(raw_train, raw_test, train_fit_mask, "train_global")
    return (
        torch.tensor(X_train, dtype=torch.float32, device=cfg.DEVICE),
        torch.tensor(y_train, dtype=torch.long, device=cfg.DEVICE),
        torch.tensor(X_test, dtype=torch.float32, device=cfg.DEVICE),
        torch.tensor(y_test, dtype=torch.long, device=cfg.DEVICE),
    )


def create_model(args):
    return PDGNet(
        args.ablation, args.graph_mode, args.use_residual_gcn, args.encoder_mode, args.relation_mode
    ).to(cfg.DEVICE)


def accuracy(logits, labels, mask=None):
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    return accuracy_score(labels.detach().cpu(), logits.argmax(dim=1).detach().cpu())


def warmup(model, X, y, loss_mask, val_mask=None):
    """Original 30-epoch identity-adjacency warm-up stage."""
    identity = torch.eye(len(X), device=cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    best_val = float("-inf")
    best_state = None
    for _ in range(30):
        model.train()
        logits = model(X, identity)
        loss = criterion(logits[loss_mask], y[loss_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                value = accuracy(model(X, identity), y, val_mask)
            if value > best_val:
                best_val = value
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def train_with_validation(model, X, y, train_mask, val_mask):
    """Original phase-two optimization with validation early stopping."""
    identity = torch.eye(len(X), device=cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR * 0.3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, threshold=cfg.EARLY_STOP_DELTA
    )
    best_val, best_epoch, patience, best_state = float("-inf"), 0, 0, None
    for epoch in range(cfg.EPOCHS):
        model.train()
        logits = model(X, identity)
        loss = criterion(logits[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(X, identity), y, val_mask)
        if val_acc > best_val + cfg.EARLY_STOP_DELTA:
            best_val, best_epoch, patience = val_acc, epoch + 1, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
        scheduler.step(val_acc)
        if patience >= cfg.PATIENCE:
            break
    model.load_state_dict(best_state)
    return best_val, best_epoch


def train_fixed(model, X, y, epochs):
    """Phase-two fixed epoch training without a validation scheduler."""
    identity = torch.eye(len(X), device=cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR * 0.3)
    full_mask = torch.ones(len(X), dtype=torch.bool, device=cfg.DEVICE)
    for _ in range(epochs):
        model.train()
        logits = model(X, identity)
        loss = criterion(logits[full_mask], y[full_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, X_train, y_train, X_test, y_test, train_mask, val_mask=None):
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train, torch.eye(len(X_train), device=cfg.DEVICE))
        test_logits = model(X_test, torch.eye(len(X_test), device=cfg.DEVICE))
    train_acc = accuracy(train_logits, y_train, train_mask)
    val_acc = float("nan") if val_mask is None else accuracy(train_logits, y_train, val_mask)
    test_acc = accuracy(test_logits, y_test)
    return train_acc, val_acc, test_acc


def append_result(path, row):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PROTOCOL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_csv(path, row, columns):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_fixed_with_curve(model, X, y, epochs, dataset, seed, curve_file):
    """Full-train fixed protocol with post-update diagnostics at fixed epochs."""
    identity = torch.eye(len(X), device=cfg.DEVICE)
    full_mask = torch.ones(len(X), dtype=torch.bool, device=cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR * 0.3)
    checkpoints = {1, 10, 25, 50, 75, 100}
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(X, identity)
        loss = criterion(logits[full_mask], y[full_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                eval_logits = model(X, identity)
                eval_loss = criterion(eval_logits, y).item()
                eval_acc = accuracy(eval_logits, y)
            append_csv(
                curve_file,
                {
                    "dataset": dataset, "seed": seed, "epoch": epoch,
                    "train_loss": f"{eval_loss:.6f}", "train_acc": f"{eval_acc:.6f}",
                },
                CURVE_COLUMNS,
            )


def run_final12(args):
    """Train the selected final configuration on all original training samples."""
    expected = {
        "norm_mode": "train_global", "graph_mode": "no_graph", "encoder_mode": "current",
        "relation_mode": "lowN_channel_attn", "train_protocol": "train_all_fixed_100",
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"Final-12 evaluation requires --{name.replace('_', '-')} {value}")
    if not args.training_curve_file:
        raise ValueError("Final-12 evaluation requires --training-curve-file")

    set_seed(args.seed)
    raw_train, y_train_raw, raw_test, y_test_raw, dataset = load_raw_data(args.data)
    cfg.NUM_NODES = raw_train.shape[1]
    cfg.NUM_CLASSES = len(np.unique(y_train_raw))
    full_mask_np = np.ones(len(raw_train), dtype=bool)
    X_train, y_train, X_test, y_test = tensorize(
        raw_train, y_train_raw, raw_test, y_test_raw, full_mask_np
    )
    model = create_model(args)
    train_fixed_with_curve(model, X_train, y_train, 100, dataset, args.seed, args.training_curve_file)
    full_mask = torch.ones(len(X_train), dtype=torch.bool, device=cfg.DEVICE)
    train_acc, _, test_acc = evaluate(model, X_train, y_train, X_test, y_test, full_mask)
    append_csv(
        args.results_file,
        {
            "dataset": dataset, "seed": args.seed, "norm_mode": args.norm_mode,
            "graph_mode": args.graph_mode, "encoder_mode": args.encoder_mode,
            "relation_mode": args.relation_mode, "train_protocol": args.train_protocol,
            "train_acc": f"{train_acc:.6f}", "test_acc": f"{test_acc:.6f}",
            "final_epoch": 100, "num_train": len(X_train), "num_test": len(X_test),
            "num_channels": cfg.NUM_NODES, "seq_len": raw_train.shape[2], "num_classes": cfg.NUM_CLASSES,
        },
        FINAL12_COLUMNS,
    )
    print(f"Final12 | Dataset={dataset} | seed={args.seed} | Test Acc={test_acc:.4f}")


def class_count_string(predictions, num_classes):
    counts = torch.bincount(predictions.detach().cpu(), minlength=num_classes).tolist()
    return ",".join(f"{index}:{count}" for index, count in enumerate(counts))


def run_schedule(args):
    """Fixed full-train scheduling experiment; checkpoints use training loss only."""
    expected = {
        "norm_mode": "train_global", "graph_mode": "no_graph", "encoder_mode": "current",
        "relation_mode": "lowN_channel_attn",
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"Schedule experiments require --{name.replace('_', '-')} {value}")
    if not args.schedule_curve_file:
        raise ValueError("Schedule experiments require --schedule-curve-file")

    set_seed(args.seed)
    raw_train, y_train_raw, raw_test, y_test_raw, dataset = load_raw_data(args.data)
    cfg.NUM_NODES = raw_train.shape[1]
    cfg.NUM_CLASSES = len(np.unique(y_train_raw))
    full_mask_np = np.ones(len(raw_train), dtype=bool)
    X_train, y_train, X_test, y_test = tensorize(raw_train, y_train_raw, raw_test, y_test_raw, full_mask_np)
    model = create_model(args)
    identity_train = torch.eye(len(X_train), device=cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    # This is the effective LR used by the existing fixed-training phase.
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR * 0.3 * args.lr_mult)
    checkpoints = {1, 10, 25, 50, 75, 100, 150, 200}
    best_loss, selected_epoch, selected_state = float("inf"), 0, None
    selected_train_acc, selected_train_loss = float("nan"), float("nan")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        logits = model(X_train, identity_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss is measured after the update in eval mode; it is the only signal
        # used by best_train_loss and never accesses test data.
        model.eval()
        with torch.no_grad():
            eval_logits = model(X_train, identity_train)
            eval_loss = criterion(eval_logits, y_train).item()
            eval_acc = accuracy(eval_logits, y_train)

        if args.checkpoint_mode == "best_train_loss" and eval_loss < best_loss:
            best_loss, selected_epoch = eval_loss, epoch
            selected_train_acc, selected_train_loss = eval_acc, eval_loss
            selected_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch in checkpoints:
            append_csv(
                args.schedule_curve_file,
                {
                    "dataset": dataset, "seed": args.seed, "max_epochs": args.max_epochs,
                    "lr_mult": args.lr_mult, "checkpoint_mode": args.checkpoint_mode,
                    "epoch": epoch, "train_loss": f"{eval_loss:.6f}", "train_acc": f"{eval_acc:.6f}",
                },
                SCHEDULE_CURVE_COLUMNS,
            )

    if args.checkpoint_mode == "best_train_loss":
        model.load_state_dict(selected_state)
    else:
        selected_epoch = args.max_epochs
        selected_train_acc, selected_train_loss = eval_acc, eval_loss

    model.eval()
    with torch.no_grad():
        train_logits = model(X_train, identity_train)
        test_logits = model(X_test, torch.eye(len(X_test), device=cfg.DEVICE))
    train_acc = accuracy(train_logits, y_train)
    train_loss = criterion(train_logits, y_train).item()
    test_acc = accuracy(test_logits, y_test)
    append_csv(
        args.results_file,
        {
            "dataset": dataset, "seed": args.seed, "norm_mode": args.norm_mode,
            "graph_mode": args.graph_mode, "encoder_mode": args.encoder_mode,
            "relation_mode": args.relation_mode, "max_epochs": args.max_epochs,
            "lr_mult": args.lr_mult, "checkpoint_mode": args.checkpoint_mode,
            "selected_epoch": selected_epoch, "train_acc": f"{train_acc:.6f}",
            "train_loss": f"{train_loss:.6f}", "test_acc": f"{test_acc:.6f}",
            "num_train": len(X_train), "num_test": len(X_test), "num_channels": cfg.NUM_NODES,
            "seq_len": raw_train.shape[2], "num_classes": cfg.NUM_CLASSES,
        },
        SCHEDULE_COLUMNS,
    )
    if dataset == "condition" and args.collapse_diagnostic_file:
        append_csv(
            args.collapse_diagnostic_file,
            {
                "dataset": dataset, "seed": args.seed, "max_epochs": args.max_epochs,
                "lr_mult": args.lr_mult, "checkpoint_mode": args.checkpoint_mode,
                "selected_epoch": selected_epoch, "train_acc": f"{train_acc:.6f}",
                "test_acc": f"{test_acc:.6f}",
                "train_pred_class_counts": class_count_string(train_logits.argmax(dim=1), cfg.NUM_CLASSES),
                "test_pred_class_counts": class_count_string(test_logits.argmax(dim=1), cfg.NUM_CLASSES),
                "num_classes": cfg.NUM_CLASSES,
            },
            COLLAPSE_COLUMNS,
        )
    print(
        f"Schedule | Dataset={dataset} | seed={args.seed} | epochs={args.max_epochs} | "
        f"lr_mult={args.lr_mult} | checkpoint={args.checkpoint_mode} | test={test_acc:.4f}"
    )


def run_protocol(args):
    if args.graph_mode != "no_graph" or args.norm_mode != "train_global" or args.encoder_mode != "current":
        raise ValueError("Protocol experiments require no_graph, train_global, and encoder_mode=current")

    set_seed(args.seed)
    raw_train, y_train_raw, raw_test, y_test_raw, dataset = load_raw_data(args.data)
    cfg.NUM_NODES = raw_train.shape[1]
    cfg.NUM_CLASSES = len(np.unique(y_train_raw))
    seq_len = raw_train.shape[2]
    protocol = args.train_protocol
    if protocol in {"current_split", "stratified_split", "retrain_trainval"}:
        if protocol == "current_split":
            stage_train_mask_np, val_mask_np = random_split_mask(len(raw_train), cfg.VAL_RATIO, args.seed)
        else:
            stage_train_mask_np, val_mask_np = stratified_split_mask(y_train_raw, cfg.VAL_RATIO, args.seed)
        X_stage, y_stage, X_test, y_test = tensorize(
            raw_train, y_train_raw, raw_test, y_test_raw, stage_train_mask_np
        )
        stage_train_mask = torch.tensor(stage_train_mask_np, dtype=torch.bool, device=cfg.DEVICE)
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool, device=cfg.DEVICE)

        model = create_model(args)
        warmup(model, X_stage, y_stage, stage_train_mask, val_mask)
        best_val, best_epoch = train_with_validation(model, X_stage, y_stage, stage_train_mask, val_mask)

        if protocol == "retrain_trainval":
            # Fresh initialization; second stage uses all original training samples
            # and train-global normalization fitted on that full training set.
            set_seed(args.seed)
            full_mask_np = np.ones(len(raw_train), dtype=bool)
            X_full, y_full, X_test, y_test = tensorize(
                raw_train, y_train_raw, raw_test, y_test_raw, full_mask_np
            )
            model = create_model(args)
            full_mask = torch.ones(len(X_full), dtype=torch.bool, device=cfg.DEVICE)
            # The retraining phase is exactly best_epoch updates after a fresh
            # initialization, as specified by the protocol.
            train_fixed(model, X_full, y_full, best_epoch)
            train_acc, _, test_acc = evaluate(model, X_full, y_full, X_test, y_test, full_mask)
            val_acc = best_val
            num_train_used, num_val = len(X_full), int(val_mask.sum().item())
            final_epoch = best_epoch
        else:
            train_acc, val_acc, test_acc = evaluate(
                model, X_stage, y_stage, X_test, y_test, stage_train_mask, val_mask
            )
            num_train_used, num_val = int(stage_train_mask.sum().item()), int(val_mask.sum().item())
            final_epoch = best_epoch
    else:
        fixed_epochs = 50 if protocol == "train_all_fixed_50" else 100
        full_mask_np = np.ones(len(raw_train), dtype=bool)
        X_full, y_full, X_test, y_test = tensorize(raw_train, y_train_raw, raw_test, y_test_raw, full_mask_np)
        full_mask = torch.ones(len(X_full), dtype=torch.bool, device=cfg.DEVICE)
        model = create_model(args)
        # Fixed protocols perform exactly 50 or 100 total optimization epochs.
        train_fixed(model, X_full, y_full, fixed_epochs)
        train_acc, val_acc, test_acc = evaluate(model, X_full, y_full, X_test, y_test, full_mask)
        best_val, best_epoch, final_epoch = float("nan"), float("nan"), fixed_epochs
        num_train_used, num_val = len(X_full), 0

    row = {
        "dataset": dataset, "seed": args.seed, "norm_mode": args.norm_mode,
        "graph_mode": args.graph_mode, "encoder_mode": args.encoder_mode,
        "relation_mode": args.relation_mode, "train_protocol": protocol,
        "train_acc": f"{train_acc:.6f}", "val_acc": f"{val_acc:.6f}",
        "test_acc": f"{test_acc:.6f}", "best_val_acc": f"{best_val:.6f}",
        "best_epoch": best_epoch, "final_epoch": final_epoch,
        "num_train_used_for_loss": num_train_used, "num_val": num_val,
        "num_test": len(X_test), "num_channels": cfg.NUM_NODES, "seq_len": seq_len,
        "num_classes": cfg.NUM_CLASSES,
    }
    append_result(args.results_file, row)
    print(
        f"Protocol={protocol} | Dataset={dataset} | seed={args.seed} | "
        f"Test Acc={test_acc:.4f} | final_epoch={final_epoch}"
    )
