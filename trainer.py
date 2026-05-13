import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config as cfg
from data_process import process_data
from utils import load_dataset, get_loader, compute_metrics
from model import BaselineModel


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    """评估一个 epoch，返回真实标签和预测标签。"""
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            pred = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)


def main():
    # ==================== 1. 数据加载与预处理 ====================
    X_train, y_train, X_test, y_test = load_dataset(cfg.DATASET, cfg.DATA_DIR)
    X_train, X_test = process_data(X_train, X_test, cfg.PERIOD_LENGTH)

    # 划分训练集 / 验证集
    n_total = len(X_train)
    n_val = int(n_total * cfg.VAL_RATIO)
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    train_loader = get_loader(X_tr, y_tr, cfg.BATCH_SIZE, shuffle=True)
    val_loader = get_loader(X_val, y_val, cfg.BATCH_SIZE, shuffle=False)
    test_loader = get_loader(X_test, y_test, cfg.BATCH_SIZE, shuffle=False)

    print(f"Dataset: {cfg.DATASET}")
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Input shape: [B, {cfg.NUM_NODES}, {cfg.PERIOD_LENGTH}]")

    # ==================== 2. 模型构建 ====================
    model = BaselineModel(
        input_len=cfg.INPUT_LEN,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        num_nodes=cfg.NUM_NODES,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    ).to(cfg.DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ==================== 3. 训练循环 ====================
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        y_true, y_pred = eval_epoch(model, val_loader, cfg.DEVICE)
        val_metrics = compute_metrics(y_true, y_pred)
        val_acc = val_metrics['accuracy']

        scheduler.step(val_acc)

        # 早停判断
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # 打印日志
        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(
                f"Epoch {epoch+1:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        if patience_counter >= cfg.EARLY_STOPPING:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # ==================== 4. 测试最佳模型 ====================
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred = eval_epoch(model, test_loader, cfg.DEVICE)
    test_metrics = compute_metrics(y_true, y_pred)

    print("\n========== Test Results ==========")
    for k, v in test_metrics.items():
        print(f"  {k.capitalize():12s}: {v:.4f}")
    print("==================================\n")


if __name__ == "__main__":
    main()
