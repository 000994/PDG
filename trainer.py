import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from data_process import process_data
from utils import load_dataset
from model import PDGNet
from build_sample_graph import build_sample_graph
import config as cfg
import argparse
import numpy as np

# 固定随机种子，确保结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def get_label_mask(N, val_ratio):
    idx = torch.randperm(N)
    val_num = int(N * val_ratio)
    train_mask = torch.zeros(N).bool()
    train_mask[idx[val_num:]] = True
    val_mask = torch.zeros(N).bool()
    val_mask[idx[:val_num]] = True
    return train_mask, val_mask


def main():
    # ===================== 新增 =====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    # 1. 加载并预处理数据
    if args.data is not None:
        # 从指定路径加载
        X_train = np.load(f"{args.data}/X_train.npy")
        y_train = np.load(f"{args.data}/y_train.npy")
        X_test = np.load(f"{args.data}/X_test.npy")
        y_test = np.load(f"{args.data}/y_test.npy")
    else:
        # 原来的默认方式
        X_train, y_train, X_test, y_test = load_dataset(cfg.DATASET, cfg.DATA_DIR)

    X_train, X_test = process_data(X_train, X_test)
    X_train = torch.tensor(X_train).float().to(cfg.DEVICE)
    y_train = torch.tensor(y_train).long().to(cfg.DEVICE).squeeze()
    X_test = torch.tensor(X_test).float().to(cfg.DEVICE)
    y_test = torch.tensor(y_test).long().to(cfg.DEVICE).squeeze()

    # 2. 初始化模型
    model = PDGNet().to(cfg.DEVICE)
    train_mask, val_mask = get_label_mask(len(X_train), cfg.VAL_RATIO)
    criterion = nn.CrossEntropyLoss()

    # ========== 阶段一：预训练（单位矩阵，让特征提取器快速收敛） ==========
    print("=== Phase 1: Pre-training ===")
    eye_adj = torch.eye(len(X_train)).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    best_pre = 0
    for ep in range(30):
        model.train()
        logits = model(X_train, eye_adj)
        loss = criterion(logits[train_mask], y_train[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_train, eye_adj)
            pred_val = logits_val[val_mask].argmax(1)
            acc_val = accuracy_score(y_train[val_mask].cpu(), pred_val.cpu())
        if acc_val > best_pre:
            best_pre = acc_val
            torch.save(model.state_dict(), 'best_pretrain.pth')
        if (ep + 1) % 10 == 0:
            print(f'Ep {ep + 1} | Loss {loss.item():.2f} | Val {acc_val:.2f}')

    model.load_state_dict(torch.load('best_pretrain.pth'))

    # ========== 阶段二：端到端训练（每轮动态重建样本图） ==========
    print("\n=== Phase 2: End-to-end training with adaptive DTW graph ===")
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR * 0.3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=cfg.EARLY_STOP_DELTA
    )
    best = 0
    patience = 0
    min_delta = cfg.EARLY_STOP_DELTA

    for ep in range(cfg.EPOCHS):
        # 每轮基于当前特征重建样本图（动态图更新）
        model.eval()
        with torch.no_grad():
            xf_train = model(X_train)
        sample_adj = build_sample_graph(xf_train, cfg.K_SAMPLE)

        model.train()
        logits = model(X_train, sample_adj)
        loss = criterion(logits[train_mask], y_train[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_train, sample_adj)
            pred_val = logits_val[val_mask].argmax(1)
            acc_val = accuracy_score(y_train[val_mask].cpu(), pred_val.cpu())

        # 早停与学习率调度
        if acc_val > best + min_delta:
            best = acc_val
            patience = 0
            torch.save(model.state_dict(), 'best.pth')
        else:
            patience += 1

        scheduler.step(acc_val)

        if (ep + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Ep {ep + 1} | Loss {loss.item():.2f} | Val {acc_val:.2f} | '
                  f'LR {current_lr:.1e} | Patience {patience}/{cfg.PATIENCE}')

        if patience >= cfg.PATIENCE:
            print(f"Early stopping triggered at epoch {ep + 1} (best val acc: {best:.4f})")
            break

    # ========== 测试 ==========
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    with torch.no_grad():
        xf_train = model(X_train)
        xf_test = model(X_test)
        xf_all = torch.cat([xf_train, xf_test])
        adj_all = build_sample_graph(xf_all, cfg.K_SAMPLE)
        logits_all = model(torch.cat([X_train, X_test]), adj_all)
        logits_test = logits_all[len(X_train):]
        pred = logits_test.argmax(1).cpu()
    yt = y_test.cpu()
    print(f'\nTest Acc: {accuracy_score(yt, pred):.2f}')
    print(f'Test F1:  {f1_score(yt, pred, average="macro"):.2f}')


if __name__ == '__main__':
    main()
