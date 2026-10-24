"""Isolated FT ablation trainer, derived from the unchanged baseline trainer.

Only the train-only class-weight exponent and explicit metric label set differ.
Epoch selection, scheduler, loader, optimizer and recovery follow the baseline.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

BASELINE_TRAINER_SHA256 = (
    "b16153373470383138da703c5a70b9161451ade9f8deea0c44f42ac4b0326c29"
)


def get_device(verbose: bool = True) -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if dev.type == "cuda":
            print(f"[thiết bị] {torch.cuda.get_device_name(0)}")
        else:
            print(
                f"[thiết bị] CPU ({torch.get_num_threads()} luồng) - "
                f"không có GPU, huấn luyện sẽ chậm hơn đáng kể"
            )
    return dev


def compute_class_weights(
    y_train: np.ndarray, n_classes: int, exponent: float = 1.0
) -> torch.Tensor:
    """[N_train / (C * n_c)] ** exponent, computed only from training labels."""
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("Class-weight exponent must be positive and finite.")
    counts = np.bincount(y_train, minlength=n_classes).astype("float64")
    # Lớp vắng mặt trong train: đặt trọng số 0 để không sinh ra inf.
    w = np.zeros_like(counts)
    nz = counts > 0
    w[nz] = (len(y_train) / (n_classes * counts[nz])) ** exponent
    return torch.tensor(w, dtype=torch.float32)


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = 42
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=0,  # 0 để tránh lỗi spawn trên Windows và Colab notebook
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (nhãn dự đoán, ma trận xác suất) - xác suất cần cho AUC/PR-AUC."""
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    proba = np.concatenate(probs, axis=0)
    return proba.argmax(axis=1), proba


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    train_cfg: dict,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
    tag: str = "",
    checkpoint_path: str | Path | None = None,
    checkpoint_context: dict | None = None,
    deadline_monotonic: float | None = None,
) -> dict:
    """Huấn luyện một mô hình PyTorch. Trả về dict gồm history, best_epoch, thời gian.

    history có đúng các khóa mà report_export.plot_training_history() và
    training_summary_table() mong đợi:
        train_loss, val_loss, train_macro_f1, val_macro_f1
    """
    device = device or get_device(verbose=verbose)
    model = model.to(device)

    t = train_cfg
    bs = t["batch_size"]

    # --- Hàm mất mát ---
    if t["imbalance"] == "class_weight":
        w = compute_class_weights(y_train, n_classes, t["class_weight_exponent"]).to(
            device
        )
        criterion = nn.CrossEntropyLoss(weight=w)
        if verbose:
            print(
                f"\n  Trọng số lớp: [N_train / (C * n_c)] ** {t['class_weight_exponent']}:"
            )
            counts = np.bincount(y_train, minlength=n_classes)
            for c in range(n_classes):
                print(f"    lớp {c}: {counts[c]:>9,} mẫu  ->  w = {w[c].item():8.3f}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t["lr_factor"], patience=t["lr_patience"]
    )

    train_loader = make_loader(X_train, y_train, bs, shuffle=True, seed=seed)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "val_macro_f1_smoothed": [],
        "lr": [],
    }
    best_score, best_smoothed, best_epoch, best_state, bad_epochs = (
        -1.0,
        -1.0,
        0,
        None,
        0,
    )
    t0 = time.monotonic()
    elapsed_before, start_epoch = 0.0, 1
    recovery_contract = None
    if checkpoint_path is not None:
        from flow_based.training import recovery

        checkpoint_path = Path(checkpoint_path)
        recovery_contract = recovery.contract(
            model,
            (X_train, y_train, X_val, y_val),
            t,
            seed,
            n_classes,
            device,
            checkpoint_context,
        )
        if checkpoint_path.is_file():
            state = recovery.load(
                checkpoint_path,
                recovery_contract,
                model,
                optimizer,
                scheduler,
                train_loader,
            )
            history = state["history"]
            best_score, best_smoothed = state["best_score"], state["best_smoothed"]
            best_epoch, best_state = state["best_epoch"], state["best_state"]
            bad_epochs, elapsed_before = state["bad_epochs"], state["elapsed_s"]
            start_epoch = state["epoch"] + 1
            if state["finished"]:
                start_epoch = t["max_epochs"] + 1
            if verbose:
                print(
                    f"[recovery] saved epoch {state['epoch']}; next epoch {start_epoch}",
                    flush=True,
                )

    # Cửa sổ trung bình trượt cho quyết định dừng sớm - xem giải thích ở
    # TRAIN["early_stop_smoothing"] trong config.py. smoothing=1 tương đương
    # hành vi cũ (không làm mượt): smoothed_f1 == val_f1 mỗi epoch.
    smoothing = max(1, int(t.get("early_stop_smoothing", 1)))

    if verbose:
        print(
            f"\n  {'Epoch':<8}{'loss':>9}{'val_loss':>10}"
            f"{'train_F1':>10}{'val_F1':>9}{'lr':>10}"
        )
        print("  " + "-" * 56)

    paused = False
    for epoch in range(start_epoch, t["max_epochs"] + 1):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            paused = True
            if verbose:
                print(
                    "[session] Time budget reached; retain recovery/output to continue.",
                    flush=True,
                )
            break
        epoch_started = last_progress = time.monotonic()
        # ---------------- huấn luyện ----------------
        model.train()
        total_loss, n_seen = 0.0, 0
        tr_pred, tr_true = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if t.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()

            total_loss += loss.item() * len(yb)
            n_seen += len(yb)
            tr_pred.append(logits.detach().argmax(1).cpu().numpy())
            tr_true.append(yb.cpu().numpy())
            if verbose and time.monotonic() - last_progress >= 60:
                print(
                    f"  [{tag}] epoch {epoch}/{t['max_epochs']}: "
                    f"{n_seen:,}/{len(y_train):,} mẫu train "
                    f"({time.monotonic() - epoch_started:.0f}s)",
                    flush=True,
                )
                last_progress = time.monotonic()

        train_loss = total_loss / n_seen
        train_f1 = f1_score(
            np.concatenate(tr_true),
            np.concatenate(tr_pred),
            labels=np.arange(n_classes),
            average="macro",
            zero_division=0,
        )

        # ---------------- kiểm định ----------------
        model.eval()
        with torch.no_grad():
            val_loss_sum, n_val = 0.0, 0
            for i in range(0, len(X_val), 1024):
                xb = torch.from_numpy(X_val[i : i + 1024]).to(device)
                yb = torch.from_numpy(y_val[i : i + 1024]).to(device)
                val_loss_sum += criterion(model(xb), yb).item() * len(yb)
                n_val += len(yb)
            val_loss = val_loss_sum / n_val
        val_pred, _ = predict(model, X_val, device)
        val_f1 = f1_score(
            y_val,
            val_pred,
            labels=np.arange(n_classes),
            average="macro",
            zero_division=0,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(float(train_f1))
        history["val_macro_f1"].append(float(val_f1))
        history["lr"].append(lr_now)
        history.setdefault("epoch_time_s", []).append(time.monotonic() - epoch_started)

        # Trung bình trượt K epoch gần nhất - chỉ dùng để QUYẾT ĐỊNH dừng sớm
        # và giảm lr, KHÔNG thay thế giá trị val_macro_f1 thô vừa ghi ở trên.
        # smoothing=1 -> smoothed_f1 luôn bằng val_f1 (hành vi cũ, không đổi).
        smoothed_f1 = float(np.mean(history["val_macro_f1"][-smoothing:]))
        history["val_macro_f1_smoothed"].append(smoothed_f1)

        scheduler.step(smoothed_f1)

        # ---------------- dừng sớm theo macro-F1 đã làm mượt ----------------
        improved = smoothed_f1 > best_smoothed
        if improved:
            best_smoothed, best_epoch, bad_epochs = smoothed_f1, epoch, 0
            best_score = float(val_f1)  # giá trị THÔ tại epoch tốt nhất - để báo cáo
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        if verbose:
            mark = "  <- tốt nhất" if improved else ""
            print(
                f"  {epoch:>3}/{t['max_epochs']:<4}{train_loss:>9.4f}"
                f"{val_loss:>10.4f}{train_f1:>10.4f}{val_f1:>9.4f}"
                f"{lr_now:>10.2e}{mark}"
            )

        if checkpoint_path is not None:
            recovery.save(
                checkpoint_path,
                {
                    "contract": recovery_contract,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_score": best_score,
                    "best_smoothed": best_smoothed,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "bad_epochs": bad_epochs,
                    "elapsed_s": elapsed_before + time.monotonic() - t0,
                    "finished": (
                        bad_epochs >= t["early_stop_patience"]
                        or epoch == t["max_epochs"]
                    ),
                },
                train_loader,
            )

        if bad_epochs >= t["early_stop_patience"]:
            if verbose:
                print(
                    f"  Dừng sớm - {t['early_stop_patience']} epoch liên tiếp "
                    f"không cải thiện val_macro_f1."
                )
            break

    total_time = elapsed_before + time.monotonic() - t0

    # Nạp lại trọng số của epoch tốt nhất TRƯỚC khi đánh giá test.
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(
            f"\n  Huấn luyện {total_time:.1f}s. "
            f"Tốt nhất: epoch {best_epoch} (val macro-F1 {best_score:.4f})"
        )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "total_time_s": total_time,
        "n_epochs": len(history["train_loss"]),
        "device": str(device),
        "tag": tag,
        "paused": paused,
    }
