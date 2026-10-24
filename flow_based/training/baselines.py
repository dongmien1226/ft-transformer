"""
baselines.py - Kịch bản KB1: bốn mô hình cơ sở (Chương 3, mục 3.3.1).

Mục tiêu của KB1 KHÔNG phải là tìm mô hình tốt nhất, mà là xác lập mốc tham
chiếu để trả lời câu hỏi: kiến trúc Transformer có thực sự mang lại lợi ích so
với những phương pháp đơn giản hơn hay không.

Nếu các mô hình cơ sở đơn giản đã đạt mức hiệu năng gần bão hòa thì dư địa cải
thiện của mô hình đề xuất rất hẹp, và điều đó phải được nêu thẳng trong phần
phân tích thay vì lờ đi.

Bốn mô hình đại diện bốn nhóm phương pháp khác nhau (Chương 3, mục 3.2.3):
    Logistic Regression  - tuyến tính
    Random Forest        - tập hợp cây, bagging
    XGBoost              - tập hợp cây, boosting
    MLP                  - mạng nơ-ron truyền thẳng

MLP đóng vai trò đặc biệt: nó dùng CHUNG vòng lặp huấn luyện với Transformer,
nên chênh lệch giữa MLP và Transformer quy được về đúng một nguyên nhân là
cơ chế tự chú ý.

Điều kiện kiểm soát: cùng phép chia, cùng bộ chuẩn hóa đã khớp trên train,
cùng hạt giống, cùng tập kiểm tra, cùng hệ độ đo.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from flow_based import config as C
from flow_based.models.mlp_baseline import build_mlp
from flow_based.reporting.report_export import set_global_seed
from flow_based.training.trainer import get_device, predict, train_torch_model

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _fit_sklearn(model, name, X_train, y_train, X_test, verbose=True):
    if verbose:
        print(
            f"\n  Đang huấn luyện {name} trên {X_train.shape[0]:,} mẫu "
            f"x {X_train.shape[1]} đặc trưng..."
        )
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    if verbose:
        print(f"  {name} xong sau {dt:.1f}s")
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "train_time_s": dt,
        "model": model,
        "n_params": None,
        "device": "cpu",
    }


def run_logreg(X_train, y_train, X_test, verbose=True, *, seed=C.SEED):
    params = {**C.LOGREG_PARAMS, "random_state": seed}
    return _fit_sklearn(
        LogisticRegression(**params),
        "Logistic Regression",
        X_train,
        y_train,
        X_test,
        verbose,
    )


def run_random_forest(X_train, y_train, X_test, verbose=True, *, seed=C.SEED):
    params = {**C.RF_PARAMS, "random_state": seed}
    return _fit_sklearn(
        RandomForestClassifier(**params),
        "Random Forest",
        X_train,
        y_train,
        X_test,
        verbose,
    )


def run_xgboost(X_train, y_train, X_test, n_classes, verbose=True, *, seed=C.SEED):
    if not HAS_XGB:
        raise ImportError("Cần đủ năm mô hình: cài xgboost bằng pip install xgboost")
    params = {**C.XGB_PARAMS, "random_state": seed}
    # XGBoost không có class_weight; dùng sample_weight tương đương công thức
    # w_c = N / (C * n_c) để điều kiện xử lý mất cân bằng giống các mô hình khác.
    counts = np.bincount(y_train, minlength=n_classes).astype("float64")
    w_per_class = np.zeros_like(counts)
    nz = counts > 0
    w_per_class[nz] = len(y_train) / (n_classes * counts[nz])
    sample_weight = w_per_class[y_train]

    model = XGBClassifier(**params)
    if verbose:
        print(f"\n  Đang huấn luyện XGBoost trên {X_train.shape[0]:,} mẫu...")
    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    dt = time.time() - t0
    if verbose:
        print(f"  XGBoost xong sau {dt:.1f}s")
    return {
        "y_pred": model.predict(X_test),
        "y_proba": model.predict_proba(X_test),
        "train_time_s": dt,
        "model": model,
        "n_params": None,
        "device": "cpu",
    }


def run_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    n_classes,
    seed=C.SEED,
    train_cfg=None,
    verbose=True,
):
    """MLP dùng CHUNG trainer với Transformer - chỉ khác kiến trúc."""
    train_cfg = train_cfg or C.TRAIN
    set_global_seed(seed)  # Phải đặt TRƯỚC khi khởi tạo trọng số MLP.
    device = get_device(verbose=verbose)
    model = build_mlp(X_train.shape[1], n_classes, C.MLP)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(
            f"\n  MLP: {X_train.shape[1]} đặc trưng -> "
            f"{C.MLP['hidden_dims']} -> {n_classes} lớp, {n_params:,} tham số"
        )

    res = train_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        n_classes=n_classes,
        train_cfg=train_cfg,
        seed=seed,
        device=device,
        verbose=verbose,
        tag="MLP",
    )

    y_pred, y_proba = predict(res["model"], X_test, device)
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "train_time_s": res["total_time_s"],
        "model": res["model"],
        "n_params": n_params,
        "history": res["history"],
        "best_epoch": res["best_epoch"],
        "n_epochs": res["n_epochs"],
        "device": res["device"],
    }


def run_all_baselines(
    data: dict,
    task: str,
    seed: int = C.SEED,
    verbose: bool = True,
    train_cfg: dict | None = None,
    on_complete=None,
) -> dict:
    """Chạy cả bốn mô hình cơ sở cho một bài toán ('binary' hoặc 'multi').

    data: dict từ load_dataset() trong run_all.py
    Trả về {tên mô hình: {y_pred, y_proba, train_time_s, ...}}
    """
    if not HAS_XGB:
        raise ImportError("Cần đủ năm mô hình: cài xgboost bằng pip install xgboost")
    key = "bin" if task == "binary" else "multi"
    y_train = data[f"y_{key}_train"]
    y_val = data[f"y_{key}_val"]
    n_classes = 2 if task == "binary" else data["n_classes_multi"]

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]

    print("\n" + "=" * 78)
    print(f"KB1 - MÔ HÌNH CƠ SỞ  ({'nhị phân' if task == 'binary' else 'đa lớp'})")
    print("=" * 78)

    out = {}
    out["Logistic Regression"] = run_logreg(
        X_train, y_train, X_test, verbose, seed=seed
    )
    if on_complete:
        on_complete("Logistic Regression", out["Logistic Regression"])
    out["Random Forest"] = run_random_forest(
        X_train, y_train, X_test, verbose, seed=seed
    )
    if on_complete:
        on_complete("Random Forest", out["Random Forest"])
    xgb = run_xgboost(X_train, y_train, X_test, n_classes, verbose, seed=seed)
    if xgb is not None:
        out["XGBoost"] = xgb
        if on_complete:
            on_complete("XGBoost", xgb)
    out["MLP"] = run_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        n_classes,
        seed=seed,
        train_cfg=train_cfg,
        verbose=verbose,
    )
    if on_complete:
        on_complete("MLP", out["MLP"])
    return out
