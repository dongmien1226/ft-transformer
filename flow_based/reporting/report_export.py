"""
report_export.py
================
Module sinh toàn bộ bảng biểu và hình vẽ cho Chương 3 của đồ án.

Nguyên tắc thiết kế:
    Mọi con số xuất hiện trong báo cáo đều phải đi qua module này.
    Không nhập số liệu thủ công vào file Word. Mỗi bảng trong Chương 3
    có đúng một hàm sinh tương ứng (xem Phụ lục 3.A của chương).

Cách dùng trong Google Colab:
    !cp /content/drive/MyDrive/DATN/report_export.py .
    from report_export import *
    set_global_seed(42)
    store = ResultStore(out_dir="/content/drive/MyDrive/DATN/ketqua")

Phụ thuộc bắt buộc: numpy, pandas, scikit-learn, matplotlib
Phụ thuộc tùy chọn: torch, psutil  (module vẫn chạy được nếu thiếu)
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None
    _HAS_PSUTIL = False

matplotlib.rcParams["figure.dpi"] = 130
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["font.size"] = 10

__all__ = [
    "set_global_seed",
    "seed_worker",
    "collect_hardware_info",
    "collect_software_info",
    "label_distribution_table",
    "split_summary_table",
    "model_config_table",
    "count_parameters",
    "training_summary_table",
    "evaluate_predictions",
    "per_class_table",
    "comparison_table",
    "measure_inference_time",
    "aggregate_seeds",
    "plot_training_history",
    "plot_model_comparison",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_confusion_matrix",
    "ResultStore",
    "save_table",
    "to_vn_decimal",
]


# =============================================================================
# 1. TÁI LẬP  ->  Bảng 3.3
# =============================================================================


def set_global_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Cố định mọi nguồn ngẫu nhiên. Gọi hàm này TRƯỚC mọi thao tác khác.

    deterministic=True buộc cuDNN dùng thuật toán tất định. Việc này làm chậm
    huấn luyện khoảng 5-15% nhưng loại bỏ được sai khác giữa các lần chạy.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    print(f"[seed] Đã cố định seed = {seed} (deterministic={deterministic})")
    return seed


def seed_worker(worker_id: int) -> None:
    """Truyền vào DataLoader(worker_init_fn=seed_worker) để ổn định thứ tự nạp dữ liệu."""
    worker_seed = (torch.initial_seed() % 2**32) if _HAS_TORCH else 0
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# 2. MÔI TRƯỜNG  ->  Bảng 3.1, Bảng 3.2
# =============================================================================


def _run_cmd(cmd: str) -> str:
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip()
    except Exception:
        return ""


def collect_hardware_info() -> pd.DataFrame:
    """Bảng 3.1 — Cấu hình phần cứng, ghi nhận tại thời điểm chạy."""
    rows: List[Dict[str, str]] = []

    def add(k: str, v: Any) -> None:
        rows.append(
            {
                "Thành phần": k,
                "Thông số": str(v) if v not in (None, "") else "Không xác định",
            }
        )

    in_colab = "google.colab" in sys.modules
    add(
        "Nền tảng",
        "Google Colab" if in_colab else f"{platform.system()} {platform.release()}",
    )

    cpu_name = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        cpu_name = platform.processor()
    add("Bộ xử lý (CPU)", cpu_name or platform.processor())

    n_logical = os.cpu_count()
    n_physical = psutil.cpu_count(logical=False) if _HAS_PSUTIL else None
    add(
        "Số nhân CPU",
        f"{n_physical} nhân vật lý / {n_logical} luồng"
        if n_physical
        else f"{n_logical} luồng",
    )

    if _HAS_PSUTIL:
        add("Bộ nhớ chính (RAM)", f"{psutil.virtual_memory().total / 1024**3:.2f} GB")
    else:
        add("Bộ nhớ chính (RAM)", _run_cmd("free -h | awk '/Mem:/ {print $2}'"))

    if _HAS_TORCH and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        add("Bộ xử lý đồ họa (GPU)", props.name)
        add("Bộ nhớ GPU", f"{props.total_memory / 1024**3:.2f} GB")
        add("Compute capability", f"{props.major}.{props.minor}")
        add("Phiên bản CUDA (PyTorch)", torch.version.cuda)
        drv = _run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        add("Phiên bản driver NVIDIA", drv)
    else:
        add("Bộ xử lý đồ họa (GPU)", "Không sử dụng GPU")
        add("Bộ nhớ GPU", "—")

    disk = _run_cmd('df -h / | awk \'NR==2 {print $4" khả dụng / "$2" tổng"}\'')
    add("Dung lượng đĩa", disk)
    add("Thời điểm ghi nhận", time.strftime("%Y-%m-%d %H:%M:%S"))

    return pd.DataFrame(rows)


def collect_software_info() -> pd.DataFrame:
    """Bảng 3.2 — Phiên bản các thư viện. Chỉ liệt kê thư viện thực sự cài được."""
    roles = {
        "Python": "Ngôn ngữ triển khai",
        "PyTorch": "Xây dựng và huấn luyện mô hình Transformer, MLP",
        "CUDA": "Tăng tốc tính toán trên GPU",
        "cuDNN": "Thư viện nhân tích chập và tối ưu GPU",
        "NumPy": "Tính toán trên mảng số",
        "pandas": "Đọc và xử lý dữ liệu dạng bảng",
        "scikit-learn": "Mô hình cơ sở, chia dữ liệu, tính độ đo",
        "XGBoost": "Mô hình cơ sở dựa trên cây tăng cường",
        "Matplotlib": "Vẽ biểu đồ kết quả",
    }
    versions: Dict[str, Optional[str]] = {"Python": platform.python_version()}

    if _HAS_TORCH:
        versions["PyTorch"] = torch.__version__
        versions["CUDA"] = (
            torch.version.cuda if torch.cuda.is_available() else "Không dùng"
        )
        try:
            versions["cuDNN"] = str(torch.backends.cudnn.version())
        except Exception:
            versions["cuDNN"] = None

    versions["NumPy"] = np.__version__
    versions["pandas"] = pd.__version__
    import sklearn

    versions["scikit-learn"] = sklearn.__version__
    versions["Matplotlib"] = matplotlib.__version__

    try:
        import xgboost

        versions["XGBoost"] = xgboost.__version__
    except Exception:
        versions["XGBoost"] = None

    rows = [
        {"Thành phần": k, "Phiên bản": v, "Vai trò trong thực nghiệm": roles.get(k, "")}
        for k, v in versions.items()
        if v is not None
    ]
    return pd.DataFrame(rows)


# =============================================================================
# 3. DỮ LIỆU  ->  Bảng 3.4, Bảng 3.5
# =============================================================================


def label_distribution_table(
    y: Sequence, class_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Bảng 3.4 — Phân bố nhãn sau tiền xử lý, kèm tỷ số mất cân bằng."""
    s = pd.Series(np.asarray(y).ravel())
    counts = s.value_counts().sort_index()
    total = int(counts.sum())

    names = (
        list(class_names) if class_names is not None else [str(i) for i in counts.index]
    )
    df = pd.DataFrame(
        {
            "Nhãn lớp": names,
            "Số mẫu": counts.values.astype(int),
            "Tỷ lệ (%)": np.round(counts.values / total * 100, 4),
        }
    )
    df = df.sort_values("Số mẫu", ascending=False).reset_index(drop=True)

    ir = counts.max() / counts.min()
    df.loc[len(df)] = ["Tổng cộng", total, 100.0]

    print(f"[dữ liệu] Tổng số mẫu: {total:,}")
    print(f"[dữ liệu] Số lớp: {len(counts)}")
    print(f"[dữ liệu] Tỷ số mất cân bằng (lớp đông nhất / lớp thưa nhất): {ir:.2f} : 1")
    print(f"[dữ liệu] --> Điền giá trị {ir:.2f} vào Mục 3.2.1 của Chương 3")
    df.attrs["imbalance_ratio"] = float(ir)
    return df


def split_summary_table(
    y_train, y_val, y_test, class_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Bảng 3.5 — Số mẫu từng lớp trong ba tập, kèm kiểm tra phân tầng."""
    parts = {"Tập huấn luyện": y_train, "Tập kiểm định": y_val, "Tập kiểm tra": y_test}
    all_labels = sorted(
        set(np.concatenate([np.asarray(v).ravel() for v in parts.values()]).tolist())
    )
    names = (
        list(class_names) if class_names is not None else [str(i) for i in all_labels]
    )

    data = {"Lớp": names}
    for pname, yv in parts.items():
        c = pd.Series(np.asarray(yv).ravel()).value_counts()
        data[pname] = [int(c.get(l, 0)) for l in all_labels]

    df = pd.DataFrame(data)
    df["Tổng"] = df[list(parts.keys())].sum(axis=1)

    totals = {"Lớp": "Tổng cộng"}
    for c in list(parts.keys()) + ["Tổng"]:
        totals[c] = int(df[c].sum())
    df.loc[len(df)] = totals

    grand = totals["Tổng"]
    ratios = {"Lớp": "Tỷ lệ (%)"}
    for c in list(parts.keys()) + ["Tổng"]:
        ratios[c] = round(totals[c] / grand * 100, 2)
    df.loc[len(df)] = ratios

    # Cảnh báo nếu một lớp vắng mặt ở tập kiểm tra -> độ đo theo lớp sẽ không xác định
    for i, name in enumerate(names):
        if df.loc[i, "Tập kiểm tra"] == 0:
            print(
                f"[CẢNH BÁO] Lớp '{name}' không có mẫu nào trong tập kiểm tra. "
                f"Precision/Recall của lớp này sẽ không xác định."
            )
    return df


# =============================================================================
# 4. CẤU HÌNH MÔ HÌNH  ->  Bảng 3.6, Bảng 3.7
# =============================================================================


def model_config_table(
    config: Dict[str, Any], group_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Bảng 3.6 / 3.7 — Trích siêu tham số TRỰC TIẾP từ dict cấu hình đang chạy.

    Không gõ lại giá trị vào Word. Truyền chính đối tượng config đã dùng để
    khởi tạo mô hình, nhờ đó bảng luôn khớp với lần chạy sinh ra kết quả.
    """
    rows = []
    for k, v in config.items():
        if isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False)
        rows.append(
            {
                "Nhóm": (group_map or {}).get(k, ""),
                "Tham số": k,
                "Giá trị": str(v),
            }
        )
    return pd.DataFrame(rows)


def count_parameters(model) -> Dict[str, int]:
    """Đếm số tham số. Dùng cho Bảng 3.7 và Bảng 3.17."""
    if not _HAS_TORCH or not hasattr(model, "parameters"):
        return {"total": -1, "trainable": -1}
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[mô hình] Tổng tham số: {total:,} | Khả huấn luyện: {trainable:,}")
    return {"total": int(total), "trainable": int(trainable)}


def training_summary_table(
    history: Dict[str, List[float]],
    best_epoch: int,
    total_time_s: float,
    patience: int,
    monitor: str = "val_macro_f1",
) -> pd.DataFrame:
    """Bảng 3.11 — Tóm tắt quá trình huấn luyện, lấy từ history đã lưu."""
    n_epochs = len(history.get("train_loss", []))
    avg_epoch = total_time_s / n_epochs if n_epochs else float("nan")

    def at(key, idx):
        arr = history.get(key)
        return f"{arr[idx]:.4f}" if arr and 0 <= idx < len(arr) else "—"

    rows = [
        (
            "Số epoch đã chạy",
            str(n_epochs),
            "Dừng sớm được kích hoạt" if n_epochs else "",
        ),
        (
            "Epoch cho kết quả kiểm định tốt nhất",
            str(best_epoch),
            "Trọng số được lưu tại epoch này",
        ),
        (
            "Hàm mất mát kiểm định tại checkpoint được chọn",
            at("val_loss", best_epoch - 1),
            "Không nhất thiết là loss nhỏ nhất",
        ),
        (
            f"Chỉ số tại checkpoint được chọn ({monitor})",
            at(monitor, best_epoch - 1),
            "Checkpoint chọn theo Macro-F1 làm mượt",
        ),
        ("Số epoch chờ trước khi dừng sớm", str(patience), ""),
        ("Thời gian trung bình mỗi epoch", f"{avg_epoch:.2f} giây", "Đo thực tế"),
        (
            "Tổng thời gian huấn luyện",
            f"{total_time_s:.2f} giây ({total_time_s / 60:.2f} phút)",
            "Đo thực tế",
        ),
    ]
    return pd.DataFrame(rows, columns=["Chỉ số", "Giá trị", "Ghi chú"])


# =============================================================================
# 5. ĐÁNH GIÁ  ->  Bảng 3.12 - 3.17
# =============================================================================


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba=None,
    class_names: Optional[Sequence[str]] = None,
    positive_label: Optional[int] = None,
) -> Dict[str, Any]:
    """Tính toàn bộ độ đo cho MỘT mô hình trên MỘT tập dữ liệu.

    y_proba: ma trận (n_samples, n_classes) hoặc vector xác suất lớp dương.
             Nếu None thì AUC-ROC và PR-AUC trả về NaN thay vì bịa giá trị.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)
    is_binary = n_classes <= 2

    res: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
        "task": "nhị phân" if is_binary else "đa lớp",
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0, labels=labels
        )
        res[f"precision_{avg}"] = float(p)
        res[f"recall_{avg}"] = float(r)
        res[f"f1_{avg}"] = float(f)

    # Độ đo theo từng lớp
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    names = list(class_names) if class_names is not None else [str(l) for l in labels]
    res["per_class"] = {
        names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(sup[i]),
        }
        for i in range(len(labels))
    }

    # AUC-ROC và PR-AUC — chỉ tính khi có xác suất dự đoán
    res["roc_auc_macro"] = float("nan")
    res["roc_auc_weighted"] = float("nan")
    res["pr_auc_macro"] = float("nan")
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            if is_binary:
                score = (
                    y_proba[:, 1]
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2
                    else y_proba.ravel()
                )
                auc = float(roc_auc_score(y_true, score))
                res["roc_auc_macro"] = res["roc_auc_weighted"] = auc
                res["pr_auc_macro"] = float(average_precision_score(y_true, score))
            else:
                # Chiến lược một-đối-phần-còn-lại, nêu rõ trong Mục 3.2.5
                res["roc_auc_macro"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=labels,
                    )
                )
                res["roc_auc_weighted"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                        labels=labels,
                    )
                )
                y_bin = np.zeros_like(y_proba)
                for i, l in enumerate(labels):
                    y_bin[:, i] = (y_true == l).astype(int)
                res["pr_auc_macro"] = float(
                    average_precision_score(y_bin, y_proba, average="macro")
                )
        except Exception as e:
            print(
                f"[CẢNH BÁO] Không tính được AUC: {e}. Giá trị được ghi là NaN, "
                f"KHÔNG được điền số ước lượng vào báo cáo."
            )

    res["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    res["labels"] = labels.tolist()
    res["class_names"] = names
    return res


def per_class_table(
    y_true, y_pred, class_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Bảng 3.13 — Precision / Recall / F1 / support theo từng lớp."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    names = list(class_names) if class_names is not None else [str(l) for l in labels]

    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for n in names:
        d = rep[n]
        rows.append(
            {
                "Lớp": n,
                "Precision": round(d["precision"], 4),
                "Recall": round(d["recall"], 4),
                "F1-score": round(d["f1-score"], 4),
                "Số mẫu (support)": int(d["support"]),
            }
        )
    for key, label in (
        ("macro avg", "Trung bình vĩ mô"),
        ("weighted avg", "Trung bình có trọng số"),
    ):
        d = rep[key]
        rows.append(
            {
                "Lớp": label,
                "Precision": round(d["precision"], 4),
                "Recall": round(d["recall"], 4),
                "F1-score": round(d["f1-score"], 4),
                "Số mẫu (support)": int(d["support"]),
            }
        )
    return pd.DataFrame(rows)


def comparison_table(
    results: Dict[str, Dict[str, Any]],
    extra: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Bảng 3.12 / 3.14 — So sánh nhiều mô hình.

    results: {"Tên mô hình": <dict trả về bởi evaluate_predictions>}
    extra:   {"Tên mô hình": {"Số tham số": ..., "Thời gian huấn luyện (giây)": ...}}
    """
    rows = []
    for name, r in results.items():
        row = {
            "Mô hình": name,
            "Accuracy": round(r["accuracy"], 4),
            "Precision (macro)": round(r["precision_macro"], 4),
            "Recall (macro)": round(r["recall_macro"], 4),
            "F1 (macro)": round(r["f1_macro"], 4),
            "F1 (weighted)": round(r["f1_weighted"], 4),
            "AUC-ROC (macro)": round(r["roc_auc_macro"], 4)
            if not np.isnan(r["roc_auc_macro"])
            else "—",
            "PR-AUC (macro)": round(r["pr_auc_macro"], 4)
            if not np.isnan(r["pr_auc_macro"])
            else "—",
        }
        if extra and name in extra:
            row.update(extra[name])
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values("F1 (macro)", ascending=False)
        .reset_index(drop=True)
    )
    best = df.iloc[0]
    print(
        f"[so sánh] Mô hình đạt macro-F1 cao nhất: {best['Mô hình']} ({best['F1 (macro)']})"
    )
    if len(df) > 1:
        gap = df.iloc[0]["F1 (macro)"] - df.iloc[1]["F1 (macro)"]
        print(f"[so sánh] Chênh lệch so với mô hình đứng thứ hai: {gap:.4f}")
        print(
            "[so sánh] Trung bình và SD mô tả biến động giữa các seed trên cùng "
            "phép chia; riêng chúng chưa chứng minh khác biệt có ý nghĩa thống kê."
        )
    return df


def measure_inference_time(
    model,
    X,
    batch_size: int = 256,
    n_warmup: int = 5,
    n_repeat: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """Bảng 3.17 — Đo thời gian suy luận thực tế.

    Có giai đoạn khởi động để loại bỏ chi phí biên dịch kernel lần đầu, và
    đồng bộ GPU trước khi dừng đồng hồ để tránh đo thiếu.
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(X)

    is_torch = _HAS_TORCH and hasattr(model, "eval")
    if is_torch:
        model.eval().to(device)
        Xt = torch.from_numpy(X).to(device)

        def _forward():
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    model(Xt[i : i + batch_size])
            if device.startswith("cuda"):
                torch.cuda.synchronize()
    else:

        def _forward():
            model.predict(X)

    for _ in range(n_warmup):
        _forward()

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _forward()
        times.append(time.perf_counter() - t0)

    mean_s, std_s = float(np.mean(times)), float(np.std(times))
    out = {
        "Thời gian suy luận (ms/mẫu)": round(mean_s / n * 1000, 5),
        "Độ lệch chuẩn (ms/mẫu)": round(std_s / n * 1000, 5),
        "Thông lượng (mẫu/giây)": round(n / mean_s, 1),
        "Số mẫu đo": n,
        "Số lần lặp": n_repeat,
        "Thiết bị": device if is_torch else "cpu",
    }
    print(
        f"[suy luận] {out['Thời gian suy luận (ms/mẫu)']} ms/mẫu | "
        f"{out['Thông lượng (mẫu/giây)']:,} mẫu/giây trên {out['Thiết bị']}"
    )
    return out


def aggregate_seeds(
    runs: List[Dict[str, Any]],
    metrics: Sequence[str] = (
        "accuracy",
        "f1_macro",
        "recall_macro",
        "f1_weighted",
        "roc_auc_macro",
    ),
) -> pd.DataFrame:
    """Tổng hợp nhiều lần chạy thành 'trung bình ± độ lệch chuẩn'.

    SD mẫu (ddof=1) mô tả biến động giữa seed trên cùng phép chia dữ liệu.
    Một lần chạy chưa đủ để ước lượng SD; hiển thị n<2 thay vì số 0.
    """
    rows = []
    for m in metrics:
        vals = np.array(
            [r[m] for r in runs if m in r and not np.isnan(r[m])], dtype=float
        )
        if len(vals) == 0:
            continue
        rows.append(
            {
                "Độ đo": m,
                "Trung bình": round(float(vals.mean()), 4),
                "Độ lệch chuẩn": round(float(vals.std(ddof=1)), 4)
                if len(vals) > 1
                else np.nan,
                "Nhỏ nhất": round(float(vals.min()), 4),
                "Lớn nhất": round(float(vals.max()), 4),
                "Số lần chạy": len(vals),
                "Báo cáo dạng": (
                    f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
                    if len(vals) > 1
                    else f"{vals.mean():.4f} (n<2)"
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 6. BIỂU ĐỒ  ->  Hình 3.1 - 3.7
# =============================================================================


def plot_training_history(
    history: Dict[str, List[float]],
    out_path: str,
    metric_key: str = "macro_f1",
    metric_label: str = "Macro-F1",
    best_epoch: Optional[int] = None,
) -> str:
    """Hình 3.1 và Hình 3.2 — Diễn biến loss và metric theo epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Tập huấn luyện", linewidth=1.6)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Tập kiểm định", linewidth=1.6)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Hàm mất mát")
    axes[0].set_title("Diễn biến hàm mất mát")

    tk, vk = f"train_{metric_key}", f"val_{metric_key}"
    if tk in history:
        axes[1].plot(epochs, history[tk], label="Tập huấn luyện", linewidth=1.6)
    if vk in history:
        axes[1].plot(epochs, history[vk], label="Tập kiểm định", linewidth=1.6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_label)
    axes[1].set_title(f"Diễn biến {metric_label}")

    for ax in axes:
        if best_epoch:
            ax.axvline(
                best_epoch,
                color="crimson",
                linestyle="--",
                linewidth=1.1,
                label=f"Epoch tốt nhất ({best_epoch})",
            )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[hình] Đã lưu: {out_path}")
    return out_path


def plot_model_comparison(
    df: pd.DataFrame,
    out_path: str,
    metric: str = "F1 (macro)",
    err_col: Optional[str] = None,
) -> str:
    """Hình 3.3 — Biểu đồ cột so sánh, kèm thanh sai số nếu có nhiều lần chạy."""
    d = df[df["Mô hình"].notna()].copy().sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(d))))
    err = d[err_col] if (err_col and err_col in d.columns) else None
    bars = ax.barh(
        d["Mô hình"], d[metric], xerr=err, capsize=4, color="#4C72B0", alpha=0.9
    )
    bars[-1].set_color("#C44E52")  # làm nổi mô hình tốt nhất
    for b, v in zip(bars, d[metric]):
        ax.text(
            b.get_width() + 0.005,
            b.get_y() + b.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel(metric)
    ax.set_xlim(0, min(1.08, d[metric].max() * 1.18))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[hình] Đã lưu: {out_path}")
    return out_path


def plot_confusion_matrix(
    cm,
    class_names: Sequence[str],
    out_path: str,
    normalize: bool = False,
    title: str = "",
) -> str:
    """Hình 3.6 và Hình 3.7 — Ma trận nhầm lẫn, dạng tuyệt đối hoặc chuẩn hóa theo hàng."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.35 * n + 3, 1.15 * n + 2.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.045)

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            txt = f"{cm[i, j]:.3f}" if normalize else f"{int(cm[i, j]):,}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title(
        title
        or (
            "Ma trận nhầm lẫn chuẩn hóa theo lớp thực tế"
            if normalize
            else "Ma trận nhầm lẫn (số tuyệt đối)"
        )
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[hình] Đã lưu: {out_path}")
    return out_path


def plot_roc_curves(curves: Dict[str, tuple], out_path: str) -> str:
    """Hình 3.4 — Đường cong ROC. curves = {"Tên mô hình": (y_true, y_score)}."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, (y_true, y_score) in curves.items():
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(
            fpr,
            tpr,
            linewidth=1.6,
            label=f"{name} (AUC = {roc_auc_score(y_true, y_score):.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="Phân loại ngẫu nhiên")
    ax.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương tính thật (TPR)")
    ax.set_title("Đường cong ROC")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[hình] Đã lưu: {out_path}")
    return out_path


def plot_pr_curves(curves: Dict[str, tuple], out_path: str) -> str:
    """Hình 3.5 — Đường cong Precision-Recall, nhạy hơn ROC với lớp thưa."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    baseline = None
    for name, (y_true, y_score) in curves.items():
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, linewidth=1.6, label=f"{name} (AP = {ap:.4f})")
        baseline = y_true.mean()
    if baseline is not None:
        ax.axhline(
            baseline,
            color="k",
            linestyle="--",
            linewidth=0.9,
            label=f"Mức cơ sở ({baseline:.4f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Đường cong Precision–Recall")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[hình] Đã lưu: {out_path}")
    return out_path


# =============================================================================
# 7. XUẤT KẾT QUẢ
# =============================================================================


def to_vn_decimal(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Đổi dấu thập phân sang dấu phẩy theo quy chuẩn trình bày tiếng Việt."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(
                lambda v: f"{v:.{decimals}f}".replace(".", ",") if pd.notna(v) else "—"
            )
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:,}".replace(",", "."))
    return out


def save_table(
    df: pd.DataFrame, name: str, out_dir: str, vn_format: bool = True
) -> Dict[str, str]:
    """Lưu một bảng ra CSV (để lưu trữ) và Markdown (để dán vào Word)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    paths["csv"] = csv_path

    shown = to_vn_decimal(df) if vn_format else df
    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(shown.to_markdown(index=False, disable_numparse=True))
    paths["md"] = md_path
    print(f"[bảng] Đã lưu: {csv_path}")
    return paths


@dataclass
class ResultStore:
    """Gom toàn bộ kết quả của một phiên thực nghiệm vào một nơi duy nhất.

    Mỗi lần chạy sinh ra một thư mục con chứa: cấu hình, số liệu thô dạng JSON,
    các bảng CSV và các hình PNG. Nhờ đó mọi con số trong báo cáo đều truy vết
    được về đúng lần chạy đã tạo ra nó.
    """

    out_dir: str
    run_name: str = field(default_factory=lambda: time.strftime("run_%Y%m%d_%H%M%S"))
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.run_dir = os.path.join(self.out_dir, self.run_name)
        os.makedirs(os.path.join(self.run_dir, "bang"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "hinh"), exist_ok=True)
        print(f"[store] Thư mục kết quả: {self.run_dir}")

    def add_table(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        self.tables[name] = df
        save_table(df, name, os.path.join(self.run_dir, "bang"))
        return df

    def add_metrics(self, model_name: str, result: Dict[str, Any]) -> None:
        self.metrics[model_name] = result

    def fig_path(self, name: str) -> str:
        return os.path.join(self.run_dir, "hinh", f"{name}.png")

    def finalize(
        self,
        hardware: Optional[pd.DataFrame] = None,
        software: Optional[pd.DataFrame] = None,
    ) -> str:
        """Ghi manifest tổng hợp. Gọi ở cuối notebook."""
        if hardware is not None:
            self.add_table("bang_3_1_phan_cung", hardware)
        if software is not None:
            self.add_table("bang_3_2_phan_mem", software)

        manifest = {
            "run_name": self.run_name,
            "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "cac_bang_da_sinh": sorted(self.tables.keys()),
            "cac_mo_hinh_da_danh_gia": sorted(self.metrics.keys()),
        }
        with open(
            os.path.join(self.run_dir, "manifest.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(self.run_dir, "metrics_raw.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2, default=str)

        print("\n" + "=" * 66)
        print(f"HOÀN TẤT. Kết quả tại: {self.run_dir}")
        print(f"  - Số bảng đã sinh : {len(self.tables)}")
        print(f"  - Số mô hình      : {len(self.metrics)}")
        print("  - Đối chiếu Phụ lục 3.A của Chương 3 để kiểm tra còn thiếu bảng nào.")
        print("=" * 66)
        return self.run_dir


if __name__ == "__main__":
    print("Kiểm tra nhanh module trên dữ liệu giả lập (chỉ để xác nhận mã chạy được).")
    set_global_seed(42)
    print(collect_hardware_info().to_string(index=False))
    print()
    print(collect_software_info().to_string(index=False))
