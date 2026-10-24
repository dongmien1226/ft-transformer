"""
data_prep.py - Tiền xử lý CIC-IDS2017 theo Chương 2 mục 2.3 (các bước B1-B12).

Chạy:  python -m flow_based.data.data_prep

Đầu ra trong data/processed/:
    dataset.npz    - X_train/X_val/X_test (float32)
                     y_multi_* (int64, 4 lớp) và y_bin_* (int64, 2 lớp)
    meta.json      - tên đặc trưng, tên lớp, thống kê làm sạch, cột đã bỏ
    scaler.joblib  - bộ chuẩn hóa đã fit CHỈ TRÊN TẬP TRAIN

THỨ TỰ CÁC BƯỚC LÀ BẤT BIẾN (Chương 2, Bảng 2.6 - sáu biện pháp chống rò rỉ):

    B3  bỏ cột định danh / thời gian   <- chống mô hình học thuộc IP, mốc giờ
    B6  khử trùng lặp TRƯỚC khi chia   <- chống bản ghi có mặt ở cả train và test
    B8  chia phân tầng                 <- giữ tỷ lệ lớp thưa trong cả ba tập
    B9  fit scaler CHỈ trên train      <- chống rò rỉ thống kê của test
    B10 transform cho val/test
    B11 tính trọng số lớp TỪ TRAIN     <- chống rò rỉ phân bố lớp của test

Bài toán nhị phân và bài toán đa lớp DÙNG CHUNG một phép chia, chỉ khác ở bước
mã hóa nhãn (Chương 3, mục 3.2.1a). Nhờ đó kết quả hai bài toán đối chiếu trực
tiếp được với nhau.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from flow_based import config as C
from flow_based.data.labels import assert_web_labels, normalize_label, to_group

warnings.filterwarnings("ignore", category=FutureWarning)

SEP = "=" * 78


# ---------------------------------------------------------------------------
# B1-B2. Đọc dữ liệu thô
# ---------------------------------------------------------------------------
def load_raw(strict_counts: bool = True) -> pd.DataFrame:
    """Đọc các file CSV cần thiết theo cấu hình BENIGN_SOURCE."""
    print(SEP)
    print("TIỀN XỬ LÝ CIC-IDS2017 - Hướng phát hiện tấn công Web")
    print(SEP)
    print(f"\n  Phạm vi   : WEB_ONLY={C.WEB_ONLY}, lớp = {C.CLASS_NAMES}")
    print(f"  Nguồn benign: {C.BENIGN_SOURCE!r}\n")

    web_file = C.RAW_DIR / C.WEB_ATTACK_FILE
    if not web_file.is_file():
        sys.exit(
            f"[LỖI] Không tìm thấy file bắt buộc:\n    {web_file}\n"
            f"       Chạy python -m flow_based.data.verify_data để chẩn đoán."
        )

    if C.BENIGN_SOURCE == "same_file":
        files = [web_file]
    elif C.BENIGN_SOURCE == "all_files":
        files = sorted(C.RAW_DIR.glob("*.csv"))
    else:
        sys.exit(f"[LỖI] BENIGN_SOURCE không hợp lệ: {C.BENIGN_SOURCE!r}")

    frames = []
    for f in files:
        # encoding='latin1' để không vỡ ở byte 0x96 trong nhãn Web Attack.
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]  # B2: bỏ dấu cách thừa
        # Hạ float64 -> float32 ngay khi đọc: giảm một nửa RAM.
        for c in df.columns:
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
        print(f"  {f.name:<58} {len(df):>9,} dòng")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    del frames
    print(f"  {'TỔNG':<58} {len(df):>9,} dòng, {df.shape[1]} cột")
    return df


# ---------------------------------------------------------------------------
# B4. Ánh xạ nhãn + lọc phạm vi lớp
# ---------------------------------------------------------------------------
def map_and_filter_labels(
    df: pd.DataFrame, strict_counts: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Chuẩn hóa nhãn, ánh xạ về nhóm, ASSERT CỨNG, rồi lọc theo phạm vi."""
    if "Label" not in df.columns:
        sys.exit(f"[LỖI] Thiếu cột 'Label'. Cột hiện có: {list(df.columns)[:12]}")

    print("\nPhân bố nhãn thô (repr - để lộ byte hỏng nếu có):")
    raw_counts = df["Label"].value_counts()
    for raw, n in raw_counts.items():
        print(f"  {n:>9,}  {normalize_label(raw)!r}   <- gốc {raw!r}")

    df["y_name"] = df["Label"].map(to_group)
    counts = Counter(df["y_name"])

    print("\nSau khi ánh xạ nhóm:")
    for name, n in counts.most_common():
        print(f"  {n:>9,}  {name}")

    # ---- CHỐT CHẶN: lỗi nhãn phải DỪNG CỨNG, không được im lặng ----
    assert_web_labels(counts, C.EXPECTED_WEB_COUNTS, strict=strict_counts)

    n_before = len(df)
    if C.WEB_ONLY:
        df = df[df["y_name"].isin(C.CLASS_NAMES)].copy()
        n_other = n_before - len(df)
        if n_other:
            print(
                f"\nBỏ {n_other:,} dòng tấn công phi-web (ngoài phạm vi đề tài): "
                f"DoS/DDoS/PortScan/Bot/Infiltration/Heartbleed/Patator"
            )

    df = df.drop(columns=["Label"])
    return df, dict(counts)


# ---------------------------------------------------------------------------
# B3 + B5. Bỏ cột rò rỉ, cột lặp, làm sạch Inf/NaN
# ---------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    # B3: cột định danh và thời gian
    leaky = [c for c in df.columns if c in C.LEAKY_COLS]
    if C.DROP_DEST_PORT and "Destination Port" in df.columns:
        leaky.append("Destination Port")
    if leaky:
        df = df.drop(columns=leaky)
        print(f"\nBỏ {len(leaky)} cột gây rò rỉ nhãn: {leaky}")
    meta["dropped_leaky_cols"] = leaky

    # Cột lặp tên: CIC-IDS2017 có 'Fwd Header Length' hai lần.
    dup_named = [c for c in df.columns if c.endswith(".1")]
    if dup_named:
        df = df.drop(columns=dup_named)
        print(f"Bỏ {len(dup_named)} cột lặp tên: {dup_named}")
    meta["dropped_duplicate_cols"] = dup_named

    # Mọi cột đặc trưng phải là số.
    feat_cols = [c for c in df.columns if c != "y_name"]
    for c in feat_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # B5: Inf -> NaN -> bỏ dòng. 'Flow Bytes/s' và 'Flow Packets/s' chứa Inf khi
    # thời lượng luồng bằng 0.
    n_inf = int(
        np.isinf(df[feat_cols].to_numpy(dtype="float64", na_value=np.nan)).sum()
    )
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=feat_cols)
    n_dropped = n_before - len(df)
    print(f"Bỏ {n_dropped:,} dòng chứa NaN/Inf (trong đó {n_inf:,} ô Inf)")
    meta["rows_dropped_nan_inf"] = n_dropped
    meta["inf_cells"] = n_inf

    return df


# ---------------------------------------------------------------------------
# B6. Khử trùng lặp - BẮT BUỘC trước khi chia tập
# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if not C.DEDUPLICATE:
        print(
            "[CẢNH BÁO] DEDUPLICATE=False - kết quả sẽ bị thổi phồng do bản ghi "
            "trùng lặp xuất hiện ở cả train và test."
        )
        meta["rows_dropped_duplicate"] = 0
        return df
    n_before = len(df)
    df = df.drop_duplicates()
    n = n_before - len(df)
    print(f"Bỏ {n:,} dòng trùng lặp hoàn toàn (TRƯỚC khi chia tập)")
    meta["rows_dropped_duplicate"] = n
    return df


# ---------------------------------------------------------------------------
# B7. Bỏ cột hằng
# ---------------------------------------------------------------------------
def drop_constant(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if not C.DROP_CONSTANT_COLS:
        meta["dropped_constant_cols"] = []
        return df
    feat_cols = [c for c in df.columns if c != "y_name"]
    nunique = df[feat_cols].nunique()
    const = nunique[nunique <= 1].index.tolist()
    if const:
        df = df.drop(columns=const)
        print(f"Bỏ {len(const)} cột hằng (phương sai = 0): {const}")
    meta["dropped_constant_cols"] = const
    return df


# ---------------------------------------------------------------------------
# B8. Chia phân tầng - MỘT phép chia dùng chung cho cả hai bài toán
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame, meta: dict):
    feat_cols = [c for c in df.columns if c != "y_name"]
    X = df[feat_cols].to_numpy(dtype="float32")

    # Mã hóa nhãn theo THỨ TỰ CỐ ĐỊNH trong config, không theo thứ tự xuất hiện,
    # để chỉ số lớp ổn định giữa mọi lần chạy.
    present = [c for c in C.CLASS_NAMES if c in set(df["y_name"])]
    name_to_idx = {name: i for i, name in enumerate(present)}
    y_multi = df["y_name"].map(name_to_idx).to_numpy(dtype="int64")

    # Nhãn nhị phân: 0 = BENIGN, 1 = bất kỳ tấn công web nào.
    y_bin = (df["y_name"] != "BENIGN").to_numpy().astype("int64")

    if C.SUBSAMPLE and len(X) > C.SUBSAMPLE:
        # Lấy mẫu con nhưng GIỮ NGUYÊN toàn bộ mẫu của các lớp thưa.
        rng = np.random.default_rng(C.SEED)
        benign_idx = np.flatnonzero(y_multi == name_to_idx["BENIGN"])
        attack_idx = np.flatnonzero(y_multi != name_to_idx["BENIGN"])
        keep_benign = rng.choice(
            benign_idx, size=max(0, C.SUBSAMPLE - len(attack_idx)), replace=False
        )
        keep = np.sort(np.concatenate([attack_idx, keep_benign]))
        X, y_multi, y_bin = X[keep], y_multi[keep], y_bin[keep]
        print(
            f"\n[SUBSAMPLE] Rút xuống {len(X):,} dòng, giữ nguyên toàn bộ "
            f"{len(attack_idx):,} mẫu tấn công."
        )

    # Chia hai giai đoạn: tách test trước, rồi tách val khỏi phần còn lại.
    idx = np.arange(len(X))
    idx_tr_val, idx_test = train_test_split(
        idx, test_size=C.TEST_SIZE, stratify=y_multi, random_state=C.SEED
    )
    val_frac = C.VAL_SIZE / (1.0 - C.TEST_SIZE)
    idx_train, idx_val = train_test_split(
        idx_tr_val,
        test_size=val_frac,
        stratify=y_multi[idx_tr_val],
        random_state=C.SEED,
    )

    meta["feature_names"] = feat_cols
    meta["class_names"] = present
    meta["binary_names"] = C.BINARY_NAMES
    return X, y_multi, y_bin, idx_train, idx_val, idx_test, present


# ---------------------------------------------------------------------------
# B9-B10. Chuẩn hóa - fit CHỈ trên train
# ---------------------------------------------------------------------------
def make_scaler(n_train: int):
    if C.SCALER == "quantile":
        # output_distribution='normal': ánh xạ về phân bố chuẩn theo thứ hạng,
        # khống chế giá trị ngoại lệ một cách tự nhiên (Chương 2 mục 2.3.3a).
        return QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, n_train),
            subsample=200_000,
            random_state=C.SEED,
        )
    if C.SCALER == "standard":
        return StandardScaler()
    if C.SCALER == "robust":
        return RobustScaler()
    sys.exit(f"[LỖI] SCALER không hợp lệ: {C.SCALER!r}")


def main(strict_counts: bool = True) -> dict:
    if C.DATASET_ID == "cse_cicids2018":
        if not strict_counts:
            raise ValueError(
                "Profile 2018 yêu cầu kiểm tra dữ liệu thật; không dùng --synthetic."
            )
        from flow_based.data.cse2018 import prepare

        return prepare()
    meta: dict = {
        "dataset": "CIC-IDS2017 (MachineLearningCVE)",
        "dataset_id": C.DATASET_ID,
        "web_only": C.WEB_ONLY,
        "benign_source": C.BENIGN_SOURCE,
        "seed": C.SEED,
        "scaler": C.SCALER,
        "deduplicate": C.DEDUPLICATE,
        "drop_dest_port": C.DROP_DEST_PORT,
    }

    df = load_raw()
    meta["rows_raw"] = len(df)

    df, raw_group_counts = map_and_filter_labels(df, strict_counts=strict_counts)
    meta["label_counts_after_mapping"] = raw_group_counts

    df = clean_columns(df, meta)
    df = deduplicate(df, meta)
    df = drop_constant(df, meta)

    n_feat = df.shape[1] - 1
    print(
        f"\nCòn lại {len(df):,} dòng "
        f"({len(df) / meta['rows_raw'] * 100:.1f}% so với ban đầu), "
        f"{n_feat} đặc trưng"
    )

    X, y_multi, y_bin, i_tr, i_va, i_te, class_names = split(df, meta)
    del df

    # --- B9/B10: fit trên train, transform cho val/test ---
    scaler = make_scaler(len(i_tr))
    X_train = scaler.fit_transform(X[i_tr]).astype("float32")
    X_val = scaler.transform(X[i_va]).astype("float32")
    X_test = scaler.transform(X[i_te]).astype("float32")

    y_multi_train, y_multi_val, y_multi_test = (
        y_multi[i_tr],
        y_multi[i_va],
        y_multi[i_te],
    )
    y_bin_train, y_bin_val, y_bin_test = y_bin[i_tr], y_bin[i_va], y_bin[i_te]

    # --- Bảng phân bố lớp theo tập ---
    print("\nPhân bố lớp sau khi chia tập:")
    print(f"  {'Lớp':<18}{'Train':>10}{'Val':>9}{'Test':>9}")
    print("  " + "-" * 46)
    for i, name in enumerate(class_names):
        star = " *" if name in C.WEB_ATTACK_CLASSES else ""
        print(
            f"  {name:<18}{int((y_multi_train == i).sum()):>10,}"
            f"{int((y_multi_val == i).sum()):>9,}"
            f"{int((y_multi_test == i).sum()):>9,}{star}"
        )
    print("  " + "-" * 46)
    print(
        f"  {'TỔNG':<18}{len(y_multi_train):>10,}{len(y_multi_val):>9,}"
        f"{len(y_multi_test):>9,}"
    )
    print("\n  (*) lớp trọng tâm của đề tài")

    # Cảnh báo lớp quá thưa trong tập test - độ đo theo lớp sẽ không đáng tin.
    for i, name in enumerate(class_names):
        n_test = int((y_multi_test == i).sum())
        if 0 < n_test < 10:
            print(
                f"\n  [CẢNH BÁO] Lớp '{name}' chỉ có {n_test} mẫu trong tập kiểm tra. "
                f"Precision/Recall của lớp này KHÔNG có ý nghĩa thống kê - "
                f"phải nêu rõ điều này trong báo cáo."
            )
        elif n_test == 0:
            print(
                f"\n  [CẢNH BÁO NẶNG] Lớp '{name}' KHÔNG có mẫu nào trong tập kiểm tra."
            )

    counts = Counter(y_multi.tolist())
    ir = max(counts.values()) / max(min(counts.values()), 1)
    meta["imbalance_ratio"] = float(ir)
    print(f"\n  Tỷ số mất cân bằng (lớp đông nhất / lớp thưa nhất): {ir:,.1f} : 1")

    meta.update(
        {
            "n_features": int(X_train.shape[1]),
            "n_classes_multi": len(class_names),
            "n_classes_binary": 2,
            "n_train": int(len(y_multi_train)),
            "n_val": int(len(y_multi_val)),
            "n_test": int(len(y_multi_test)),
            "leak_prevention": {
                "deduplicated_before_split": bool(C.DEDUPLICATE),
                "scaler_fitted_on_train_only": True,
                "identifier_columns_dropped": True,
                "resampling_on_train_only": True,
            },
        }
    )

    # --- Ghi ra đĩa ---
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.PROC_DIR / "dataset.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_multi_train=y_multi_train,
        y_multi_val=y_multi_val,
        y_multi_test=y_multi_test,
        y_bin_train=y_bin_train,
        y_bin_val=y_bin_val,
        y_bin_test=y_bin_test,
    )
    joblib.dump(scaler, C.PROC_DIR / "scaler.joblib")
    with open(C.PROC_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu vào {C.PROC_DIR}")
    print(f"  dataset.npz    X_train {X_train.shape}  X_test {X_test.shape}")
    print(f"  meta.json      {len(class_names)} lớp đa lớp + 2 lớp nhị phân")
    print(f"  scaler.joblib  {C.SCALER} (fit CHỈ trên train)")
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu tấn công web")
    parser.add_argument("--dataset", choices=list(C.DATASETS), default=None)
    parser.add_argument("--synthetic", action="store_true")
    options = parser.parse_args()
    if options.dataset:
        C.select_dataset(options.dataset)
    main(strict_counts=not options.synthetic)
