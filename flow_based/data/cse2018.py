"""Nguồn AWS, kiểm tra và tiền xử lý riêng ngày tấn công web 23/02/2018."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd

from flow_based import config as C
from flow_based.data.labels import to_group

BASE_URL = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
)
# SHA-256 được ghim từ bản tải HTTPS trực tiếp trên kho AWS chính thức.
ATTACK_DAY = C.DATASETS["cse_cicids2018"]["attack_day"]
SOURCES = [
    {
        "name": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
        "size": 382840456,
        "rows": 1048575,
        "sha256": "d0a7f5059d9823b6e9b392b759e306481a3502d190dea7a1b5502ae079ea069b",
        "labels": {
            "Benign": 1048009,
            "Brute Force -Web": 362,
            "Brute Force -XSS": 151,
            "SQL Injection": 53,
        },
    },
]
LEAKY_COLUMNS = {
    "Dst Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Source IP",
    "Destination IP",
    "Source Port",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_bytes(path: Path, source: dict) -> str:
    if not path.is_file() or path.stat().st_size != source["size"]:
        raise ValueError(
            f"Sai kích thước hoặc thiếu {path}; cần {source['size']} byte."
        )
    digest = sha256_file(path)
    if digest != source["sha256"]:
        raise ValueError(f"SHA-256 không khớp: {path.name}; không sử dụng dữ liệu này.")
    return digest


def download(raw_dir: Path | None = None) -> list[dict]:
    """Tải streaming, kiểm tra trước khi thay thế; giữ file đích nếu tải hỏng."""
    root = Path(raw_dir or C.RAW_DIR)
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for source in SOURCES:
        target = root / source["name"]
        url = BASE_URL + source["name"]
        if target.is_file():
            try:
                digest = check_bytes(target, source)
                print(f"[đã xác minh] {target.name}", flush=True)
                receipts.append(
                    {**source, "url": url, "sha256": digest, "reused": True}
                )
                continue
            except ValueError:
                print(
                    f"[tải lại] {target.name}: bản hiện có không khớp checksum",
                    flush=True,
                )
        partial = target.with_suffix(".csv.part")
        downloaded, last = 0, time.monotonic()
        print(f"[download] {url}", flush=True)
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            partial.open("wb") as out,
        ):
            headers = dict(response.headers)
            while block := response.read(4 * 1024 * 1024):
                out.write(block)
                downloaded += len(block)
                if time.monotonic() - last >= 10:
                    print(
                        f"  {target.name}: {downloaded / source['size']:.1%}",
                        flush=True,
                    )
                    last = time.monotonic()
        digest = check_bytes(partial, source)
        partial.replace(target)
        receipts.append(
            {
                **source,
                "url": url,
                "sha256": digest,
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
            }
        )
        print(f"[download xong] {target.name}: {digest}", flush=True)
    (root / f"source_receipt_{ATTACK_DAY}.json").write_text(
        json.dumps(
            {
                "dataset_id": "cse_cicids2018",
                "attack_day": ATTACK_DAY,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "official_registry": "https://registry.opendata.aws/cse-cic-ids2018/",
                "checksum_origin": "SHA-256 pinned from direct AWS HTTPS download on 2026-10-20",
                "files": receipts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return receipts


def verify(raw_dir: Path | None = None) -> list[dict]:
    root = Path(raw_dir or C.RAW_DIR)
    reports = []
    for source in SOURCES:
        path = root / source["name"]
        print(f"[verify] {path.name}", flush=True)
        digest = check_bytes(path, source)
        counts, rows = Counter(), 0
        for frame in pd.read_csv(path, usecols=["Label"], chunksize=200_000):
            counts.update(frame["Label"].value_counts().to_dict())
            rows += len(frame)
        if rows != source["rows"] or dict(counts) != source["labels"]:
            raise ValueError(
                f"Sai số dòng/nhãn trong {path.name}: {rows}, {dict(counts)}"
            )
        print(f"  ĐẠT: {rows:,} dòng; {dict(counts)}", flush=True)
        reports.append({**source, "url": BASE_URL + source["name"], "sha256": digest})
    return reports


def load_frames(raw_dir: Path, meta: dict) -> pd.DataFrame:
    """Đọc theo chunk; giữ độ chính xác float64 cho làm sạch/khử trùng lặp."""
    frames, headers = [], None
    for source in SOURCES:
        count = 0
        for chunk in pd.read_csv(raw_dir / source["name"], chunksize=100_000):
            chunk.columns = chunk.columns.str.strip()
            current = list(chunk.columns)
            if headers is None:
                headers = current
            if current != headers:
                raise ValueError("CSV 2018 có schema không nhất quán giữa các chunk.")
            count += len(chunk)
            chunk = chunk.drop(columns=[c for c in chunk if c in LEAKY_COLUMNS])
            # Ánh xạ từng giá trị nhãn duy nhất, tránh gọi regex hàng triệu lần.
            mapping = {label: to_group(label) for label in chunk["Label"].unique()}
            chunk["y_name"] = chunk.pop("Label").map(mapping).astype("category")
            if not chunk["y_name"].isin(C.CLASS_NAMES).all():
                raise ValueError(
                    "Có nhãn ngoài phạm vi trong file tấn công web 23/02/2018."
                )
            for col in chunk.columns:
                if col != "y_name":
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            frames.append(chunk)
            print(f"[đọc] {source['name']}: {count:,}/{source['rows']:,}", flush=True)
    meta["dropped_leaky_cols"] = [c for c in headers if c in LEAKY_COLUMNS]
    frame = pd.concat(frames, ignore_index=True)
    meta["rows_raw"] = len(frame)
    meta["label_counts_after_mapping"] = frame["y_name"].value_counts().to_dict()
    return frame


def prepare_frame(frame: pd.DataFrame, meta: dict) -> tuple[dict, object]:
    """Phép chia/biến đổi train-only; tách riêng để kiểm tra bằng fixture nhỏ."""
    from flow_based.data.data_prep import make_scaler, split

    features = [c for c in frame if c != "y_name"]
    valid = np.ones(len(frame), dtype=bool)
    meta["inf_cells"] = 0
    for feature in features:
        values = frame[feature].to_numpy(dtype=np.float64, na_value=np.nan)
        meta["inf_cells"] += int(np.isinf(values).sum())
        valid &= np.isfinite(values)
    before = len(frame)
    frame = frame.loc[valid]
    meta["rows_dropped_nan_inf"] = before - len(frame)
    before = len(frame)
    frame = frame.drop_duplicates()
    meta["rows_dropped_duplicate"] = before - len(frame)
    print(
        f"[làm sạch] NaN/Inf: {meta['rows_dropped_nan_inf']:,}; trùng: {meta['rows_dropped_duplicate']:,}",
        flush=True,
    )
    # Giữ chỉ số gốc để audit phép chia trước khi chọn đặc trưng.
    X, ym, yb, itr, iva, ite, names = split(frame, meta)
    del frame
    gc.collect()
    # Cột hằng được xác định chỉ từ train, sau khi chia dữ liệu.
    keep = np.ptp(X[itr], axis=0) > 0
    meta["dropped_constant_cols"] = [f for f, ok in zip(features, keep) if not ok]
    meta["feature_names"] = [f for f, ok in zip(features, keep) if ok]
    X = X[:, keep]
    scaler = make_scaler(len(itr))
    print(
        f"[chuẩn hóa] Fit quantile chỉ trên {len(itr):,} dòng train, {X.shape[1]} đặc trưng",
        flush=True,
    )
    arrays = {}
    for part, indices in [("train", itr), ("val", iva), ("test", ite)]:
        transform = scaler.fit_transform if part == "train" else scaler.transform
        arrays[f"X_{part}"] = transform(X[indices]).astype(np.float32)
        arrays[f"y_multi_{part}"] = ym[indices]
        arrays[f"y_bin_{part}"] = yb[indices]
        arrays[f"indices_{part}"] = indices
        meta[f"n_{part}"] = int(len(indices))
        meta.setdefault("split_class_counts", {})[part] = {
            name: int((ym[indices] == i).sum()) for i, name in enumerate(names)
        }
    counts = Counter(ym.tolist())
    meta.update(
        n_features=int(X.shape[1]),
        n_classes_multi=len(names),
        n_classes_binary=2,
        imbalance_ratio=float(max(counts.values()) / min(counts.values())),
        leak_prevention={
            "deduplicated_before_split": True,
            "constant_columns_fitted_on_train_only": True,
            "scaler_fitted_on_train_only": True,
            "identifier_columns_dropped": True,
            "resampling_on_train_only": True,
        },
    )
    for name in names:
        n = meta["split_class_counts"]["test"][name]
        if n < 20:
            print(
                f"[lưu ý] Test {name}: chỉ {n} mẫu; không kết luận mạnh từ lớp này.",
                flush=True,
            )
    return arrays, scaler


def prepare() -> dict:
    if C.DATASET_ID != "cse_cicids2018":
        raise ValueError("Chọn --dataset cse_cicids2018 trước khi tiền xử lý.")
    if C.SUBSAMPLE is not None:
        raise ValueError("Profile 2018 đầy đủ không cho phép SUBSAMPLE.")
    sources = verify()
    meta = {
        "dataset_id": C.DATASET_ID,
        "dataset": C.DATASET_NAME,
        "attack_day": ATTACK_DAY,
        "web_only": True,
        "benign_source": "same_file_23_february",
        "seed": C.SEED,
        "scaler": C.SCALER,
        "deduplicate": True,
        "drop_dest_port": True,
        "source_verified": True,
        "source_files": sources,
        "dropped_duplicate_cols": [],
    }
    arrays, scaler = prepare_frame(load_frames(C.RAW_DIR, meta), meta)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = C.PROC_DIR / "dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    joblib.dump(scaler, C.PROC_DIR / "scaler.joblib")
    meta["processed_sha256"] = sha256_file(dataset_path)
    (C.PROC_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: meta[k]
                for k in [
                    "n_features",
                    "n_train",
                    "n_val",
                    "n_test",
                    "split_class_counts",
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"[đã lưu] {C.PROC_DIR}", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "verify", "prepare"])
    args = parser.parse_args()
    C.select_dataset("cse_cicids2018")
    {"download": download, "verify": verify, "prepare": prepare}[args.action]()


if __name__ == "__main__":
    main()
