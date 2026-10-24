"""
run_all.py - Điều phối ba kịch bản cốt lõi KB1, KB2, KB3.

Chạy:
    python -m flow_based.experiments.run_all                      # cả hai bài toán, 3 seed
    python -m flow_based.experiments.run_all --task binary        # chỉ bài toán nhị phân
    python -m flow_based.experiments.run_all --seeds 42           # chỉ một seed (chạy nhanh)
    python -m flow_based.experiments.run_all --quick              # 3 epoch, 1 seed - chỉ để thử pipeline

KB1  Mô hình cơ sở      -> Bảng 3.12
KB2  Transformer đề xuất -> Bảng 3.7, 3.11, 3.13 + Hình 3.1, 3.2
KB3  So sánh            -> Bảng 3.14, 3.17 + Hình 3.3-3.7

MỌI CON SỐ ĐỀU ĐI QUA report_export.py. Không nhập tay số nào vào file Word.
Mỗi lần chạy sinh một thư mục riêng trong ketqua/ kèm manifest.json để truy vết.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flow_based import config as C
from flow_based.experiments.seed_results import (
    SCALAR_METRICS,
    save_seed_result,
    scalar_summary,
    write_task_summary,
)
from flow_based.training.baselines import run_all_baselines
from flow_based.models.ft_transformer import build_transformer
from flow_based.reporting.report_export import (
    ResultStore,
    aggregate_seeds,
    collect_hardware_info,
    collect_software_info,
    comparison_table,
    count_parameters,
    evaluate_predictions,
    label_distribution_table,
    measure_inference_time,
    model_config_table,
    per_class_table,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_history,
    set_global_seed,
    split_summary_table,
    training_summary_table,
)
from flow_based.training.trainer import get_device, predict, train_torch_model

SEP = "=" * 78


def _mean_seed_metrics(seed_runs: list[dict]) -> dict:
    """Cùng cách lấy trung bình cho baseline và FT; giữ số thô theo từng seed."""
    out: dict = {}
    for k in SCALAR_METRICS:
        vals = [r[k] for r in seed_runs if k in r and not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _mean_per_class_recall(seed_runs: list[dict], class_names: list[str]) -> dict:
    """Recall trung bình qua nhiều seed cho từng lớp trong class_names.

    Dùng khi cần in/báo cáo recall theo lớp mà kết quả đã được tổng hợp qua
    _mean_seed_metrics() (dict đó KHÔNG có khóa "per_class", vì precision/
    recall/F1 theo lớp không nằm trong danh sách độ đo vô hướng được trung
    bình ở đó).
    """
    out: dict = {}
    for c in class_names:
        vals = [
            ev["per_class"][c]["recall"]
            for ev in seed_runs
            if c in ev.get("per_class", {})
        ]
        out[c] = float(np.mean(vals)) if vals else None
    return out


# ---------------------------------------------------------------------------
# Nạp dữ liệu đã tiền xử lý
# ---------------------------------------------------------------------------
def load_dataset() -> dict:
    npz_path = C.PROC_DIR / "dataset.npz"
    meta_path = C.PROC_DIR / "meta.json"
    if not npz_path.is_file():
        sys.exit(
            f"[LỖI] Chưa có {npz_path}. Chạy python -m flow_based.data.data_prep trước."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stored_id = meta.get("dataset_id")
    if stored_id is None and meta.get("dataset", "").startswith("CIC-IDS2017"):
        stored_id = "cicids2017"  # Đọc lại được dữ liệu 2017 đã chuẩn bị trước đây.
    if stored_id != C.DATASET_ID:
        raise ValueError(
            f"Sai dataset: yêu cầu {C.DATASET_ID}, dữ liệu đã xử lý là {stored_id!r}."
        )
    if C.DATASET_ID == "cse_cicids2018" and not meta.get("source_verified"):
        raise ValueError("Dữ liệu 2018 chưa có bằng chứng xác minh nguồn.")
    if C.DATASET_ID == "cse_cicids2018":
        if meta.get("attack_day") != C.ATTACK_DAY:
            raise ValueError(
                f"Sai ngày dữ liệu 2018: cần {C.ATTACK_DAY}, "
                f"đã xử lý {meta.get('attack_day')!r}. Chạy lại tiền xử lý một ngày."
            )
        from flow_based.data.cse2018 import sha256_file

        if sha256_file(npz_path) != meta.get("processed_sha256"):
            raise ValueError(
                "Mảng dữ liệu đã xử lý 2018 không khớp SHA-256 trong meta.json."
            )
    with np.load(npz_path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data["meta"] = meta
    data["class_names_multi"] = meta["class_names"]
    data["class_names_bin"] = meta.get("binary_names", C.BINARY_NAMES)
    data["n_classes_multi"] = meta["n_classes_multi"]
    data["n_features"] = meta["n_features"]

    print(
        f"[dữ liệu] train {data['X_train'].shape}  "
        f"val {data['X_val'].shape}  test {data['X_test'].shape}"
    )
    print(
        f"[dữ liệu] {data['n_features']} đặc trưng, "
        f"lớp đa lớp: {data['class_names_multi']}"
    )
    return data


def task_view(data: dict, task: str) -> dict:
    """Trả về nhãn và tên lớp tương ứng một bài toán, dùng chung một phép chia."""
    key = "bin" if task == "binary" else "multi"
    return {
        "y_train": data[f"y_{key}_train"],
        "y_val": data[f"y_{key}_val"],
        "y_test": data[f"y_{key}_test"],
        "class_names": data["class_names_bin"]
        if task == "binary"
        else data["class_names_multi"],
        "n_classes": 2 if task == "binary" else data["n_classes_multi"],
        "label": "nhị phân" if task == "binary" else "đa lớp",
    }


def binary_scores(
    y_true: np.ndarray, y_proba: np.ndarray, task: str, class_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Quy về lát cắt nhị phân BENIGN vs tấn công, để vẽ ROC và PR.

    plot_roc_curves() / plot_pr_curves() của report_export.py chỉ nhận nhãn nhị
    phân. Với bài toán đa lớp, điểm số tấn công = 1 - P(BENIGN).
    """
    if task == "binary":
        return y_true, y_proba[:, 1]
    benign_idx = class_names.index("BENIGN")
    y_bin = (y_true != benign_idx).astype(int)
    return y_bin, 1.0 - y_proba[:, benign_idx]


# ---------------------------------------------------------------------------
# KB2 - huấn luyện Transformer
# ---------------------------------------------------------------------------
def train_transformer_once(
    data: dict,
    view: dict,
    seed: int,
    train_cfg: dict,
    model_cfg: dict,
    verbose: bool = True,
) -> dict:
    set_global_seed(seed)
    device = get_device(verbose=verbose)

    model = build_transformer(data["n_features"], view["n_classes"], model_cfg)
    n_params = count_parameters(model)["trainable"]

    print(
        f"\n  FT-Transformer | {data['n_features']} đặc trưng "
        f"(+1 token [CLS]) | {view['n_classes']} lớp | {n_params:,} tham số"
    )

    res = train_torch_model(
        model,
        data["X_train"],
        view["y_train"],
        data["X_val"],
        view["y_val"],
        n_classes=view["n_classes"],
        train_cfg=train_cfg,
        seed=seed,
        device=device,
        verbose=verbose,
        tag=f"FT-Transformer seed={seed}",
    )

    y_pred, y_proba = predict(res["model"], data["X_test"], device)
    res.update(
        {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "n_params": n_params,
            "train_time_s": res["total_time_s"],
        }
    )
    return res


# ---------------------------------------------------------------------------
# Chạy trọn một bài toán
# ---------------------------------------------------------------------------
def run_task(
    data: dict,
    task: str,
    store: ResultStore,
    seeds: list[int],
    train_cfg: dict,
    model_cfg: dict,
) -> dict:
    view = task_view(data, task)
    cn = view["class_names"]
    y_test = view["y_test"]
    suffix = "nhiphan" if task == "binary" else "dalop"

    print("\n\n" + SEP)
    print(f"BÀI TOÁN: {view['label'].upper()}  ({view['n_classes']} lớp)")
    print(SEP)

    results: dict[str, dict] = {}  # tên -> kết quả evaluate_predictions
    extras: dict[str, dict] = {}  # tên -> số tham số / thời gian huấn luyện
    models: dict[str, object] = {}
    model_runs: dict[str, list[dict]] = {}
    split_seed = data["meta"].get("seed", C.SEED)

    def record(name, seed, evaluation, result):
        configs = {
            "Logistic Regression": {**C.LOGREG_PARAMS, "random_state": seed},
            "Random Forest": {**C.RF_PARAMS, "random_state": seed},
            "XGBoost": {**C.XGB_PARAMS, "random_state": seed},
            "MLP": {"model": C.MLP, "train": train_cfg},
            "FT-Transformer": {"model": model_cfg, "train": train_cfg},
        }
        raw = save_seed_result(
            store.run_dir,
            task,
            name,
            seed,
            evaluation,
            result,
            y_test,
            split_seed=split_seed,
            effective_config=configs[name],
        )
        model_runs.setdefault(name, []).append(raw)
        store.add_metrics(f"{name} seed={seed} [{view['label']}]", evaluation)

    def summary_extra(runs, n_params):
        timing = scalar_summary(r["train_time_s"] for r in runs)
        f1 = scalar_summary(r["metrics"]["f1_macro"] for r in runs)
        return {
            "Số tham số": n_params if n_params else "—",
            "Thời gian huấn luyện (giây)": timing["mean"],
            "SD thời gian huấn luyện (giây)": timing["std"],
            "SD F1 (macro)": f1["std"],
            "Số seed": len(runs),
        }

    # ---------------- KB1 ----------------
    kb1 = {}
    for seed in seeds:
        print(f"\n--- Baseline seed = {seed} ---")

        def completed(name, result):
            ev = evaluate_predictions(
                y_test, result["y_pred"], result["y_proba"], class_names=cn
            )
            record(name, seed, ev, result)
            print(
                f"  {name} seed={seed}: macro-F1 test = {ev['f1_macro']:.4f}",
                flush=True,
            )

        baseline_results = run_all_baselines(
            data, task, seed=seed, train_cfg=train_cfg, on_complete=completed
        )
        if seed == seeds[0]:
            kb1 = baseline_results  # Seed minh họa được ấn định trước khi xem test.
    for name, r in kb1.items():
        runs = model_runs[name]
        results[name] = _mean_seed_metrics([run["metrics"] for run in runs])
        extras[name] = summary_extra(runs, r["n_params"])
        models[name] = r["model"]

    df_kb1 = comparison_table({k: results[k] for k in kb1}, extra=extras)
    store.add_table(f"bang_3_12_kb1_mo_hinh_co_so_{suffix}", df_kb1)

    # ---------------- KB2 ----------------
    print("\n" + SEP)
    print(f"KB2 - FT-TRANSFORMER ĐỀ XUẤT  ({view['label']}, {len(seeds)} seed)")
    print(SEP)

    seed_runs, all_res = [], []
    for seed in seeds:
        print(f"\n--- seed = {seed} ---")
        res = train_transformer_once(data, view, seed, train_cfg, model_cfg)
        ev = evaluate_predictions(y_test, res["y_pred"], res["y_proba"], class_names=cn)
        seed_runs.append(ev)
        all_res.append(res)
        record("FT-Transformer", seed, ev, res)
        print(f"\n  seed {seed}: macro-F1 test = {ev['f1_macro']:.4f}")

    # Cố định seed minh họa trước khi đánh giá test, giống các baseline.
    rep_idx = 0
    first, first_res = seed_runs[rep_idx], all_res[rep_idx]
    if len(seeds) > 1:
        print(
            f"\n  [minh họa] seed={seeds[rep_idx]} cố định trước cho bảng/hình "
            "của một mô hình; bảng so sánh dùng trung bình của toàn bộ seed."
        )

    # Bảng 3.7 - cấu hình mô hình, trích từ CHÍNH cấu hình đã chạy
    store.add_table(
        f"bang_3_7_cau_hinh_transformer_{suffix}",
        model_config_table(
            C.bang_3_7(
                data["n_features"],
                view["n_classes"],
                first_res["n_params"],
                model_cfg,
                train_cfg=train_cfg,
            ),
            C.GROUP_MAP,
        ),
    )

    # Bảng 3.11 - tóm tắt huấn luyện (seed đầu tiên)
    store.add_table(
        f"bang_3_11_tom_tat_huan_luyen_{suffix}",
        training_summary_table(
            first_res["history"],
            first_res["best_epoch"],
            first_res["total_time_s"],
            train_cfg["early_stop_patience"],
            monitor="val_macro_f1",
        ),
    )

    # Bảng 3.13 - kết quả theo từng lớp. Bảng quan trọng nhất của chương.
    store.add_table(
        f"bang_3_13_ket_qua_theo_lop_{suffix}",
        per_class_table(y_test, first_res["y_pred"], class_names=cn),
    )

    # Hình 3.1 / 3.2 - đường cong huấn luyện
    plot_training_history(
        first_res["history"],
        store.fig_path(f"hinh_3_1_duong_cong_huan_luyen_{suffix}"),
        metric_key="macro_f1",
        metric_label="Macro-F1",
        best_epoch=first_res["best_epoch"],
    )

    # Cả năm mô hình đều dùng trung bình qua cùng danh sách seed.
    ft_summary = _mean_seed_metrics(seed_runs)
    results["FT-Transformer"] = ft_summary
    extras["FT-Transformer"] = summary_extra(
        model_runs["FT-Transformer"], first_res["n_params"]
    )
    models["FT-Transformer"] = first_res["model"]

    # SD giữa seed được lưu riêng với SD đo lặp tốc độ suy luận bên dưới.
    stability = []
    for name, runs in model_runs.items():
        values = [{**r["metrics"], "train_time_s": r["train_time_s"]} for r in runs]
        table = aggregate_seeds(values, metrics=(*SCALAR_METRICS, "train_time_s"))
        table.insert(0, "Mô hình", name)
        stability.append(table)
    df_seed = pd.concat(stability, ignore_index=True)
    store.add_table(f"bang_do_on_dinh_theo_seed_{suffix}", df_seed)
    write_task_summary(
        store.run_dir,
        task,
        seeds,
        model_runs,
        split_seed=split_seed,
        quick=store.config.get("quick", False),
    )
    print(f"\n  Độ ổn định qua {len(seeds)} seed cho cả năm mô hình:")
    print(df_seed[df_seed["Độ đo"] == "f1_macro"].to_string(index=False))

    # ---------------- KB3 ----------------
    print("\n" + SEP)
    print(f"KB3 - SO SÁNH TOÀN BỘ MÔ HÌNH  ({view['label']})")
    print(SEP)

    df_kb3 = comparison_table(results, extra=extras)
    store.add_table(f"bang_3_14_so_sanh_tong_hop_{suffix}", df_kb3)
    print("\n" + df_kb3.to_string(index=False))

    # Hình 3.3 - biểu đồ cột so sánh
    plot_model_comparison(df_kb3, store.fig_path(f"hinh_3_3_so_sanh_macro_f1_{suffix}"))

    # Hình 3.4 / 3.5 - ROC và PR trên lát cắt nhị phân
    curves = {}
    for name in results:
        r = kb1.get(name)
        proba = r["y_proba"] if r else first_res["y_proba"]
        yb, sb = binary_scores(y_test, proba, task, cn)
        curves[f"{name} (seed={seeds[0]})"] = (yb, sb)
    plot_roc_curves(curves, store.fig_path(f"hinh_3_4_roc_{suffix}"))
    plot_pr_curves(curves, store.fig_path(f"hinh_3_5_precision_recall_{suffix}"))

    # Hình 3.6 / 3.7 - ma trận nhầm lẫn của mô hình đề xuất
    cm = np.array(first["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        cn,
        store.fig_path(f"hinh_3_6_ma_tran_nham_lan_{suffix}"),
        normalize=False,
        title=f"FT-Transformer — {view['label']} — seed={seeds[0]}",
    )
    plot_confusion_matrix(
        cm,
        cn,
        store.fig_path(f"hinh_3_7_ma_tran_chuan_hoa_{suffix}"),
        normalize=True,
        title=f"FT-Transformer — {view['label']} — seed={seeds[0]}",
    )

    # Bảng 3.17 - chi phí tính toán
    n_sample = min(20_000, len(data["X_test"]))
    Xs = data["X_test"][:n_sample]
    rows = []
    for name, mdl in models.items():
        dev = (
            "cuda"
            if (
                name in ("MLP", "FT-Transformer")
                and __import__("torch").cuda.is_available()
            )
            else "cpu"
        )
        try:
            # n_warmup/n_repeat cao hơn mặc định gợi ý ở KB1/KB2 (2/5): suy
            # luận vốn đã rất nhanh (<1s cho 20.000 mẫu ở mọi mô hình) nên
            # tăng số lần lặp gần như không tốn thêm thời gian đáng kể, đổi
            # lại độ lệch chuẩn ước lượng ổn định hơn cho Bảng 3.17.
            t = measure_inference_time(
                mdl, Xs, batch_size=1024, n_warmup=3, n_repeat=10, device=dev
            )
        except Exception as e:  # pragma: no cover
            print(f"  [BỎ QUA] Không đo được {name}: {e}")
            continue
        rows.append({"Mô hình": name, "Số tham số": extras[name]["Số tham số"], **t})
    if rows:
        store.add_table(f"bang_3_17_chi_phi_tinh_toan_{suffix}", pd.DataFrame(rows))

    # Nhấn mạnh recall của nhóm tấn công web - chỉ số trọng tâm đề tài.
    # In cả hai: recall của seed đại diện (để đối chiếu với Bảng 3.13) VÀ
    # trung bình qua toàn bộ seed (đây mới là con số nên trích dẫn - xem
    # docstring của _mean_seed_metrics() về lý do không dùng 1 seed đơn lẻ).
    if task == "multi":
        web_recalls = {
            k: first["per_class"][k]["recall"]
            for k in C.WEB_ATTACK_CLASSES
            if k in first["per_class"]
        }
        if web_recalls:
            print(
                f"\n  Recall theo từng lớp tấn công web "
                f"(FT-Transformer, seed đại diện = {seeds[rep_idx]}):"
            )
            for k, v in web_recalls.items():
                print(f"    {k:<18} {v:.4f}")
            print(
                f"    {'TRUNG BÌNH (1 seed)':<18} "
                f"{float(np.mean(list(web_recalls.values()))):.4f}"
            )

        if len(seed_runs) > 1:
            per_seed_means = [
                float(
                    np.mean(
                        [
                            ev["per_class"][k]["recall"]
                            for k in C.WEB_ATTACK_CLASSES
                            if k in ev["per_class"]
                        ]
                    )
                )
                for ev in seed_runs
            ]
            overall = float(np.mean(per_seed_means))
            print(
                f"    {'TRUNG BÌNH (' + str(len(seeds)) + ' seed)':<18} {overall:.4f}"
                f"   <- CHỈ SỐ TRỌNG TÂM ĐỀ TÀI, dùng số này thay vì số 1 seed ở trên"
            )

    return {"comparison": df_kb3, "results": results, "seed_runs": seed_runs}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy KB1, KB2, KB3")
    ap.add_argument("--dataset", choices=list(C.DATASETS), default=None)
    ap.add_argument("--task", choices=["binary", "multi", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="3 epoch, 1 seed - chỉ để thử pipeline, KHÔNG báo cáo",
    )
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    train_cfg = dict(C.TRAIN)
    model_cfg = dict(C.MODEL)
    seeds = args.seeds or C.SEEDS
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        ap.error("Seed phải khác nhau và nằm trong khoảng 0 đến 2**32 - 1.")
    if args.dataset:
        C.select_dataset(args.dataset)

    if args.quick:
        train_cfg["max_epochs"] = 3
        train_cfg["early_stop_patience"] = 3
        seeds = seeds[:1]
        print("[QUICK] 3 epoch, 1 seed. Số liệu KHÔNG dùng để báo cáo.")

    data = load_dataset()
    n_tasks = 2 if args.task == "both" else 1
    print(
        f"[protocol] Cùng seeds={seeds} cho 5 mô hình, "
        f"{n_tasks} bài toán: {5 * len(seeds) * n_tasks} lần huấn luyện."
    )
    print(
        f"[protocol] Giữ nguyên dữ liệu đã chia, split_seed={data['meta'].get('seed', C.SEED)}."
    )

    store = ResultStore(
        out_dir=str(C.RESULTS_DIR),
        **({"run_name": args.run_name} if args.run_name else {}),
    )
    store.config = {
        "dataset_id": C.DATASET_ID,
        "attack_day": data["meta"].get("attack_day"),
        "source_package_sha256": os.environ.get("IDS_PACKAGE_SHA256"),
        "model": model_cfg,
        "train": train_cfg,
        "seeds": seeds,
        "web_only": C.WEB_ONLY,
        "benign_source": data["meta"].get("benign_source", C.BENIGN_SOURCE),
        "scaler": C.SCALER,
        "quick": args.quick,
        "data_meta": data["meta"],
        "seed_protocol": {
            "training_seeds": seeds,
            "fixed_split": True,
            "split_seed": data["meta"].get("seed", C.SEED),
            "std_ddof": 1,
            "representative_seed": seeds[0],
        },
    }

    # --- Bảng 3.4, 3.5: mô tả dữ liệu (chung cho cả hai bài toán) ---
    cn_multi = data["class_names_multi"]
    y_all = np.concatenate(
        [data["y_multi_train"], data["y_multi_val"], data["y_multi_test"]]
    )
    store.add_table(
        "bang_3_4_phan_bo_nhan", label_distribution_table(y_all, class_names=cn_multi)
    )
    store.add_table(
        "bang_3_5_phan_chia_du_lieu",
        split_summary_table(
            data["y_multi_train"],
            data["y_multi_val"],
            data["y_multi_test"],
            class_names=cn_multi,
        ),
    )
    # --- Bảng 3.6: cấu hình mô hình cơ sở ---
    store.add_table(
        "bang_3_6_cau_hinh_mo_hinh_co_so",
        model_config_table(C.bang_3_6(train_cfg=train_cfg, seeds=seeds)),
    )

    tasks = ["binary", "multi"] if args.task == "both" else [args.task]
    for task in tasks:
        run_task(data, task, store, seeds, train_cfg, model_cfg)

    store.finalize(hardware=collect_hardware_info(), software=collect_software_info())

    print("\nĐối chiếu Phụ lục 3.A của Chương 3 để kiểm tra còn thiếu bảng nào.")
    if C.DRIVE_DIR.is_dir():
        if Path(store.run_dir).resolve().is_relative_to(C.DRIVE_DIR.resolve()):
            print(f"\nKết quả đã được ghi trực tiếp vào Drive: {store.run_dir}")
        else:
            print(
                f"\nNhớ copy kết quả về Drive:\n  "
                f"cp -r {store.run_dir} {C.DRIVE_DIR}/ketqua/"
            )


if __name__ == "__main__":
    main()
