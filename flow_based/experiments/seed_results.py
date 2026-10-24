"""Lưu bằng chứng từng seed và tổng hợp các mô hình trên cùng phép chia."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

SCALAR_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_macro",
    "roc_auc_weighted",
    "pr_auc_macro",
)
EXPECTED_MODELS = {
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "MLP",
    "FT-Transformer",
}


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def scalar_summary(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    return {
        "n": n,
        "mean": float(values.mean()) if n else None,
        "std": float(values.std(ddof=1)) if n > 1 else None,
        "min": float(values.min()) if n else None,
        "max": float(values.max()) if n else None,
        "ddof": 1,
    }


def save_seed_result(
    run_dir,
    task,
    model_name,
    seed,
    evaluation,
    result,
    y_test,
    *,
    split_seed,
    effective_config,
) -> dict:
    """Ghi ngay khi một mô hình xong; giữ số liệu nếu lần chạy sau bị ngắt."""
    root = Path(run_dir)
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    folder = root / "seeds" / task / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    prediction_path = folder / f"{slug}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        y_true=y_test,
        y_pred=result["y_pred"],
        y_proba=result["y_proba"],
    )
    metric_path = folder / f"{slug}.json"
    record = {
        "model": model_name,
        "seed": seed,
        "split_seed": split_seed,
        "task": task,
        "metrics": evaluation,
        "train_time_s": float(result["train_time_s"]),
        "device": result["device"],
        "n_params": result["n_params"],
        "config": effective_config,
        "history": result.get("history"),
        "best_epoch": result.get("best_epoch"),
        "n_epochs": result.get("n_epochs"),
        "predictions": prediction_path.relative_to(root).as_posix(),
    }
    model = result.get("model")
    if model is not None and hasattr(model, "state_dict"):
        import torch

        checkpoint = folder / f"{slug}_checkpoint.pt"
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "model": model_name,
                "seed": seed,
                "task": task,
                "config": effective_config,
                "best_epoch": result.get("best_epoch"),
            },
            checkpoint,
        )
        record["checkpoint"] = checkpoint.relative_to(root).as_posix()
    write_json(metric_path, record)
    record["metrics_file"] = metric_path.relative_to(root).as_posix()
    return record


def write_task_summary(
    run_dir, task, seeds, model_runs, *, split_seed, quick=False
) -> dict:
    """Không gộp các mô hình có danh sách seed thiếu hoặc khác nhau."""
    if set(model_runs) != EXPECTED_MODELS:
        raise ValueError(f"Cần đủ năm mô hình, hiện có: {sorted(model_runs)}")
    models = {}
    for name, runs in model_runs.items():
        actual_seeds = [r["seed"] for r in runs]
        if actual_seeds != seeds:
            raise ValueError(f"{name}: seeds={actual_seeds}, yêu cầu {seeds}")
        first = runs[0]["metrics"]
        per_class = {}
        for label, values in first["per_class"].items():
            supports = [r["metrics"]["per_class"][label]["support"] for r in runs]
            if len(set(supports)) != 1:
                raise ValueError(f"Support thay đổi giữa các seed: {name}/{label}")
            per_class[label] = {
                "support": values["support"],
                **{
                    metric: scalar_summary(
                        r["metrics"]["per_class"][label][metric] for r in runs
                    )
                    for metric in ("precision", "recall", "f1")
                },
            }
        models[name] = {
            "seeds": actual_seeds,
            "metrics": {
                metric: scalar_summary(r["metrics"].get(metric, np.nan) for r in runs)
                for metric in SCALAR_METRICS
            },
            "train_time_s": scalar_summary(r["train_time_s"] for r in runs),
            "per_class": per_class,
            "devices": [r["device"] for r in runs],
            "raw_results": [r["metrics_file"] for r in runs],
        }
    summary = {
        "task": task,
        "training_seeds": seeds,
        "split_seed": split_seed,
        "fixed_split": True,
        "std_ddof": 1,
        "quick": quick,
        "representative_seed": seeds[0],
        "representative_selection": "first requested seed, independent of test metrics",
        "models": models,
    }
    write_json(Path(run_dir) / f"summary_{task}.json", summary)
    return summary
