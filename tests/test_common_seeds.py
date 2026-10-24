"""Kiểm tra giao thức seed bằng dữ liệu giả; không phải kết quả thực nghiệm."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flow_based import config as C
from flow_based.experiments.run_all import run_task
from flow_based.experiments import run_all as runner
from flow_based.experiments.seed_results import scalar_summary, write_task_summary
from flow_based.reporting.report_export import ResultStore, aggregate_seeds
from flow_based.training import baselines as B


def synthetic_data():
    rng = np.random.default_rng(987)
    data = {
        "meta": {"seed": 42},
        "n_features": 6,
        "n_classes_multi": 4,
        "class_names_multi": C.CLASS_NAMES,
        "class_names_bin": C.BINARY_NAMES,
    }
    for part, n in (("train", 80), ("val", 40), ("test", 40)):
        y = np.arange(n, dtype=np.int64) % 4
        X = rng.normal(size=(n, 6)).astype(np.float32)
        X[:, 0] += y * 2
        data[f"X_{part}"] = X
        data[f"y_multi_{part}"] = y
        data[f"y_bin_{part}"] = (y != 0).astype(np.int64)
    return data


class CommonSeedsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_sample_sd_and_one_seed(self):
        self.assertAlmostEqual(scalar_summary([1, 2, 3])["std"], 1.0)
        self.assertIsNone(scalar_summary([1])["std"])
        table = aggregate_seeds([{"f1_macro": 0.5}], metrics=("f1_macro",))
        self.assertTrue(np.isnan(table.iloc[0]["Độ lệch chuẩn"]))
        self.assertIn("n<2", table.iloc[0]["Báo cáo dạng"])

    def test_actual_baseline_random_states(self):
        data = synthetic_data()
        X, y, test = data["X_train"], data["y_bin_train"], data["X_test"]
        original = (dict(C.LOGREG_PARAMS), dict(C.RF_PARAMS), dict(C.XGB_PARAMS))
        with (
            patch.dict(C.LOGREG_PARAMS, {"max_iter": 200, "n_jobs": 1}),
            patch.dict(C.RF_PARAMS, {"n_estimators": 3, "n_jobs": 1}),
            patch.dict(C.XGB_PARAMS, {"n_estimators": 3, "n_jobs": 1}),
        ):
            for seed in C.SEEDS:
                runs = (
                    B.run_logreg(X, y, test, False, seed=seed),
                    B.run_random_forest(X, y, test, False, seed=seed),
                    B.run_xgboost(X, y, test, 2, False, seed=seed),
                )
                for result in runs:
                    self.assertEqual(result["model"].get_params()["random_state"], seed)
        self.assertEqual(original, (C.LOGREG_PARAMS, C.RF_PARAMS, C.XGB_PARAMS))

    def test_mlp_seed_covers_initialization_and_quick_epochs(self):
        data = synthetic_data()
        cfg = {**C.TRAIN, "max_epochs": 1, "batch_size": 32}

        def fit(seed):
            torch.rand(31)  # Nhiễu trạng thái global từ mô hình trước đó.
            return B.run_mlp(
                data["X_train"],
                data["y_bin_train"],
                data["X_val"],
                data["y_bin_val"],
                data["X_test"],
                2,
                seed=seed,
                train_cfg=cfg,
                verbose=False,
            )

        with (
            patch.dict(C.MLP, {"hidden_dims": [12]}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            first, repeated, different = fit(42), fit(42), fit(1337)
        for key, value in first["model"].state_dict().items():
            torch.testing.assert_close(
                value, repeated["model"].state_dict()[key], rtol=0, atol=0
            )
        self.assertFalse(np.array_equal(first["y_proba"], different["y_proba"]))
        self.assertEqual(first["n_epochs"], 1)
        self.assertEqual(len(first["history"]["train_loss"]), 1)

    def test_missing_model_or_seed_is_not_reported_complete(self):
        with tempfile.TemporaryDirectory(prefix="common_seed_incomplete_") as tmp:
            with self.assertRaises(ValueError):
                write_task_summary(tmp, "multi", C.SEEDS, {}, split_seed=42)
            names = (
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "MLP",
                "FT-Transformer",
            )
            with self.assertRaises(ValueError):
                write_task_summary(
                    tmp, "multi", C.SEEDS, {n: [] for n in names}, split_seed=42
                )
            self.assertFalse((Path(tmp) / "summary_multi.json").exists())

    def test_quick_cli_uses_one_seed_and_three_epochs_for_both_neural_models(self):
        with (
            tempfile.TemporaryDirectory(prefix="common_seed_quick_") as tmp,
            patch.object(runner, "load_dataset", return_value=synthetic_data()),
            patch.object(runner, "collect_hardware_info", return_value=None),
            patch.object(runner, "collect_software_info", return_value=None),
            patch.object(C, "RESULTS_DIR", Path(tmp)),
            patch.object(
                sys,
                "argv",
                ["run_all", "--quick", "--task", "multi", "--run-name", "QUICK_ONLY"],
            ),
            patch.dict(C.LOGREG_PARAMS, {"max_iter": 200, "n_jobs": 1}),
            patch.dict(C.RF_PARAMS, {"n_estimators": 3, "n_jobs": 1}),
            patch.dict(C.XGB_PARAMS, {"n_estimators": 3, "n_jobs": 1}),
            patch.dict(C.MLP, {"hidden_dims": [12]}),
            patch.dict(
                C.MODEL, {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32}
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runner.main()
            root = Path(tmp) / "QUICK_ONLY"
            summary = json.loads(
                (root / "summary_multi.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["quick"])
            self.assertEqual(summary["training_seeds"], [42])
            for name, model in summary["models"].items():
                self.assertIsNone(model["metrics"]["f1_macro"]["std"])
                if name in ("MLP", "FT-Transformer"):
                    raw = json.loads(
                        (root / model["raw_results"][0]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(raw["n_epochs"], 3)
                    self.assertEqual(raw["config"]["train"]["max_epochs"], 3)

    def test_duplicate_seeds_rejected_before_loading_data(self):
        with (
            patch.object(sys, "argv", ["run_all", "--seeds", "42", "42"]),
            patch.object(runner, "load_dataset") as load,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as caught:
                runner.main()
        self.assertEqual(caught.exception.code, 2)
        load.assert_not_called()

    def test_all_five_models_both_tasks_three_seeds(self):
        data = synthetic_data()
        arrays = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
        train_cfg = {
            **C.TRAIN,
            "max_epochs": 1,
            "early_stop_patience": 1,
            "batch_size": 32,
        }
        model_cfg = {**C.MODEL, "d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32}
        with (
            tempfile.TemporaryDirectory(prefix="common_seed_integration_") as tmp,
            patch.dict(C.LOGREG_PARAMS, {"max_iter": 200, "n_jobs": 1}),
            patch.dict(C.RF_PARAMS, {"n_estimators": 3, "n_jobs": 1}),
            patch.dict(C.XGB_PARAMS, {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}),
            patch.dict(C.MLP, {"hidden_dims": [12]}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            store = ResultStore(tmp, run_name="SYNTHETIC_VALIDATION_ONLY")
            store.config = {"quick": True, "seeds": C.SEEDS}
            for task in ("binary", "multi"):
                result = run_task(data, task, store, C.SEEDS, train_cfg, model_cfg)
                root = Path(store.run_dir)
                summary = json.loads(
                    (root / f"summary_{task}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(summary["representative_seed"], 42)
                self.assertEqual(len(summary["models"]), 5)
                self.assertEqual(len(result["comparison"]), 5)
                self.assertTrue((result["comparison"]["Số seed"] == 3).all())
                for name, model in summary["models"].items():
                    self.assertEqual(model["seeds"], C.SEEDS)
                    self.assertEqual(model["metrics"]["f1_macro"]["n"], 3)
                    raw_f1 = []
                    for seed, filename in zip(C.SEEDS, model["raw_results"]):
                        raw = json.loads((root / filename).read_text(encoding="utf-8"))
                        self.assertEqual(raw["seed"], seed)
                        self.assertEqual(raw["split_seed"], 42)
                        raw_f1.append(raw["metrics"]["f1_macro"])
                        with np.load(root / raw["predictions"]) as pred:
                            np.testing.assert_array_equal(
                                pred["y_true"],
                                data[
                                    f"y_{'bin' if task == 'binary' else 'multi'}_test"
                                ],
                            )
                        if name in ("MLP", "FT-Transformer"):
                            self.assertEqual(raw["n_epochs"], 1)
                            self.assertEqual(raw["config"]["train"]["max_epochs"], 1)
                        else:
                            self.assertEqual(raw["config"]["random_state"], seed)
                    self.assertAlmostEqual(
                        model["metrics"]["f1_macro"]["mean"], np.mean(raw_f1)
                    )
                    self.assertAlmostEqual(
                        model["metrics"]["f1_macro"]["std"], np.std(raw_f1, ddof=1)
                    )
                    self.assertAlmostEqual(
                        result["results"][name]["f1_macro"], np.mean(raw_f1)
                    )
            store.finalize()
            metrics = json.loads(
                (Path(store.run_dir) / "metrics_raw.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(metrics), 30)
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, data[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
