import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

import task4_1_dnn_only as t41

CONFIG = {
    "task2_outdir": "data_prepared_task2",
    "task4_outdir": "results_task4_1_dnn_only",
    "task6_3_outdir": "results_task6_3_feature_ablation",
    # 10 representative splits chosen from SHAP trends:
    # keep the original 5 and add 5 earlier / complementary regimes so the
    # family ablation is less clustered near the end of the sample.
    "selected_splits": [
        # original 5
        "split_032_2022-12-30_2025-04-04",
        "split_033_2023-04-03_2025-07-08",
        "split_034_2023-07-05_2025-10-06",
        "split_031_2022-09-30_2025-01-02",
        "split_008_2016-12-28_2019-04-01",
        # added 5 from SHAP outputs
        "split_001_2015-03-31_2017-06-28",
        "split_009_2017-03-30_2019-07-01",
        "split_010_2017-06-29_2019-09-30",
        "split_012_2017-12-28_2020-03-31",
        "split_024_2020-12-30_2023-03-31",
    ],
    "split_family_context": {
        # original 5
        "split_032_2022-12-30_2025-04-04": "return-level dominated from SHAP",
        "split_033_2023-04-03_2025-07-08": "volatility dominated from SHAP",
        "split_034_2023-07-05_2025-10-06": "momentum dominated from SHAP",
        "split_031_2022-09-30_2025-01-02": "RSI dominated from SHAP",
        "split_008_2016-12-28_2019-04-01": "MACD-signal dominated from SHAP",
        # added 5
        "split_001_2015-03-31_2017-06-28": "short-horizon return dominated from SHAP",
        "split_009_2017-03-30_2019-07-01": "10-day average-return dominated from SHAP",
        "split_010_2017-06-29_2019-09-30": "volume dominated from SHAP",
        "split_012_2017-12-28_2020-03-31": "volatility dominated from SHAP (crisis window)",
        "split_024_2020-12-30_2023-03-31": "60-day momentum dominated from SHAP",
    },
    "feature_families": {
        "return_level": ["ret_mean_5", "ret_mean_10", "ret_mean_20", "ret_mean_60"],
        "volatility": ["ret_std_5", "ret_std_10", "ret_std_20", "ret_std_60"],
        "momentum": ["mom_5", "mom_10", "mom_20", "mom_60"],
        "macd": ["macd_12_26", "macd_signal_9", "macd_hist"],
        "rsi": ["rsi_14"],
        "volume": ["volume", "volume_chg", "volume_roll_mean_20"],
    },
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_task2_split_map(task2_outdir: Path) -> Dict[str, Dict]:
    meta = t41._load_walk_forward_meta(task2_outdir)
    split_map: Dict[str, Dict] = {}
    for s in meta.get("splits", []):
        split_name = s.get("name") or s.get("split_name")
        if split_name is None:
            train_path = s.get("train_path", "")
            split_name = Path(train_path).parent.name if train_path else None
        if split_name is None:
            continue
        split_map[str(split_name)] = s
    return split_map


def _resolve_path(task2_outdir: Path, p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = task2_outdir / path
    return path


def _load_split_data(task2_outdir: Path, split_meta: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = _resolve_path(task2_outdir, split_meta["train_path"])
    test_path = _resolve_path(task2_outdir, split_meta["test_path"])
    train_df = t41._read_panel(train_path)
    test_df = t41._read_panel(test_path)
    full_panel = pd.concat([train_df, test_df], axis=0).sort_index()
    return train_df, test_df, full_panel


def _load_task4_split_metadata(task4_outdir: Path, split_name: str) -> Dict:
    meta_path = task4_outdir / "metadata" / f"{split_name}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Task 4.1 metadata for {split_name}: {meta_path}")
    return _load_json(meta_path)


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if float(np.std(y_true)) < 1e-12 or float(np.std(y_pred)) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _predict_with_model(
    model: tf.keras.Model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    ticker_categories: List[str],
    use_ticker_one_hot: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x_train_full, y_train_full, _ = t41.build_xy(
        train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=use_ticker_one_hot,
    )
    x_test, y_test, _ = t41.build_xy(
        test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=use_ticker_one_hot,
    )

    pred_train = model.predict(x_train_full, verbose=0).reshape(-1)
    pred_test = model.predict(x_test, verbose=0).reshape(-1)

    train_pred_df = train_df[[target_col]].copy().rename(columns={target_col: "y_true"})
    train_pred_df["y_pred"] = pred_train.astype(np.float32)
    train_pred_df["abs_error"] = np.abs(train_pred_df["y_true"] - train_pred_df["y_pred"]).astype(np.float32)
    train_pred_df["dataset_part"] = "train"

    test_pred_df = test_df[[target_col]].copy().rename(columns={target_col: "y_true"})
    test_pred_df["y_pred"] = pred_test.astype(np.float32)
    test_pred_df["abs_error"] = np.abs(test_pred_df["y_true"] - test_pred_df["y_pred"]).astype(np.float32)
    test_pred_df["dataset_part"] = "test"

    combined_pred_df = pd.concat([train_pred_df, test_pred_df], axis=0).sort_index()
    combined_pred_df["signal_for_trade"] = combined_pred_df.groupby(level="ticker")["y_pred"].shift(1).astype(np.float32)

    diagnostics = {
        "mae": float(np.mean(np.abs(y_test - pred_test))),
        "rmse": _rmse(y_test, pred_test),
        "direction_acc": _directional_accuracy(y_test, pred_test),
        "pred_real_corr": _safe_corr(y_test, pred_test),
        "train_mae": float(np.mean(np.abs(y_train_full - pred_train))),
        "train_rmse": _rmse(y_train_full, pred_train),
        "n_train_rows": int(len(y_train_full)),
        "n_test_rows": int(len(y_test)),
    }
    return combined_pred_df, diagnostics


def _evaluate_portfolio(full_panel: pd.DataFrame, combined_pred_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    prices = t41._panel_to_prices(full_panel)
    test_dates = test_df.index.get_level_values("date")
    test_start = pd.Timestamp(test_dates.min())
    test_end = pd.Timestamp(test_dates.max())

    gross, net, turnover_total, tc_total, n_reb, reb_df = t41.simulate_dnn_signal_portfolio(
        full_prices=prices,
        pred_df=combined_pred_df,
        test_start=test_start,
        test_end=test_end,
        rebalance_every_days=int(t41.CONFIG["rebalance_every_days"]),
        transaction_cost_rate=float(t41.CONFIG["transaction_cost_rate"]),
        max_weight=float(t41.CONFIG["weight_bounds"][1]),
    )
    metrics = t41.compute_metrics(
        net,
        frequency=int(t41.CONFIG["frequency"]),
        risk_free_rate=float(t41.CONFIG["risk_free_rate"]),
    )
    out = {
        "net_ann_return": metrics["ann_return"],
        "net_ann_vol": metrics["ann_vol"],
        "net_sharpe": metrics["sharpe"],
        "net_sortino": metrics["sortino"],
        "net_cumulative_return": metrics["cumulative_return"],
        "net_max_drawdown": metrics["max_drawdown"],
        "turnover_total": float(turnover_total),
        "tc_total": float(tc_total),
        "n_reb": int(n_reb),
    }
    return out, reb_df


def _train_ablated_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    ticker_categories: List[str],
    use_ticker_one_hot: bool,
) -> tf.keras.Model:
    t41.set_seed(int(t41.CONFIG["seed"]))
    train_part, val_part = t41.time_based_train_val_split(train_df, float(t41.CONFIG["validation_ratio"]))

    x_train_fit, y_train_fit, _ = t41.build_xy(
        train_part,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=use_ticker_one_hot,
    )
    x_val, y_val, _ = t41.build_xy(
        val_part,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=use_ticker_one_hot,
    )

    model = t41.build_dnn(input_dim=x_train_fit.shape[1])
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(t41.CONFIG["patience"]),
            restore_best_weights=True,
        )
    ]
    model.fit(
        x_train_fit,
        y_train_fit,
        validation_data=(x_val, y_val),
        epochs=int(t41.CONFIG["epochs"]),
        batch_size=int(t41.CONFIG["batch_size"]),
        verbose=int(t41.CONFIG["verbose_fit"]),
        callbacks=cb,
        shuffle=False,
    )
    return model


def _meaningful_label(name: str) -> str:
    mapping = {
        "return_level": "Return-level family",
        "volatility": "Volatility family",
        "momentum": "Momentum family",
        "macd": "MACD family",
        "rsi": "RSI family",
        "volume": "Volume family",
        "full": "Full model",
    }
    return mapping.get(name, name)


def _safe_delta(full_val: float, ablated_val: float, higher_is_better: bool = True) -> float:
    if pd.isna(full_val) or pd.isna(ablated_val):
        return np.nan
    return float(full_val - ablated_val) if higher_is_better else float(ablated_val - full_val)


def save_family_delta_plot(summary_df: pd.DataFrame, outpath: Path) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["family_label"], plot_df["delta_net_sharpe_mean"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean Sharpe drop vs full")
    plt.title("Task 6.3 family ablation: mean Sharpe degradation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    task2_outdir = Path(CONFIG["task2_outdir"])
    task4_outdir = Path(CONFIG["task4_outdir"])
    outdir = Path(CONFIG["task6_3_outdir"])
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    _ensure_dir(outdir)
    _ensure_dir(tables_dir)
    _ensure_dir(figures_dir)

    split_map = _load_task2_split_map(task2_outdir)
    selected_splits = list(CONFIG["selected_splits"])

    results_rows: List[Dict[str, object]] = []

    print(f"Selected {len(selected_splits)} split(s) for Task 6.3 family ablation.")
    split_family_context = dict(CONFIG.get("split_family_context", {}))
    for s in selected_splits:
        context = split_family_context.get(s)
        if context:
            print(f"  - {s} [{context}]")
        else:
            print(f"  - {s}")

    for split_name in selected_splits:
        if split_name not in split_map:
            print(f"Skipping missing split metadata: {split_name}")
            continue

        split_context = split_family_context.get(split_name)
        if split_context:
            print(f"\nRunning Task 6.3 on split: {split_name} [{split_context}]")
        else:
            print(f"\nRunning Task 6.3 on split: {split_name}")
        split_meta_task2 = split_map[split_name]
        train_df, test_df, full_panel = _load_split_data(task2_outdir, split_meta_task2)
        split_meta_task4 = _load_task4_split_metadata(task4_outdir, split_name)

        base_feature_cols = list(split_meta_task4["feature_cols"])
        target_col = str(split_meta_task4["target_col"])
        ticker_categories = list(split_meta_task4["ticker_categories"])
        use_ticker_one_hot = bool(split_meta_task4["use_ticker_one_hot"])

        # Full baseline from saved model
        full_model_path = task4_outdir / "models" / f"{split_name}.keras"
        if not full_model_path.exists():
            raise FileNotFoundError(f"Missing saved Task 4.1 model for {split_name}: {full_model_path}")
        full_model = tf.keras.models.load_model(full_model_path)
        full_pred_df, full_diag = _predict_with_model(
            full_model, train_df, test_df, base_feature_cols, target_col, ticker_categories, use_ticker_one_hot
        )
        full_port, _ = _evaluate_portfolio(full_panel, full_pred_df, test_df)
        full_row = {
            "split": split_name,
            "variant": "full",
            "family": "full",
            "family_label": _meaningful_label("full"),
            "removed_features": "",
            "n_removed_features": 0,
            **full_diag,
            **full_port,
        }
        results_rows.append(full_row)
        if split_context:
            print(f"  Dominant SHAP family reference: {split_context}")
        print(f"  Full model: sharpe={full_port['net_sharpe']:.3f}, sortino={full_port['net_sortino']:.3f}, rmse={full_diag['rmse']:.6f}")

        for family_name, fam_features in CONFIG["feature_families"].items():
            removed = [f for f in fam_features if f in base_feature_cols]
            kept = [f for f in base_feature_cols if f not in removed]
            if not removed:
                continue
            if not kept:
                continue

            ablated_model = _train_ablated_model(
                train_df=train_df,
                feature_cols=kept,
                target_col=target_col,
                ticker_categories=ticker_categories,
                use_ticker_one_hot=use_ticker_one_hot,
            )
            ab_pred_df, ab_diag = _predict_with_model(
                ablated_model, train_df, test_df, kept, target_col, ticker_categories, use_ticker_one_hot
            )
            ab_port, _ = _evaluate_portfolio(full_panel, ab_pred_df, test_df)
            row = {
                "split": split_name,
                "variant": f"without_{family_name}",
                "family": family_name,
                "family_label": _meaningful_label(family_name),
                "removed_features": ", ".join(removed),
                "n_removed_features": len(removed),
                **ab_diag,
                **ab_port,
            }
            results_rows.append(row)
            print(
                f"  Without {family_name}: sharpe={ab_port['net_sharpe']:.3f}, "
                f"sortino={ab_port['net_sortino']:.3f}, rmse={ab_diag['rmse']:.6f}"
            )

    results_df = pd.DataFrame(results_rows)
    if results_df.empty:
        raise RuntimeError("No ablation results were generated.")

    results_df.to_csv(outdir / "task6_3_ablation_results_long.csv", index=False)

    full_df = results_df[results_df["family"] == "full"].copy().set_index("split")
    ablated_df = results_df[results_df["family"] != "full"].copy()
    delta_rows: List[Dict[str, object]] = []
    for _, row in ablated_df.iterrows():
        split = row["split"]
        if split not in full_df.index:
            continue
        full = full_df.loc[split]
        delta_rows.append({
            "split": split,
            "family": row["family"],
            "family_label": row["family_label"],
            "delta_mae": _safe_delta(full["mae"], row["mae"], higher_is_better=False),
            "delta_rmse": _safe_delta(full["rmse"], row["rmse"], higher_is_better=False),
            "delta_direction_acc": _safe_delta(full["direction_acc"], row["direction_acc"], higher_is_better=True),
            "delta_net_ann_return": _safe_delta(full["net_ann_return"], row["net_ann_return"], higher_is_better=True),
            "delta_net_sharpe": _safe_delta(full["net_sharpe"], row["net_sharpe"], higher_is_better=True),
            "delta_net_sortino": _safe_delta(full["net_sortino"], row["net_sortino"], higher_is_better=True),
            "delta_net_max_drawdown": _safe_delta(full["net_max_drawdown"], row["net_max_drawdown"], higher_is_better=False),
            "delta_turnover_total": _safe_delta(full["turnover_total"], row["turnover_total"], higher_is_better=False),
            "delta_tc_total": _safe_delta(full["tc_total"], row["tc_total"], higher_is_better=False),
        })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(tables_dir / "task6_3_ablation_deltas.csv", index=False)

    summary_rows = []
    if not delta_df.empty:
        grouped = delta_df.groupby(["family", "family_label"], dropna=False)
        for (family, family_label), g in grouped:
            summary_rows.append({
                "family": family,
                "family_label": family_label,
                "n_splits": int(g["split"].nunique()),
                "delta_mae_mean": float(g["delta_mae"].mean()),
                "delta_rmse_mean": float(g["delta_rmse"].mean()),
                "delta_direction_acc_mean": float(g["delta_direction_acc"].mean()),
                "delta_net_ann_return_mean": float(g["delta_net_ann_return"].mean()),
                "delta_net_sharpe_mean": float(g["delta_net_sharpe"].mean()),
                "delta_net_sortino_mean": float(g["delta_net_sortino"].mean()),
                "delta_net_max_drawdown_mean": float(g["delta_net_max_drawdown"].mean()),
                "delta_turnover_total_mean": float(g["delta_turnover_total"].mean()),
                "delta_tc_total_mean": float(g["delta_tc_total"].mean()),
            })
    summary_df = pd.DataFrame(summary_rows).sort_values("delta_net_sharpe_mean", ascending=False)
    summary_df.to_csv(tables_dir / "task6_3_ablation_summary_by_family.csv", index=False)

    # plain-English summary
    pe_rows = []
    for _, row in summary_df.iterrows():
        pe_rows.append({
            "family": row["family"],
            "family_label": row["family_label"],
            "summary": (
                f"Removing the {row['family_label'].lower()} reduced average Sharpe by {row['delta_net_sharpe_mean']:.4f}, "
                f"Sortino by {row['delta_net_sortino_mean']:.4f}, and changed RMSE by {row['delta_rmse_mean']:.6f} "
                f"across {int(row['n_splits'])} representative split(s)."
            ),
        })
    pd.DataFrame(pe_rows).to_csv(tables_dir / "task6_3_plain_english_summary.csv", index=False)

    save_family_delta_plot(summary_df, figures_dir / "task6_3_family_delta_sharpe.png")

    print("\nTask 6.3 family ablation summary:")
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            print(
                f"  {row['family_label']}: mean Sharpe drop={row['delta_net_sharpe_mean']:.4f}, "
                f"mean Sortino drop={row['delta_net_sortino_mean']:.4f}, mean RMSE increase={row['delta_rmse_mean']:.6f}"
            )

    print("\nDone.")
    print(f"Saved long results:   {outdir / 'task6_3_ablation_results_long.csv'}")
    print(f"Saved delta tables:   {tables_dir / 'task6_3_ablation_deltas.csv'}")
    print(f"Saved summary table:  {tables_dir / 'task6_3_ablation_summary_by_family.csv'}")
    print(f"Saved figures in:     {figures_dir}")


if __name__ == "__main__":
    main()
