"""
task4_1_dnn_only.py

Task 4.1: DNN-only pilot for portfolio optimisation
- Loads leakage-safe walk-forward splits produced by Task 2
- Trains a supervised DNN per split to predict next-period log returns
- Uses a smaller pilot subset of splits first to test computational feasibility
- Converts DNN signals into long-only, fully invested portfolio weights
- Backtests the DNN-only portfolio with the same evaluation language as Task 3
- Saves prediction diagnostics, portfolio metrics, rebalances, curves, and report-friendly plots/tables

Important
- This stage is intentionally DNN-only. RL is not included here.
- By default, trained model binaries are NOT saved to avoid repo bloat.
- This is academic tooling, not financial advice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
except Exception as e:
    raise SystemExit(
        "TensorFlow is required for Task 4.1.\n"
        "Install with: pip install tensorflow\n"
        f"Original error: {e}"
    )


# =========================
# CONFIG (edit this)
# =========================
CONFIG = {
    # Input from Task 2 #
    "task2_outdir": "data_prepared_task2",

    # Output for Task 4.1
    "task4_outdir": "results_task4_1_dnn_only",

    # Scope: set pilot_mode=False to run all Task 2 splits so this matches Task 3 exactly
    "pilot_mode": False,
    "pilot_recent_splits": 6,

    # Forecast plot context
    "prediction_plot_context_days": 60,

    # Feature selection
    "exclude_feature_cols": ["price"],
    "target_col": "target_next_log_return",
    "use_ticker_one_hot": True,

    # DNN architecture / training
    "seed": 42,
    "hidden_units": [64, 32],
    "dropout_rate": 0.10,
    "l2_reg": 1e-5,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 40,
    "patience": 6,
    "validation_ratio": 0.20,
    "verbose_fit": 0,

    # Portfolio construction from DNN signals
    "weight_bounds": (0.0, 0.60),   # same cap style as Task 3
    "rebalance_every_days": 21,
    "transaction_cost_rate": 0.001,
    "risk_free_rate": 0.0,
    "frequency": 252,

    # Artifact saving
    "save_models": False,
    "save_predictions": True,
}


# =========================
# IO helpers
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_panel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        cols = [c.lower() for c in raw.columns]
        if "date" in cols and "ticker" in cols:
            date_col = raw.columns[cols.index("date")]
            ticker_col = raw.columns[cols.index("ticker")]
            df = raw.set_index([date_col, ticker_col])
        else:
            df = raw.set_index([raw.columns[0], raw.columns[1]])
        df.index.set_names(["date", "ticker"], inplace=True)
        df.index = df.index.set_levels(
            [pd.to_datetime(df.index.levels[0]), df.index.levels[1]],
            level=[0, 1],
        )
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if not np.issubdtype(df.index.get_level_values("date").dtype, np.datetime64):
        df = df.copy()
        df.index = df.index.set_levels(
            [pd.to_datetime(df.index.levels[0]), df.index.levels[1]],
            level=[0, 1],
        )
    return df.sort_index()


def _load_walk_forward_meta(task2_outdir: Path) -> Dict:
    meta_path = task2_outdir / "walk_forward_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}. Run Task 2 first.")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _panel_to_prices(panel: pd.DataFrame) -> pd.DataFrame:
    if "price" not in panel.columns:
        raise KeyError("Panel is missing required 'price' column.")
    return panel["price"].unstack("ticker").sort_index()


# =========================
# Metrics and portfolio helpers
# =========================
def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def compute_metrics(port_rets: pd.Series, frequency: int, risk_free_rate: float) -> Dict[str, float]:
    port_rets = port_rets.dropna()
    if port_rets.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
        }

    equity = (1.0 + port_rets).cumprod()
    ann_return = float((equity.iloc[-1] ** (frequency / len(port_rets))) - 1.0)
    ann_vol = float(port_rets.std(ddof=1) * np.sqrt(frequency))

    rf_daily = (1.0 + risk_free_rate) ** (1.0 / frequency) - 1.0
    excess = port_rets - rf_daily
    sharpe = float(excess.mean() / (port_rets.std(ddof=1) + 1e-12) * np.sqrt(frequency))

    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    sortino = float(excess.mean() / (downside_dev + 1e-12) * np.sqrt(frequency))

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": _max_drawdown(equity),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if float(np.std(y_true)) < 1e-12 or float(np.std(y_pred)) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _make_rebalance_dates(test_dates: pd.DatetimeIndex, rebalance_every_days: int) -> List[pd.Timestamp]:
    if rebalance_every_days <= 0:
        return [pd.Timestamp(test_dates[0])]
    out: List[pd.Timestamp] = []
    i = 0
    n = len(test_dates)
    while i < n:
        out.append(pd.Timestamp(test_dates[i]))
        i += rebalance_every_days
    return out


def _cap_and_normalise_weights(w: pd.Series, max_weight: float) -> pd.Series:
    w = w.clip(lower=0.0).astype(float)
    if float(w.sum()) <= 0.0:
        w[:] = 1.0 / len(w)
        return w

    w = w / w.sum()
    max_weight = float(max_weight)

    for _ in range(10):
        over = w > (max_weight + 1e-12)
        if not bool(over.any()):
            break

        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if int(under.sum()) == 0:
            w[:] = 1.0 / len(w)
            break

        base = w[under]
        if float(base.sum()) <= 1e-12:
            w[under] = w[under] + (excess / int(under.sum()))
        else:
            w[under] = w[under] + excess * (base / base.sum())
        w = w / w.sum()

    return w / w.sum()


def signal_to_weights(signal: pd.Series, tickers: List[str], max_weight: float) -> pd.Series:
    s = signal.reindex(tickers).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    positive = s.clip(lower=0.0)
    if float(positive.sum()) <= 1e-12:
        w = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    else:
        w = positive / positive.sum()

    return _cap_and_normalise_weights(w, max_weight=max_weight)


# =========================
# DNN helpers
# =========================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def select_splits(meta_splits: List[Dict], pilot_mode: bool, pilot_recent_splits: int) -> List[Dict]:
    if not pilot_mode:
        return meta_splits
    n = int(max(1, pilot_recent_splits))
    return meta_splits[-n:]


def time_based_train_val_split(df: pd.DataFrame, validation_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df.index.get_level_values("date").unique())
    if len(unique_dates) < 10:
        return df.copy(), df.copy()

    n_val_dates = max(1, int(round(len(unique_dates) * validation_ratio)))
    n_val_dates = min(n_val_dates, len(unique_dates) - 1)
    cutoff_date = unique_dates[-n_val_dates]

    train_part = df.loc[(slice(None, cutoff_date - pd.Timedelta(days=1)), slice(None)), :].copy()
    val_part = df.loc[(slice(cutoff_date, None), slice(None)), :].copy()

    if train_part.empty or val_part.empty:
        split_idx = max(1, int(len(unique_dates) * (1.0 - validation_ratio)))
        split_idx = min(split_idx, len(unique_dates) - 1)
        cutoff_date = unique_dates[split_idx]
        train_part = df.loc[(slice(None, cutoff_date - pd.Timedelta(days=1)), slice(None)), :].copy()
        val_part = df.loc[(slice(cutoff_date, None), slice(None)), :].copy()

    return train_part, val_part


def build_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    ticker_categories: List[str],
    use_ticker_one_hot: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_num = df[feature_cols].astype(np.float32).reset_index(drop=True)
    feature_names = list(feature_cols)

    if use_ticker_one_hot:
        tickers = pd.Categorical(df.index.get_level_values("ticker"), categories=ticker_categories)
        x_ticker = pd.get_dummies(tickers, prefix="ticker", dtype=np.float32)
        x_df = pd.concat([x_num, x_ticker.reset_index(drop=True)], axis=1)
        feature_names.extend(list(x_ticker.columns))
    else:
        x_df = x_num

    y = df[target_col].astype(np.float32).values
    x = x_df.astype(np.float32).values
    return x, y, feature_names


def build_dnn(input_dim: int) -> tf.keras.Model:
    hidden_units = [int(x) for x in CONFIG["hidden_units"]]
    dropout_rate = float(CONFIG["dropout_rate"])
    l2_reg = float(CONFIG["l2_reg"])
    lr = float(CONFIG["learning_rate"])

    model = models.Sequential(name="task4_1_dnn_only")
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_and_predict_split(
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    ticker_categories: List[str],
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, object]]:
    history_dir = outdir / "histories"
    models_dir = outdir / "models"
    preds_dir = outdir / "predictions"
    metadata_dir = outdir / "metadata"
    _ensure_dir(history_dir)
    _ensure_dir(models_dir)
    _ensure_dir(preds_dir)
    _ensure_dir(metadata_dir)

    train_part, val_part = time_based_train_val_split(train_df, float(CONFIG["validation_ratio"]))

    x_train_fit, y_train_fit, feature_names = build_xy(
        train_part,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=bool(CONFIG["use_ticker_one_hot"]),
    )
    x_val, y_val, _ = build_xy(
        val_part,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=bool(CONFIG["use_ticker_one_hot"]),
    )
    x_train_full, y_train_full, _ = build_xy(
        train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=bool(CONFIG["use_ticker_one_hot"]),
    )
    x_test, y_test, _ = build_xy(
        test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        ticker_categories=ticker_categories,
        use_ticker_one_hot=bool(CONFIG["use_ticker_one_hot"]),
    )

    model = build_dnn(input_dim=x_train_fit.shape[1])

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(CONFIG["patience"]),
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train_fit,
        y_train_fit,
        validation_data=(x_val, y_val),
        epochs=int(CONFIG["epochs"]),
        batch_size=int(CONFIG["batch_size"]),
        verbose=int(CONFIG["verbose_fit"]),
        callbacks=cb,
        shuffle=False,
    )

    if bool(CONFIG["save_models"]):
        model.save(models_dir / f"{split_name}.keras")

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
    combined_pred_df["signal_for_trade"] = (
        combined_pred_df.groupby(level="ticker")["y_pred"].shift(1).astype(np.float32)
    )

    train_pred_df = combined_pred_df[combined_pred_df["dataset_part"] == "train"].copy()
    test_pred_df = combined_pred_df[combined_pred_df["dataset_part"] == "test"].copy()

    if bool(CONFIG["save_predictions"]):
        train_pred_df.to_csv(preds_dir / f"{split_name}_train_predictions.csv")
        test_pred_df.to_csv(preds_dir / f"{split_name}_test_predictions.csv")
        combined_pred_df.to_csv(preds_dir / f"{split_name}_combined_predictions.csv")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(history_dir / f"{split_name}_history.csv", index=False)

    diagnostics = {
        "mae": float(np.mean(np.abs(y_test - pred_test))),
        "rmse": _rmse(y_test, pred_test),
        "direction_acc": _directional_accuracy(y_test, pred_test),
        "pred_real_corr": _safe_corr(y_test, pred_test),
        "train_mae": float(np.mean(np.abs(y_train_full - pred_train))),
        "train_rmse": _rmse(y_train_full, pred_train),
        "n_train_rows": int(len(y_train_full)),
        "n_test_rows": int(len(y_test)),
        "n_features": int(x_train_fit.shape[1]),
        "n_epochs_trained": int(len(hist_df)),
    }

    split_meta = {
        "split": split_name,
        "seed": int(CONFIG["seed"]),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "ticker_categories": ticker_categories,
        "use_ticker_one_hot": bool(CONFIG["use_ticker_one_hot"]),
        "n_features": int(len(feature_names)),
        "n_train_rows": int(len(train_pred_df)),
        "n_test_rows": int(len(test_pred_df)),
        "model_config": {
            "hidden_units": [int(x) for x in CONFIG["hidden_units"]],
            "dropout_rate": float(CONFIG["dropout_rate"]),
            "l2_reg": float(CONFIG["l2_reg"]),
            "learning_rate": float(CONFIG["learning_rate"]),
            "batch_size": int(CONFIG["batch_size"]),
            "epochs": int(CONFIG["epochs"]),
            "patience": int(CONFIG["patience"]),
            "validation_ratio": float(CONFIG["validation_ratio"]),
        },
    }
    with open(metadata_dir / f"{split_name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    return train_pred_df, test_pred_df, hist_df, diagnostics | {"n_features": int(len(feature_names))}, split_meta


# =========================
# Backtest DNN-only portfolio
# =========================
def simulate_dnn_signal_portfolio(
    full_prices: pd.DataFrame,
    pred_df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    rebalance_every_days: int,
    transaction_cost_rate: float,
    max_weight: float,
) -> Tuple[pd.Series, pd.Series, float, float, int, pd.DataFrame]:
    full_prices = full_prices.sort_index()
    test_prices = full_prices.loc[test_start:test_end]
    test_dates = test_prices.index
    if len(test_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0.0, 0.0, 0, pd.DataFrame()

    tickers = list(full_prices.columns)
    asset_rets = full_prices.pct_change()
    rebalance_dates = _make_rebalance_dates(test_dates, rebalance_every_days)

    gross_parts: List[pd.Series] = []
    net_parts: List[pd.Series] = []
    debug_rows: List[Dict[str, float]] = []

    prev_w: pd.Series | None = None
    turnover_total = 0.0

    for j, reb_date in enumerate(rebalance_dates):
        if j < len(rebalance_dates) - 1:
            hold_end = rebalance_dates[j + 1] - pd.Timedelta(days=1)
            hold_end = test_dates[test_dates <= hold_end][-1]
        else:
            hold_end = pd.Timestamp(test_dates[-1])

        try:
            signal_today = pred_df.xs(reb_date, level="date")["signal_for_trade"]
        except Exception:
            signal_today = pd.Series(index=tickers, data=np.nan)

        w = signal_to_weights(signal_today, tickers=tickers, max_weight=max_weight)

        if prev_w is None:
            turnover = float(np.abs(w).sum())
        else:
            turnover = float(np.abs(w - prev_w).sum())
        tc = float(transaction_cost_rate * turnover)
        turnover_total += turnover

        period_rets = asset_rets.loc[reb_date:hold_end].dropna(how="any")
        if period_rets.empty:
            prev_w = w
            continue

        port_gross = period_rets.dot(w)
        port_net = port_gross.copy()
        port_net.iloc[0] = port_net.iloc[0] - tc

        gross_parts.append(port_gross)
        net_parts.append(port_net)

        row = {
            "rebalance_date": str(pd.Timestamp(reb_date).date()),
            "turnover": turnover,
            "tc": tc,
        }
        for t in tickers:
            row[f"signal_{t}"] = float(signal_today.reindex(tickers).fillna(0.0).get(t, 0.0))
            row[f"w_{t}"] = float(w.get(t, 0.0))
        debug_rows.append(row)

        prev_w = w

    gross = pd.concat(gross_parts).sort_index() if gross_parts else pd.Series(dtype=float)
    net = pd.concat(net_parts).sort_index() if net_parts else pd.Series(dtype=float)
    gross = gross.loc[test_start:test_end]
    net = net.loc[test_start:test_end]

    tc_total = float(transaction_cost_rate * turnover_total)
    debug_df = pd.DataFrame(debug_rows)
    return gross, net, turnover_total, tc_total, len(rebalance_dates), debug_df


# =========================
# Report-friendly plots/tables
# =========================
def save_history_plot(hist_df: pd.DataFrame, outpath: Path, split_name: str) -> None:
    if hist_df.empty:
        return
    plt.figure(figsize=(8, 4.5))
    if "loss" in hist_df.columns:
        plt.plot(hist_df.index + 1, hist_df["loss"], label="Train loss")
    if "val_loss" in hist_df.columns:
        plt.plot(hist_df.index + 1, hist_df["val_loss"], label="Validation loss")
    plt.title(f"Task 4.1 DNN loss: {split_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_equity_plot(eq_df: pd.DataFrame, outpath: Path, split_name: str) -> None:
    if eq_df.empty:
        return
    plt.figure(figsize=(8, 4.5))
    for col in eq_df.columns:
        plt.plot(eq_df.index, eq_df[col], label=col)
    plt.title(f"Task 4.1 DNN-only equity curve: {split_name}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_weights_plot(reb_df: pd.DataFrame, outpath: Path, split_name: str) -> None:
    if reb_df.empty:
        return
    weight_cols = [c for c in reb_df.columns if c.startswith("w_")]
    if not weight_cols:
        return
    plot_df = reb_df[["rebalance_date"] + weight_cols].copy()
    plot_df["rebalance_date"] = pd.to_datetime(plot_df["rebalance_date"])
    plot_df = plot_df.set_index("rebalance_date")
    plot_df.columns = [c.replace("w_", "") for c in plot_df.columns]

    plt.figure(figsize=(9, 4.8))
    plot_df.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title(f"Task 4.1 DNN-only weights by rebalance: {split_name}")
    plt.xlabel("Rebalance date")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_prediction_scatter(pred_all: pd.DataFrame, outpath: Path) -> None:
    if pred_all.empty:
        return
    plot_df = pred_all.dropna(subset=["y_true", "y_pred"]).copy()
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=int(CONFIG["seed"]))

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(plot_df["y_true"], plot_df["y_pred"], alpha=0.25)
    lim = float(max(np.abs(plot_df["y_true"]).max(), np.abs(plot_df["y_pred"]).max()))
    plt.plot([-lim, lim], [-lim, lim], linestyle="--")
    plt.title("Task 4.1 DNN-only: actual vs predicted next log return")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_future_prediction_plot(
    combined_pred_df: pd.DataFrame,
    outpath: Path,
    split_name: str,
    test_start: pd.Timestamp,
    context_days: int,
) -> None:
    if combined_pred_df.empty:
        return

    combined = combined_pred_df[["y_true", "y_pred", "dataset_part"]].rename(
        columns={"y_true": "actual", "y_pred": "predicted"}
    ).copy()

    start_boundary = pd.Timestamp(test_start)
    context_days = max(1, int(context_days))
    context_start = start_boundary - pd.Timedelta(days=context_days)
    combined = combined.loc[(slice(context_start, None), slice(None)), :].copy()
    if combined.empty:
        return

    tickers = list(combined.index.get_level_values("ticker").unique())
    n_tickers = len(tickers)
    if n_tickers == 0:
        return

    ncols = 2 if n_tickers > 1 else 1
    nrows = int(np.ceil(n_tickers / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.8 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for ax, ticker in zip(axes, tickers):
        ticker_df = combined.xs(ticker, level="ticker").sort_index()
        if ticker_df.empty:
            ax.set_visible(False)
            continue

        train_mask = ticker_df["dataset_part"].eq("train")
        test_mask = ticker_df["dataset_part"].eq("test")

        ax.plot(
            ticker_df.index,
            ticker_df["actual"],
            color="blue",
            label="Actual next log return",
        )

        if test_mask.any() and ticker_df.loc[test_mask, "predicted"].notna().any():
            ax.plot(
                ticker_df.index[train_mask],
                ticker_df.loc[train_mask, "predicted"],
                color="orange",
                linestyle="--",
                label="Cached train-period predictions",
            )
            ax.plot(
                ticker_df.index[test_mask],
                ticker_df.loc[test_mask, "predicted"],
                color="green",
                linestyle="-",
                label="Cached test-period predictions",
            )
        elif ticker_df["predicted"].notna().any():
            ax.plot(
                ticker_df.index,
                ticker_df["predicted"],
                color="green",
                linestyle="-",
                label="Cached test-period predictions",
            )

        ax.axvline(
            start_boundary,
            color="red",
            linestyle=":",
            linewidth=1.8,
            label="Test window starts",
        )
        ax.set_title(str(ticker))
        ax.set_xlabel("Date")
        ax.set_ylabel("Next log return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[n_tickers:]:
        ax.set_visible(False)

    fig.suptitle(f"Task 4.1 DNN-only cached prediction stream by ticker: {split_name}", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_metric_bar(results_df: pd.DataFrame, outpath: Path) -> None:
    if results_df.empty:
        return
    metric_cols = ["net_ann_return", "net_sharpe", "net_sortino", "net_max_drawdown"]
    plot_df = results_df[["split"] + metric_cols].copy().set_index("split")
    plt.figure(figsize=(10, 5.5))
    plot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Task 4.1 DNN-only split metrics")
    plt.xlabel("Split")
    plt.ylabel("Metric value")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_summary_tables(results_df: pd.DataFrame, diagnostics_df: pd.DataFrame, outdir: Path) -> None:
    tables_dir = outdir / "tables"
    _ensure_dir(tables_dir)

    if not results_df.empty:
        summary = results_df[[
            "net_ann_return", "net_ann_vol", "net_sharpe", "net_sortino",
            "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total"
        ]].agg(["mean", "std"]).T
        summary.columns = [f"portfolio_{c}" for c in summary.columns]
        summary.to_csv(tables_dir / "task4_1_portfolio_summary.csv")

        pivot = results_df[[
            "split", "net_ann_return", "net_ann_vol", "net_sharpe", "net_sortino",
            "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total"
        ]].copy()
        pivot.to_csv(tables_dir / "task4_1_split_portfolio_metrics.csv", index=False)

    if not diagnostics_df.empty:
        diag_summary = diagnostics_df[["mae", "rmse", "direction_acc", "pred_real_corr", "train_mae", "train_rmse", "n_epochs_trained"]].agg(["mean", "std"]).T
        diag_summary.columns = [f"diagnostic_{c}" for c in diag_summary.columns]
        diag_summary.to_csv(tables_dir / "task4_1_diagnostic_summary.csv")
        diagnostics_df.to_csv(tables_dir / "task4_1_split_diagnostics.csv", index=False)


# =========================
# Main runner
# =========================
def run_task4_1() -> None:
    set_seed(int(CONFIG["seed"]))

    task2_outdir = Path(CONFIG["task2_outdir"]).resolve()
    task4_outdir = Path(CONFIG["task4_outdir"]).resolve()
    _ensure_dir(task4_outdir)
    _ensure_dir(task4_outdir / "figures")
    _ensure_dir(task4_outdir / "equity_curves")
    _ensure_dir(task4_outdir / "rebalances")
    _ensure_dir(task4_outdir / "weights")
    _ensure_dir(task4_outdir / "metadata")

    with open(task4_outdir / "task4_1_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    meta = _load_walk_forward_meta(task2_outdir)
    meta_splits = meta.get("splits", [])
    if not meta_splits:
        raise RuntimeError("walk_forward_meta.json contains no splits. Run Task 2 first.")

    selected_splits = select_splits(
        meta_splits,
        pilot_mode=bool(CONFIG["pilot_mode"]),
        pilot_recent_splits=int(CONFIG["pilot_recent_splits"]),
    )

    target_col = str(CONFIG["target_col"])
    base_feature_cols = meta.get("feature_cols")

    results_rows: List[Dict] = []
    diagnostics_rows: List[Dict] = []
    pred_frames: List[pd.DataFrame] = []

    mode_label = "pilot" if bool(CONFIG["pilot_mode"]) else "full"
    print(f"Selected {len(selected_splits)} split(s) for Task 4.1 ({mode_label} mode).")

    for split in selected_splits:
        split_name = split["name"]
        train_path = Path(split["train_path"])
        test_path = Path(split["test_path"])
        if not train_path.is_absolute():
            train_path = (task2_outdir / train_path).resolve()
        if not test_path.is_absolute():
            test_path = (task2_outdir / test_path).resolve()

        print(f"\nRunning Task 4.1 on split: {split_name}")

        train_df = _read_panel(train_path)
        test_df = _read_panel(test_path)

        if base_feature_cols is None:
            base_feature_cols = [c for c in train_df.columns if c != target_col]
        feature_cols = [c for c in base_feature_cols if c not in set(CONFIG["exclude_feature_cols"])]

        ticker_categories = sorted(train_df.index.get_level_values("ticker").unique())

        train_pred_df, test_pred_df, hist_df, diagnostics, _split_meta = train_and_predict_split(
            split_name=split_name,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            ticker_categories=ticker_categories,
            outdir=task4_outdir,
        )
        combined_pred_df = pd.concat([train_pred_df, test_pred_df], axis=0).sort_index()
        pred_frames.append(combined_pred_df.assign(split=split_name))

        save_history_plot(hist_df, task4_outdir / "figures" / f"{split_name}_loss.png", split_name)

        test_start = pd.to_datetime(split["test_start"])
        test_end = pd.to_datetime(split["test_end"])
        save_future_prediction_plot(
            combined_pred_df=combined_pred_df,
            outpath=task4_outdir / "figures" / f"{split_name}_future_predictions.png",
            split_name=split_name,
            test_start=test_start,
            context_days=int(CONFIG["prediction_plot_context_days"]),
        )

        train_prices = _panel_to_prices(train_df)
        test_prices = _panel_to_prices(test_df)
        common = [t for t in train_prices.columns if t in test_prices.columns]
        train_prices = train_prices[common].dropna(how="any")
        test_prices = test_prices[common].dropna(how="any")
        full_prices = pd.concat([train_prices, test_prices], axis=0)
        full_prices = full_prices[~full_prices.index.duplicated(keep="first")].sort_index()

        gross, net, turnover_total, tc_total, n_reb, reb_df = simulate_dnn_signal_portfolio(
            full_prices=full_prices,
            pred_df=test_pred_df,
            test_start=test_start,
            test_end=test_end,
            rebalance_every_days=int(CONFIG["rebalance_every_days"]),
            transaction_cost_rate=float(CONFIG["transaction_cost_rate"]),
            max_weight=float(CONFIG["weight_bounds"][1]),
        )

        net_metrics = compute_metrics(net, int(CONFIG["frequency"]), float(CONFIG["risk_free_rate"]))

        eq_df = pd.DataFrame({
            "dnn_only_gross": (1.0 + gross).cumprod(),
            "dnn_only_net": (1.0 + net).cumprod(),
        })
        eq_df.to_csv(task4_outdir / "equity_curves" / f"{split_name}_equity.csv", index=True)
        reb_df.to_csv(task4_outdir / "rebalances" / f"{split_name}_dnn_only_rebalances.csv", index=False)

        if not reb_df.empty:
            last_row = reb_df.iloc[-1].to_dict()
            final_weights = {k.replace("w_", ""): float(v) for k, v in last_row.items() if k.startswith("w_")}
        else:
            final_weights = {}
        with open(task4_outdir / "weights" / f"{split_name}_dnn_only_final.json", "w", encoding="utf-8") as f:
            json.dump(final_weights, f, indent=2)

        save_equity_plot(eq_df, task4_outdir / "figures" / f"{split_name}_equity.png", split_name)
        save_weights_plot(reb_df, task4_outdir / "figures" / f"{split_name}_weights.png", split_name)

        results_rows.append({
            "split": split_name,
            "model": "dnn_only",
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "n_reb": int(n_reb),
            "turnover_total": float(turnover_total),
            "tc_total": float(tc_total),
            "net_ann_return": float(net_metrics["ann_return"]),
            "net_ann_vol": float(net_metrics["ann_vol"]),
            "net_sharpe": float(net_metrics["sharpe"]),
            "net_sortino": float(net_metrics["sortino"]),
            "net_cumulative_return": float(net_metrics["cumulative_return"]),
            "net_max_drawdown": float(net_metrics["max_drawdown"]),
        })

        diagnostics_rows.append({
            "split": split_name,
            "model": "dnn_only",
            **diagnostics,
        })

        print(
            f"  DNN-only: ann_return={net_metrics['ann_return']:.4f}, "
            f"sharpe={net_metrics['sharpe']:.3f}, sortino={net_metrics['sortino']:.3f}, "
            f"mdd={net_metrics['max_drawdown']:.3f}, mae={diagnostics['mae']:.6f}, "
            f"rmse={diagnostics['rmse']:.6f}, dir_acc={diagnostics['direction_acc']:.3f}"
        )

    results_df = pd.DataFrame(results_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    pred_all = pd.concat(pred_frames).sort_index() if pred_frames else pd.DataFrame()

    results_df.to_csv(task4_outdir / "task4_1_split_results.csv", index=False)
    diagnostics_df.to_csv(task4_outdir / "task4_1_diagnostics_by_split.csv", index=False)
    if not pred_all.empty:
        pred_all.to_csv(task4_outdir / "task4_1_predictions_aggregate.csv")

    save_summary_tables(results_df, diagnostics_df, task4_outdir)
    save_prediction_scatter(pred_all, task4_outdir / "figures" / "task4_1_prediction_scatter.png")
    save_metric_bar(results_df, task4_outdir / "figures" / "task4_1_split_metrics.png")

    if not results_df.empty:
        summary = (
            results_df
            .groupby("model")[[
                "net_ann_return", "net_ann_vol", "net_sharpe", "net_sortino",
                "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total"
            ]]
            .agg(["mean", "std"])
        )
        summary.to_csv(task4_outdir / "task4_1_summary_by_model.csv")

    print("\nDone.")
    print(f"Saved split results:  {task4_outdir / 'task4_1_split_results.csv'}")
    print(f"Saved diagnostics:    {task4_outdir / 'task4_1_diagnostics_by_split.csv'}")
    print(f"Saved predictions:    {task4_outdir / 'task4_1_predictions_aggregate.csv'}")
    print(f"Saved figures in:     {task4_outdir / 'figures'}")
    print(f"Saved metadata in:    {task4_outdir / 'metadata'}")
    print(f"Saved rebalances in:  {task4_outdir / 'rebalances'}")
    print(f"Saved weights in:     {task4_outdir / 'weights'}")


if __name__ == "__main__":
    run_task4_1()
