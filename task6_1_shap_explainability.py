"""
Task 6.1: SHAP explainability for the Task 4.1 DNN predictor

Purpose
- Load the saved Task 4.1 DNN predictor for selected walk-forward split(s)
- Apply SHAP to explain the DNN's next-period log return forecasts
- Save global and local explanation artefacts
- Save plain-English summaries that can be surfaced directly in the demo UI

Notes
- This explains the supervised DNN module, not the PPO policy directly.
- For research runs, the default is to explain all saved Task 4.1 models.
- This script explains the exact fitted DNN saved by Task 4.1. It does not retrain.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except Exception as e:
    raise SystemExit(
        "SHAP is required for Task 6.1.\n"
        "Install with: pip install shap\n"
        f"Original error: {e}"
    )

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
except Exception as e:
    raise SystemExit(
        "TensorFlow is required for Task 6.1.\n"
        "Install with: pip install tensorflow\n"
        f"Original error: {e}"
    )

import task4_1_dnn_only as t41

warnings.filterwarnings(
    "ignore",
    message=r".*The structure of `inputs` doesn\'t match the expected structure.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*The NumPy global RNG was seeded by calling `np.random.seed`.*",
    category=FutureWarning,
)
try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass


CONFIG = {
    # Inputs
    "task2_outdir": "data_prepared_task2",
    "task4_outdir": "results_task4_1_dnn_only",

    # Outputs
    "task6_1_outdir": "results_task6_1_shap_explainability",

    # Split scope
    "selection_mode": "all_saved_models",   # "all_saved_models", "selected", "recent"
    "selected_splits": [],                  # used when selection_mode == "selected"
    "recent_splits": 2,                     # used when selection_mode == "recent"

    # Rebuild / load model
    "require_saved_models": True,

    # SHAP sampling controls
    "background_sample_size": 80,
    "explain_sample_size": 120,
    "local_case_count": 4,
    "max_display_features": 15,
    "dependence_top_k": 3,
    "random_state": 42,

    # Plot / summary settings
    "save_dependence_plots": True,
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sample_rows(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=n, replace=False))
    return x[idx].copy()


def _normalise_feature_label(name: str) -> str:
    mapping = {
        "log_return_1": "1-day log return",
        "simple_return_1": "1-day simple return",
        "ret_mean_5": "5-day average return",
        "ret_mean_10": "10-day average return",
        "ret_mean_20": "20-day average return",
        "ret_mean_60": "60-day average return",
        "ret_std_5": "5-day return volatility",
        "ret_std_10": "10-day return volatility",
        "ret_std_20": "20-day return volatility",
        "ret_std_60": "60-day return volatility",
        "mom_5": "5-day momentum",
        "mom_10": "10-day momentum",
        "mom_20": "20-day momentum",
        "mom_60": "60-day momentum",
        "rsi_14": "14-day RSI",
        "macd_12_26": "MACD (12, 26)",
        "macd_signal_9": "MACD signal (9)",
        "macd_hist": "MACD histogram",
        "volume": "Trading volume",
        "volume_chg": "Volume change",
        "volume_roll_mean_20": "20-day average volume",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("ticker_"):
        return f"Asset indicator: {name.replace('ticker_', '')}"
    return name.replace("_", " ")


def _descriptor_from_value(v: float) -> str:
    av = abs(float(v))
    if av >= 0.05:
        return "strongly"
    if av >= 0.01:
        return "moderately"
    return "slightly"


def _prediction_direction_text(y_pred: float) -> str:
    if y_pred > 1e-4:
        return "positive"
    if y_pred < -1e-4:
        return "negative"
    return "near-neutral"


def _rank_case_positions(test_pred_df: pd.DataFrame, n_cases: int) -> List[int]:
    if test_pred_df.empty:
        return []

    tmp = test_pred_df.copy()
    tmp["abs_pred"] = tmp["y_pred"].abs()
    positions: List[int] = []

    # Largest positive prediction
    if (tmp["y_pred"] > 0).any():
        positions.append(int(tmp["y_pred"].idxmax()))
    # Most negative prediction
    if (tmp["y_pred"] < 0).any():
        positions.append(int(tmp["y_pred"].idxmin()))
    # Largest absolute error
    positions.append(int(tmp["abs_error"].idxmax()))
    # Largest absolute prediction
    positions.append(int(tmp["abs_pred"].idxmax()))

    ordered = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            ordered.append(pos)
            seen.add(pos)

    if len(ordered) < n_cases:
        for pos in tmp["abs_pred"].sort_values(ascending=False).index.tolist():
            pos = int(pos)
            if pos not in seen:
                ordered.append(pos)
                seen.add(pos)
            if len(ordered) >= n_cases:
                break

    return ordered[:n_cases]


def _load_task4_metadata(task4_outdir: Path, split_name: str) -> Dict:
    meta_path = task4_outdir / "metadata" / f"{split_name}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Could not find {meta_path}. Run Task 4.1 first so split metadata exists."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_task2_meta(task2_outdir: Path) -> Dict:
    return t41._load_walk_forward_meta(task2_outdir)


def _select_splits(
    meta_splits: List[Dict],
    task4_outdir: Path,
    selection_mode: str,
    selected_names: List[str],
    recent_splits: int,
) -> List[Dict]:
    by_name = {str(s.get("name")): s for s in meta_splits}
    mode = str(selection_mode).strip().lower()

    if mode == "selected":
        wanted = set(selected_names)
        selected = [s for s in meta_splits if s.get("name") in wanted]
        if not selected:
            raise RuntimeError("None of the requested selected_splits were found in walk_forward_meta.json.")
        return selected

    if mode == "recent":
        n = max(1, int(recent_splits))
        return meta_splits[-n:]

    if mode == "all_saved_models":
        models_dir = task4_outdir / "models"
        if not models_dir.exists():
            raise RuntimeError(
                f"Could not find saved-model directory: {models_dir}. "
                "Run Task 4.1 with save_models=True first."
            )
        saved_names = sorted(p.stem for p in models_dir.glob("*.keras"))
        selected = [by_name[name] for name in saved_names if name in by_name]
        if not selected:
            raise RuntimeError(
                "No saved Task 4.1 models matched the Task 2 walk-forward split metadata."
            )
        return selected

    raise RuntimeError(
        f"Unknown selection_mode={selection_mode!r}. Use 'all_saved_models', 'selected', or 'recent'."
    )


def _build_dnn_from_cfg(input_dim: int, cfg: Dict) -> tf.keras.Model:
    hidden_units = [int(x) for x in cfg["hidden_units"]]
    dropout_rate = float(cfg["dropout_rate"])
    l2_reg = float(cfg["l2_reg"])
    lr = float(cfg["learning_rate"])

    model = models.Sequential(name="task6_shap_dnn")
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


def _load_saved_model_and_data(
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_meta: Dict,
    task4_outdir: Path,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_cols = list(split_meta["feature_cols"])
    target_col = str(split_meta["target_col"])
    ticker_categories = list(split_meta["ticker_categories"])
    use_ticker_one_hot = bool(split_meta["use_ticker_one_hot"])

    model_path = task4_outdir / "models" / f"{split_name}.keras"
    if bool(CONFIG["require_saved_models"]) and not model_path.exists():
        raise FileNotFoundError(
            f"Saved Task 4.1 model not found for {split_name}: {model_path}. "
            "Rerun task4_1_dnn_only.py with save_models=True before running Task 6.1."
        )

    model = tf.keras.models.load_model(model_path)

    x_train_full, y_train_full, feature_names = t41.build_xy(
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
    return model, x_train_full, y_train_full, x_test, y_test, feature_names

    train_part, val_part = t41.time_based_train_val_split(train_df, float(model_cfg["validation_ratio"]))

    x_train_fit, y_train_fit, feature_names = t41.build_xy(
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

    model = _build_dnn_from_cfg(input_dim=x_train_fit.shape[1], cfg=model_cfg)
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(model_cfg["patience"]),
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train_fit,
        y_train_fit,
        validation_data=(x_val, y_val),
        epochs=int(model_cfg["epochs"]),
        batch_size=int(model_cfg["batch_size"]),
        verbose=0,
        callbacks=cb,
        shuffle=False,
    )
    hist_df = pd.DataFrame(history.history)
    return model, x_train_full, y_train_full, x_test, y_test, feature_names, hist_df


def _compute_shap_values(
    model: tf.keras.Model,
    x_background: np.ndarray,
    x_explain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str]:
    # Use the model's expected prediction on background data as a stable base value.
    base_value = float(model.predict(x_background, verbose=0).reshape(-1).mean())

    errors: List[str] = []

    # 1) GradientExplainer first
    try:
        explainer = shap.GradientExplainer(model, x_background)
        raw = explainer.shap_values(x_explain)
        if isinstance(raw, list):
            raw = raw[0]
        values = np.asarray(raw)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[:, :, 0]
        if values.shape[0] == x_explain.shape[0]:
            base_values = np.full(x_explain.shape[0], base_value, dtype=float)
            return values.astype(float), base_values, "GradientExplainer"
    except Exception as e:
        errors.append(f"GradientExplainer failed: {e}")

    # 2) DeepExplainer fallback
    try:
        explainer = shap.DeepExplainer(model, x_background)
        raw = explainer.shap_values(x_explain)
        if isinstance(raw, list):
            raw = raw[0]
        values = np.asarray(raw)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[:, :, 0]
        if values.shape[0] == x_explain.shape[0]:
            base_values = np.full(x_explain.shape[0], base_value, dtype=float)
            return values.astype(float), base_values, "DeepExplainer"
    except Exception as e:
        errors.append(f"DeepExplainer failed: {e}")

    # 3) KernelExplainer fallback (slower, but model-agnostic)
    try:
        predict_fn = lambda x: model.predict(np.asarray(x, dtype=np.float32), verbose=0).reshape(-1)
        explainer = shap.KernelExplainer(predict_fn, x_background)
        raw = explainer.shap_values(x_explain, nsamples="auto")
        if isinstance(raw, list):
            raw = raw[0]
        values = np.asarray(raw)
        base_values = np.full(x_explain.shape[0], base_value, dtype=float)
        return values.astype(float), base_values, "KernelExplainer"
    except Exception as e:
        errors.append(f"KernelExplainer failed: {e}")

    raise RuntimeError("All SHAP explainer attempts failed:\n" + "\n".join(errors))


def _save_summary_plot(values: np.ndarray, x_df: pd.DataFrame, outpath: Path, max_display: int) -> None:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        values,
        x_df,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def _save_bar_plot(values: np.ndarray, x_df: pd.DataFrame, outpath: Path, max_display: int) -> None:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        values,
        x_df,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def _save_dependence_plot(values: np.ndarray, x_df: pd.DataFrame, feature_name: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feature_name,
        values,
        x_df,
        interaction_index=None,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def _save_local_bar_plot(row_values: np.ndarray, row_data: np.ndarray, feature_names: List[str], outpath: Path, title: str) -> None:
    order = np.argsort(np.abs(row_values))[::-1][:10]
    vals = row_values[order]
    labels = [feature_names[i] for i in order]
    colors = ["tab:green" if v > 0 else "tab:red" for v in vals]

    plt.figure(figsize=(9, 5.5))
    y = np.arange(len(vals))
    plt.barh(y, vals, color=colors)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("SHAP contribution to predicted next log return")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def _make_global_summary_text(global_df: pd.DataFrame) -> Dict[str, object]:
    top = global_df.head(5)
    names = top["feature_label"].tolist()
    text = (
        "Across the explained rows, the DNN was mostly influenced by "
        + ", ".join(names[:-1])
        + (", and " + names[-1] if len(names) > 1 else names[0])
        + "."
    ) if names else "No SHAP explanations were generated."
    return {
        "top_features": names,
        "summary_text": text,
    }


def _make_local_summary(
    split_name: str,
    row_index: Tuple[pd.Timestamp, str],
    y_true: float,
    y_pred: float,
    base_value: float,
    row_values: np.ndarray,
    feature_names: List[str],
    human_names: List[str],
) -> Dict[str, object]:
    date_val, ticker_val = row_index

    pos_idx = np.argsort(row_values)[::-1]
    pos_idx = [i for i in pos_idx if row_values[i] > 0][:3]
    neg_idx = np.argsort(row_values)
    neg_idx = [i for i in neg_idx if row_values[i] < 0][:3]

    pos_items = [
        {
            "feature": feature_names[i],
            "feature_label": human_names[i],
            "contribution": float(row_values[i]),
        }
        for i in pos_idx
    ]
    neg_items = [
        {
            "feature": feature_names[i],
            "feature_label": human_names[i],
            "contribution": float(row_values[i]),
        }
        for i in neg_idx
    ]

    direction = _prediction_direction_text(y_pred)
    delta = float(y_pred - base_value)
    delta_desc = _descriptor_from_value(delta)

    pos_text = ", ".join(item["feature_label"] for item in pos_items) if pos_items else "no strong upward drivers"
    neg_text = ", ".join(item["feature_label"] for item in neg_items) if neg_items else "no strong downward drivers"

    summary_text = (
        f"For {ticker_val} on {pd.to_datetime(date_val).date()}, the DNN predicted a {direction} next-period log return. "
        f"Relative to the baseline prediction, the forecast moved {delta_desc} {'up' if delta >= 0 else 'down'}. "
        f"The main upward influences were {pos_text}. "
        f"The main downward influences were {neg_text}."
    )

    return {
        "split": split_name,
        "date": str(pd.to_datetime(date_val).date()),
        "ticker": str(ticker_val),
        "y_true": float(y_true),
        "y_pred": float(y_pred),
        "base_value": float(base_value),
        "prediction_direction": direction,
        "summary_text": summary_text,
        "positive_drivers": pos_items,
        "negative_drivers": neg_items,
    }


def main() -> None:
    task2_outdir = Path(CONFIG["task2_outdir"])
    task4_outdir = Path(CONFIG["task4_outdir"])
    task6_1_outdir = Path(CONFIG["task6_1_outdir"])

    _ensure_dir(task6_1_outdir)
    _ensure_dir(task6_1_outdir / "tables")
    _ensure_dir(task6_1_outdir / "figures")
    _ensure_dir(task6_1_outdir / "local_cases")
    _ensure_dir(task6_1_outdir / "plain_english")

    with open(task6_1_outdir / "task6_1_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    meta = _load_task2_meta(task2_outdir)
    meta_splits = meta.get("splits", [])
    if not meta_splits:
        raise RuntimeError("walk_forward_meta.json contains no splits. Run Task 2 first.")

    selected_splits = _select_splits(
        meta_splits=meta_splits,
        task4_outdir=task4_outdir,
        selection_mode=str(CONFIG["selection_mode"]),
        selected_names=list(CONFIG["selected_splits"]),
        recent_splits=int(CONFIG["recent_splits"]),
    )

    summary_rows: List[Dict] = []
    local_case_rows: List[Dict] = []
    global_texts: List[Dict] = []

    print(
        f"Selected {len(selected_splits)} split(s) for Task 6.1 SHAP explainability "
        f"(mode={CONFIG['selection_mode']})."
    )

    for split in selected_splits:
        split_name = split["name"]
        print(f"\nRunning Task 6.1 on split: {split_name}")

        split_meta = _load_task4_metadata(task4_outdir, split_name)

        train_path = Path(split["train_path"])
        test_path = Path(split["test_path"])
        if not train_path.is_absolute():
            train_path = (task2_outdir / train_path).resolve()
        if not test_path.is_absolute():
            test_path = (task2_outdir / test_path).resolve()

        train_df = t41._read_panel(train_path)
        test_df = t41._read_panel(test_path)

        model, x_train_full, y_train_full, x_test, y_test, feature_names = _load_saved_model_and_data(
            split_name=split_name,
            train_df=train_df,
            test_df=test_df,
            split_meta=split_meta,
            task4_outdir=task4_outdir,
        )

        human_feature_names = [_normalise_feature_label(c) for c in feature_names]
        x_test_df = pd.DataFrame(x_test, columns=human_feature_names)

        background = _sample_rows(
            x_train_full,
            n=int(CONFIG["background_sample_size"]),
            seed=int(CONFIG["random_state"]),
        )

        y_pred_test = model.predict(x_test, verbose=0).reshape(-1)
        test_pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred_test,
        }, index=test_df.index)
        test_pred_df["abs_error"] = (test_pred_df["y_true"] - test_pred_df["y_pred"]).abs()
        test_pred_df = test_pred_df.reset_index(drop=False)
        test_pred_df.index = np.arange(len(test_pred_df))

        case_positions = _rank_case_positions(test_pred_df, int(CONFIG["local_case_count"]))

        explain_n = min(int(CONFIG["explain_sample_size"]), len(x_test))
        explain_idx = np.arange(len(x_test))
        if len(x_test) > explain_n:
            rng = np.random.default_rng(int(CONFIG["random_state"]))
            sampled = np.sort(rng.choice(len(x_test), size=explain_n, replace=False))
            required = np.array(sorted(set(int(p) for p in case_positions)), dtype=int)
            explain_idx = np.array(sorted(set(sampled.tolist()).union(required.tolist())), dtype=int)

        x_explain = x_test[explain_idx].copy()
        x_explain_df = pd.DataFrame(x_explain, columns=human_feature_names)
        explain_index = test_df.index[explain_idx]

        shap_values, base_values, explainer_name = _compute_shap_values(model, background, x_explain)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        global_df = pd.DataFrame({
            "feature": feature_names,
            "feature_label": human_feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        global_df.to_csv(task6_1_outdir / "tables" / f"{split_name}_global_feature_importance.csv", index=False)

        _save_summary_plot(
            shap_values,
            x_explain_df,
            outpath=task6_1_outdir / "figures" / f"{split_name}_shap_summary.png",
            max_display=int(CONFIG["max_display_features"]),
        )
        _save_bar_plot(
            shap_values,
            x_explain_df,
            outpath=task6_1_outdir / "figures" / f"{split_name}_shap_bar.png",
            max_display=int(CONFIG["max_display_features"]),
        )

        if bool(CONFIG["save_dependence_plots"]):
            for feat in global_df["feature_label"].head(int(CONFIG["dependence_top_k"])).tolist():
                safe_name = feat.lower().replace(" ", "_").replace("/", "_")
                _save_dependence_plot(
                    shap_values,
                    x_explain_df,
                    feature_name=feat,
                    outpath=task6_1_outdir / "figures" / f"{split_name}_dependence_{safe_name}.png",
                )

        local_json_rows: List[Dict] = []

        explain_pos_lookup = {int(pos): i for i, pos in enumerate(explain_idx.tolist())}
        case_positions = [int(p) for p in case_positions if int(p) in explain_pos_lookup]

        for rank, pos in enumerate(case_positions, start=1):
            explain_row_idx = explain_pos_lookup[int(pos)]
            index_tuple = test_df.index[int(pos)]
            row_vals = shap_values[explain_row_idx]
            row_data = x_test[int(pos)]
            y_true = float(y_test[int(pos)])
            y_pred = float(y_pred_test[int(pos)])
            base_val = float(base_values[explain_row_idx])

            local_summary = _make_local_summary(
                split_name=split_name,
                row_index=index_tuple,
                y_true=y_true,
                y_pred=y_pred,
                base_value=base_val,
                row_values=row_vals,
                feature_names=feature_names,
                human_names=human_feature_names,
            )
            local_summary["case_rank"] = rank
            local_json_rows.append(local_summary)

            local_case_rows.append({
                "split": split_name,
                "case_rank": rank,
                "date": local_summary["date"],
                "ticker": local_summary["ticker"],
                "y_true": y_true,
                "y_pred": y_pred,
                "base_value": base_val,
                "summary_text": local_summary["summary_text"],
            })

            _save_local_bar_plot(
                row_values=row_vals,
                row_data=row_data,
                feature_names=human_feature_names,
                outpath=task6_1_outdir / "local_cases" / f"{split_name}_case_{rank}_local_bar.png",
                title=f"Task 6.1 local SHAP drivers: {split_name} case {rank}",
            )

        global_text = _make_global_summary_text(global_df)
        global_text_record = {
            "split": split_name,
            "explainer": explainer_name,
            **global_text,
        }
        global_texts.append(global_text_record)

        with open(task6_1_outdir / "plain_english" / f"{split_name}_global_summary.json", "w", encoding="utf-8") as f:
            json.dump(global_text_record, f, indent=2)
        with open(task6_1_outdir / "plain_english" / f"{split_name}_local_explanations.json", "w", encoding="utf-8") as f:
            json.dump(local_json_rows, f, indent=2)

        summary_rows.append({
            "split": split_name,
            "explainer": explainer_name,
            "n_background_rows": int(len(background)),
            "n_explained_rows": int(len(x_explain)),
            "n_train_rows": int(len(x_train_full)),
            "n_test_rows": int(len(x_test)),
            "n_features": int(len(feature_names)),
            "top_feature_1": global_df.iloc[0]["feature_label"] if len(global_df) > 0 else None,
            "top_feature_2": global_df.iloc[1]["feature_label"] if len(global_df) > 1 else None,
            "top_feature_3": global_df.iloc[2]["feature_label"] if len(global_df) > 2 else None,
        })

        print(
            f"  SHAP done with {explainer_name}: explained_rows={len(x_explain)}, "
            f"top_feature={global_df.iloc[0]['feature_label'] if len(global_df) > 0 else 'n/a'}"
        )
        print(f"  Plain-English SHAP summary for {split_name}:")
        print(f"    Global: {global_text_record['summary_text']}")
        for row in local_json_rows[:3]:
            print(f"    Case {row['case_rank']}: {row['summary_text']}")

    summary_df = pd.DataFrame(summary_rows)
    local_cases_df = pd.DataFrame(local_case_rows)
    global_texts_df = pd.DataFrame(global_texts)

    summary_df.to_csv(task6_1_outdir / "task6_1_shap_summary.csv", index=False)
    local_cases_df.to_csv(task6_1_outdir / "task6_1_local_case_summaries.csv", index=False)
    global_texts_df.to_csv(task6_1_outdir / "task6_1_global_plain_english.csv", index=False)

    print("\nDone.")
    print(f"Saved SHAP summary:   {task6_1_outdir / 'task6_1_shap_summary.csv'}")
    print(f"Saved local summaries:{task6_1_outdir / 'task6_1_local_case_summaries.csv'}")
    print(f"Saved tables in:      {task6_1_outdir / 'tables'}")
    print(f"Saved figures in:     {task6_1_outdir / 'figures'}")
    print(f"Saved demo text in:   {task6_1_outdir / 'plain_english'}")


if __name__ == "__main__":
    main()
