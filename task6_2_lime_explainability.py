"""
Task 6.2: LIME explainability case study for a specific Task 4.1 DNN split.

Purpose
- Load the exact saved Task 4.1 DNN model for one selected split
- Apply LIME to selected out-of-sample test cases for local explanations
- Save local explanation plots/tables and plain-English summaries
- Provide a small case-study companion to Task 6.1 SHAP outputs

Notes
- This is intentionally local and split-focused. LIME is used here as a supporting method
  for selected cases, not as the main global explainability layer.
- This script targets split_013_2018-04-02_2020-06-30.
- Requires Task 4.1 saved models and split metadata.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reduce noisy TensorFlow startup logs as much as reasonably possible.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception as e:
    raise SystemExit(
        "LIME is required for Task 6.2.\n"
        "Install with: pip install lime\n"
        f"Original error: {e}"
    )

try:
    import tensorflow as tf
except Exception as e:
    raise SystemExit(
        "TensorFlow is required for Task 6.2.\n"
        "Install with: pip install tensorflow\n"
        f"Original error: {e}"
    )

import task4_1_dnn_only as t41


CONFIG = {
    "task2_outdir": "data_prepared_task2",
    "task4_outdir": "results_task4_1_dnn_only",
    "task6_2_outdir": "results_task6_2_lime_explainability",
    "selected_split": "split_013_2018-04-02_2020-06-30",      # Change to any split
    "random_state": 42,
    "num_features": 10,
    "num_samples": 4000,
    "local_case_count": 3,
    "discretize_continuous": True,
}


FEATURE_LABELS = {
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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _normalise_feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("ticker_"):
        return f"Asset indicator: {name.replace('ticker_', '')}"
    return name.replace("_", " ")


def _prediction_direction_text(y_pred: float) -> str:
    if y_pred > 1e-4:
        return "positive"
    if y_pred < -1e-4:
        return "negative"
    return "near-neutral"


def _descriptor_from_value(v: float) -> str:
    av = abs(float(v))
    if av >= 0.05:
        return "strongly"
    if av >= 0.01:
        return "moderately"
    return "slightly"


def _clean_lime_feature_text(text: str) -> str:
    # LIME returns binned texts like "ret_mean_60 > 0.02". Replace feature tokens with labels.
    out = text
    for raw, nice in sorted(FEATURE_LABELS.items(), key=lambda x: -len(x[0])):
        out = out.replace(raw, nice)
    out = out.replace("ticker_", "Asset indicator: ")
    return out


def _load_task4_metadata(task4_outdir: Path, split_name: str) -> Dict:
    meta_path = task4_outdir / "metadata" / f"{split_name}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Could not find {meta_path}. Run Task 4.1 with model saving first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rank_case_positions(test_pred_df: pd.DataFrame, n_cases: int) -> List[int]:
    if test_pred_df.empty:
        return []

    tmp = test_pred_df.copy()
    tmp["abs_pred"] = tmp["y_pred"].abs()
    positions: List[int] = []

    if (tmp["y_pred"] > 0).any():
        positions.append(int(tmp["y_pred"].idxmax()))
    if (tmp["y_pred"] < 0).any():
        positions.append(int(tmp["y_pred"].idxmin()))
    positions.append(int(tmp["abs_error"].idxmax()))

    ordered: List[int] = []
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


def _build_predict_fn(model: tf.keras.Model):
    def _predict(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        preds = model.predict(x, verbose=0).reshape(-1)
        return preds
    return _predict



def _coerce_scalar_explanation_value(value, default: float = float("nan")) -> float:
    """Robustly convert LIME scalar-like outputs to float.

    LIME may return intercept/local_pred/score as floats, numpy arrays, lists/tuples, or dicts.
    For dicts, use the first numeric value in sorted-key order.
    """
    try:
        if value is None:
            return float(default)
        if isinstance(value, dict):
            if not value:
                return float(default)
            for key in sorted(value.keys(), key=lambda x: str(x)):
                try:
                    return float(np.ravel(value[key])[0])
                except Exception:
                    continue
            return float(default)
        if isinstance(value, (list, tuple)):
            if not value:
                return float(default)
            return float(np.ravel(value[0])[0])
        arr = np.ravel(value)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)

def _build_case_text(date_val: str, ticker_val: str, y_pred: float, intercept: float,
                     top_positive: List[str], top_negative: List[str]) -> str:
    direction = _prediction_direction_text(y_pred)
    shift_desc = _descriptor_from_value(y_pred - intercept)
    positive_txt = ", ".join(top_positive) if top_positive else "none"
    negative_txt = ", ".join(top_negative) if top_negative else "none"
    return (
        f"For {ticker_val} on {date_val}, the DNN predicted a {direction} next-period log return. "
        f"Relative to the local surrogate baseline, the forecast moved {shift_desc} "
        f"{'up' if y_pred >= intercept else 'down'}. "
        f"The main upward influences were {positive_txt}. "
        f"The main downward influences were {negative_txt}."
    )


def _aggregate_case_importance(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["feature", "mean_abs_weight", "count_appears"])
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("feature", as_index=False)
          .agg(mean_abs_weight=("abs_weight", "mean"), count_appears=("feature", "size"))
          .sort_values(["mean_abs_weight", "count_appears"], ascending=[False, False])
    )
    return agg


def save_case_plot(exp, out_path: Path, title: str) -> None:
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_case_html(exp, out_path: Path) -> None:
    exp.save_to_file(str(out_path))


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"The structure of `inputs` doesn't match the expected structure.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The NumPy global RNG was seeded by calling `np.random.seed`.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"TensorFlow GPU support is not available on native Windows.*",
        category=UserWarning,
    )

    seed = int(CONFIG["random_state"])
    _set_seed(seed)

    task2_outdir = Path(CONFIG["task2_outdir"])
    task4_outdir = Path(CONFIG["task4_outdir"])
    outdir = Path(CONFIG["task6_2_outdir"])
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    html_dir = outdir / "html"
    plain_dir = outdir / "plain_english"
    _ensure_dir(outdir)
    _ensure_dir(tables_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(html_dir)
    _ensure_dir(plain_dir)

    split_name = str(CONFIG["selected_split"])

    task2_meta = t41._load_walk_forward_meta(task2_outdir)
    split = next((s for s in task2_meta.get("splits", []) if s.get("name") == split_name), None)
    if split is None:
        raise RuntimeError(f"Split {split_name} not found in Task 2 walk_forward_meta.json")

    print(f"Running Task 6.2 LIME on split: {split_name}")

    split_meta = _load_task4_metadata(task4_outdir, split_name)
    feature_cols = list(split_meta["feature_cols"])
    target_col = str(split_meta["target_col"])
    ticker_categories = list(split_meta["ticker_categories"])
    use_ticker_one_hot = bool(split_meta["use_ticker_one_hot"])

    train_path = Path(split["train_path"])
    test_path = Path(split["test_path"])
    if not train_path.is_absolute():
        train_path = (task2_outdir / train_path).resolve()
    if not test_path.is_absolute():
        test_path = (task2_outdir / test_path).resolve()

    train_df = t41._read_panel(train_path)
    test_df = t41._read_panel(test_path)

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

    model_path = task4_outdir / "models" / f"{split_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Re-run Task 4.1 with model saving enabled."
        )
    model = tf.keras.models.load_model(model_path)
    predict_fn = _build_predict_fn(model)

    pred_test = predict_fn(x_test)
    test_pred_df = test_df[[target_col]].copy().rename(columns={target_col: "y_true"}).reset_index()
    test_pred_df["y_pred"] = pred_test.astype(np.float32)
    test_pred_df["abs_error"] = np.abs(test_pred_df["y_true"] - test_pred_df["y_pred"]).astype(np.float32)

    case_positions = _rank_case_positions(test_pred_df, int(CONFIG["local_case_count"]))
    if not case_positions:
        raise RuntimeError("No local cases found for LIME explanation.")

    explainer = LimeTabularExplainer(
        training_data=x_train_full.astype(np.float64),
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=bool(CONFIG["discretize_continuous"]),
        random_state=seed,
    )

    all_case_rows: List[Dict] = []
    case_summaries: List[Dict] = []
    aggregate_weights: List[Dict] = []

    for i, pos in enumerate(case_positions, start=1):
        row = test_pred_df.iloc[int(pos)]
        x_row = x_test[int(pos)]
        explanation = explainer.explain_instance(
            data_row=x_row.astype(np.float64),
            predict_fn=predict_fn,
            num_features=int(CONFIG["num_features"]),
            num_samples=int(CONFIG["num_samples"]),
        )

        # LIME regression explanations are local; as_list() is the simplest stable access path.
        pairs = explanation.as_list()
        case_items: List[Dict] = []
        top_positive: List[str] = []
        top_negative: List[str] = []

        for feat_text, weight in pairs:
            cleaned = _clean_lime_feature_text(feat_text)
            case_items.append({
                "split_name": split_name,
                "case_id": i,
                "date": str(pd.to_datetime(row["date"]).date()),
                "ticker": row["ticker"],
                "feature": cleaned,
                "weight": float(weight),
                "abs_weight": abs(float(weight)),
            })
            aggregate_weights.append(case_items[-1])
            if weight > 0 and len(top_positive) < 3:
                top_positive.append(cleaned)
            elif weight < 0 and len(top_negative) < 3:
                top_negative.append(cleaned)

        for item in case_items:
            item.update({
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
            })
        all_case_rows.extend(case_items)

        intercept_val = _coerce_scalar_explanation_value(getattr(explanation, "intercept", np.nan), default=float(row["y_pred"]))
        local_pred = _coerce_scalar_explanation_value(getattr(explanation, "local_pred", np.nan), default=float(row["y_pred"]))
        if not np.isfinite(local_pred):
            local_pred = float(row["y_pred"])
        fidelity = _coerce_scalar_explanation_value(getattr(explanation, "score", np.nan), default=np.nan)
        case_text = _build_case_text(
            date_val=str(pd.to_datetime(row["date"]).date()),
            ticker_val=str(row["ticker"]),
            y_pred=float(local_pred),
            intercept=intercept_val,
            top_positive=top_positive,
            top_negative=top_negative,
        )

        case_summaries.append({
            "split_name": split_name,
            "case_id": i,
            "date": str(pd.to_datetime(row["date"]).date()),
            "ticker": str(row["ticker"]),
            "y_true": float(row["y_true"]),
            "y_pred": float(row["y_pred"]),
            "local_pred": local_pred,
            "intercept": intercept_val,
            "fidelity_score": fidelity,
            "plain_english": case_text,
            "top_positive_features": "; ".join(top_positive),
            "top_negative_features": "; ".join(top_negative),
        })

        safe_date = str(pd.to_datetime(row["date"]).date())
        png_path = figures_dir / f"{split_name}_case_{i}_{row['ticker']}_{safe_date}_lime.png"
        html_path = html_dir / f"{split_name}_case_{i}_{row['ticker']}_{safe_date}_lime.html"
        title = f"Task 6.2 LIME local explanation: {row['ticker']} on {safe_date}"
        save_case_plot(explanation, png_path, title)
        save_case_html(explanation, html_path)

    local_df = pd.DataFrame(case_summaries)
    local_df.to_csv(tables_dir / f"{split_name}_lime_local_case_summaries.csv", index=False)

    if all_case_rows:
        pd.DataFrame(all_case_rows).to_csv(tables_dir / f"{split_name}_lime_case_feature_weights.csv", index=False)

    aggregate_df = _aggregate_case_importance(aggregate_weights)
    aggregate_df.to_csv(tables_dir / f"{split_name}_lime_aggregate_importance.csv", index=False)

    global_text = ""
    if not aggregate_df.empty:
        top_feats = aggregate_df["feature"].head(5).tolist()
        global_text = (
            f"Across the selected LIME cases, the DNN was most locally influenced by "
            f"{', '.join(top_feats[:-1])}, and {top_feats[-1]}."
            if len(top_feats) >= 2 else
            f"Across the selected LIME cases, the DNN was most locally influenced by {top_feats[0]}."
        )

    with open(plain_dir / f"{split_name}_lime_plain_english.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_name": split_name,
                "global_summary": global_text,
                "local_cases": case_summaries,
            },
            f,
            indent=2,
        )

    print(f"  LIME done: explained_cases={len(case_positions)}, split={split_name}")
    if global_text:
        print(f"  Plain-English LIME summary for {split_name}:")
        print(f"    Global: {global_text}")
        for row in case_summaries:
            print(f"    Case {row['case_id']}: {row['plain_english']}")

    print("\nDone.")
    print(f"Saved local summaries: {tables_dir / f'{split_name}_lime_local_case_summaries.csv'}")
    print(f"Saved feature weights: {tables_dir / f'{split_name}_lime_case_feature_weights.csv'}")
    print(f"Saved aggregate importance: {tables_dir / f'{split_name}_lime_aggregate_importance.csv'}")
    print(f"Saved figures in: {figures_dir}")
    print(f"Saved HTML in: {html_dir}")
    print(f"Saved plain English in: {plain_dir}")


if __name__ == "__main__":
    main()
