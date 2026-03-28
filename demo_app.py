from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FYP Demo - Adaptive Portfolio Optimisation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

PATHS = {
    "task2": ROOT / "data_prepared_task2",
    "task3": ROOT / "results_task3_classical",
    "task4_1": ROOT / "results_task4_1_dnn_only",
    "task4_2": ROOT / "results_task4_2_dnn_rl",
    "task5": ROOT / "results_task5_comparative_evaluation",
    "task6_1": ROOT / "results_task6_1_shap_explainability",
    "task6_2": ROOT / "results_task6_2_lime_explainability",
    "task6_3": ROOT / "results_task6_3_feature_ablation",
}

MODEL_DISPLAY = {
    "equal_weight": "Equal-weight (1/N)",
    "markowitz": "Markowitz",
    "black_litterman": "Black-Litterman",
    "dnn_only": "DNN-only",
    "dnn_rl": "DNN+RL",
}
DISPLAY_TO_MODEL = {v: k for k, v in MODEL_DISPLAY.items()}

METRIC_LABELS = {
    "net_ann_return": "Net annualised return",
    "net_ann_vol": "Net annualised volatility",
    "net_sharpe": "Net Sharpe ratio",
    "net_sortino": "Net Sortino ratio",
    "net_cumulative_return": "Net cumulative return",
    "net_max_drawdown": "Net max drawdown",
    "turnover_total": "Turnover total",
    "tc_total": "Transaction cost total",
}
LABEL_TO_METRIC = {v: k for k, v in METRIC_LABELS.items()}

TAB_NAMES = [
    "1. Overview",
    "2. Comparative results",
    "3. Split deep dive",
    "4. Explainability",
    "5. Feature ablation",
]

SUMMARY_METRICS = list(METRIC_LABELS.keys())


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def fmt_metric(value: object, metric: Optional[str] = None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if metric in {"net_ann_return", "net_ann_vol", "net_cumulative_return", "turnover_total", "tc_total"}:
        return f"{x:.4f}"
    if metric in {"net_sharpe", "net_sortino", "net_max_drawdown", "rank"}:
        return f"{x:.3f}"
    return f"{x:.4f}"


@st.cache_data(show_spinner=False)
def read_csv(path_str: str, **kwargs) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def read_json(path_str: str) -> Dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_image_bytes(path_str: str) -> Optional[bytes]:
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_bytes()


@st.cache_data(show_spinner=False)
def find_first_matching_file(root_str: str, pattern: str) -> str:
    root = Path(root_str)
    matches = sorted(root.glob(pattern))
    return str(matches[0]) if matches else ""


@st.cache_data(show_spinner=False)
def discover_artifacts(root_str: str) -> Dict[str, pd.DataFrame]:
    root = Path(root_str)
    data: Dict[str, pd.DataFrame] = {}
    data["prep_config"] = pd.DataFrame([read_json(str(root / "data_prepared_task2" / "prep_config.json"))])
    data["task2_overview"] = read_csv(str(root / "data_prepared_task2" / "tables" / "task2_dataset_overview.csv"), index_col=0)
    data["task2_split_schedule"] = read_csv(str(root / "data_prepared_task2" / "tables" / "task2_split_schedule.csv"))

    data["task5_common"] = read_csv(str(root / "results_task5_comparative_evaluation" / "task5_common_splits_long.csv"))
    data["task5_summary_raw"] = read_csv(str(root / "results_task5_comparative_evaluation" / "tables" / "task5_summary_by_model.csv"))
    data["task5_win_counts"] = read_csv(str(root / "results_task5_comparative_evaluation" / "tables" / "task5_win_counts.csv"))
    data["task5_average_ranks"] = read_csv(str(root / "results_task5_comparative_evaluation" / "tables" / "task5_average_ranks.csv"))
    data["task5_rl_vs_dnn"] = read_csv(str(root / "results_task5_comparative_evaluation" / "tables" / "task5_dnn_rl_minus_dnn_only.csv"))
    data["task5_rl_vs_bl"] = read_csv(str(root / "results_task5_comparative_evaluation" / "tables" / "task5_dnn_rl_minus_black_litterman.csv"))

    data["task4_1_diag"] = read_csv(str(root / "results_task4_1_dnn_only" / "task4_1_diagnostics_by_split.csv"))
    data["task4_2_traininfo"] = read_csv(str(root / "results_task4_2_dnn_rl" / "task4_2_training_info.csv"))

    data["task6_1_summary"] = read_csv(str(root / "results_task6_1_shap_explainability" / "task6_1_shap_summary.csv"))
    data["task6_1_local"] = read_csv(str(root / "results_task6_1_shap_explainability" / "task6_1_local_case_summaries.csv"))
    data["task6_1_global_text"] = read_csv(str(root / "results_task6_1_shap_explainability" / "task6_1_global_plain_english.csv"))
    data["task6_1_feature_trends"] = read_csv(str(root / "results_task6_1_shap_explainability" / "tables" / "task6_1_feature_trends_across_splits.csv"))
    data["task6_1_top1"] = read_csv(str(root / "results_task6_1_shap_explainability" / "tables" / "task6_1_top1_feature_counts.csv"))
    data["task6_1_top3"] = read_csv(str(root / "results_task6_1_shap_explainability" / "tables" / "task6_1_top3_feature_counts.csv"))

    lime_local_path = find_first_matching_file(root_str, "results_task6_2_lime_explainability/tables/*_lime_local_case_summaries.csv")
    lime_agg_path = find_first_matching_file(root_str, "results_task6_2_lime_explainability/tables/*_lime_aggregate_importance.csv")
    data["task6_2_local"] = read_csv(lime_local_path) if lime_local_path else pd.DataFrame()
    data["task6_2_agg"] = read_csv(lime_agg_path) if lime_agg_path else pd.DataFrame()

    data["task6_3_summary"] = read_csv(str(root / "results_task6_3_feature_ablation" / "tables" / "task6_3_ablation_summary_by_family.csv"))
    data["task6_3_plain"] = read_csv(str(root / "results_task6_3_feature_ablation" / "tables" / "task6_3_plain_english_summary.csv"))
    data["task6_3_delta"] = read_csv(str(root / "results_task6_3_feature_ablation" / "tables" / "task6_3_ablation_deltas.csv"))
    return data


DATA = discover_artifacts(str(ROOT))


def list_splits() -> List[str]:
    common = DATA["task5_common"]
    if not common.empty and "split" in common.columns:
        return sorted(common["split"].astype(str).unique().tolist())

    schedule = DATA["task2_split_schedule"]
    if schedule.empty:
        return []
    for candidate in ["split", "split_name", "name"]:
        if candidate in schedule.columns:
            return schedule[candidate].astype(str).tolist()
    return []


@st.cache_data(show_spinner=False)
def build_summary_table(common_df: pd.DataFrame) -> pd.DataFrame:
    if common_df.empty or "model" not in common_df.columns:
        return pd.DataFrame()
    metrics = [m for m in SUMMARY_METRICS if m in common_df.columns]
    if not metrics:
        return pd.DataFrame()
    grouped = common_df.groupby("model")[metrics].agg(["mean", "std", "median", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped["model_display"] = grouped["model"].map(display_model)
    ordered = ["equal_weight", "markowitz", "black_litterman", "dnn_only", "dnn_rl"]
    grouped["_order"] = grouped["model"].apply(lambda x: ordered.index(x) if x in ordered else 999)
    grouped = grouped.sort_values("_order").drop(columns="_order")
    return grouped


@st.cache_data(show_spinner=False)
def build_metric_means(common_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if common_df.empty or metric not in common_df.columns or "model" not in common_df.columns:
        return pd.DataFrame()
    out = common_df.groupby("model", as_index=False)[metric].mean()
    out["model_display"] = out["model"].map(display_model)
    ordered = ["equal_weight", "markowitz", "black_litterman", "dnn_only", "dnn_rl"]
    out["_order"] = out["model"].apply(lambda x: ordered.index(x) if x in ordered else 999)
    return out.sort_values("_order").drop(columns="_order")


@st.cache_data(show_spinner=False)
def build_metric_cards(common_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return build_metric_means(common_df, metric)


@st.cache_data(show_spinner=False)
def prepare_ranks_df(rank_df: pd.DataFrame, selected_metric: str) -> pd.DataFrame:
    if rank_df.empty:
        return pd.DataFrame()
    tmp = rank_df.copy()
    if "metric" in tmp.columns:
        metric_filtered = tmp.loc[tmp["metric"].astype(str) == selected_metric].copy()
        if not metric_filtered.empty:
            tmp = metric_filtered
    if "model_display" not in tmp.columns and "model" in tmp.columns:
        tmp["model_display"] = tmp["model"].map(display_model)

    value_col = None
    for candidate in ["average_rank", "rank"]:
        if candidate in tmp.columns:
            value_col = candidate
            break
    if value_col is None:
        numeric_cols = [c for c in tmp.columns if c not in {"model", "model_display", "metric", "metric_label"}]
        for c in numeric_cols:
            coerced = pd.to_numeric(tmp[c], errors="coerce")
            if coerced.notna().any():
                tmp[c] = coerced
                value_col = c
                break
    if value_col is None:
        return pd.DataFrame()

    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    ordered = ["equal_weight", "markowitz", "black_litterman", "dnn_only", "dnn_rl"]
    if "model" in tmp.columns:
        tmp["_order"] = tmp["model"].apply(lambda x: ordered.index(x) if x in ordered else 999)
        tmp = tmp.sort_values("_order").drop(columns="_order")
    tmp.attrs["value_col"] = value_col
    return tmp


@st.cache_data(show_spinner=False)
def summarise_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in df.columns if c.startswith("delta_")]
    if not numeric_cols:
        return df
    out = df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean().reset_index()
    out.columns = ["delta_metric", "mean_delta"]
    return out



def read_equity_comparison(split_name: str) -> pd.DataFrame:
    path = PATHS["task5"] / "equity_comparisons" / f"{split_name}_equity_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df



def read_rebalances(split_name: str, model_key: str) -> pd.DataFrame:
    if model_key in {"equal_weight", "markowitz", "black_litterman"}:
        path = PATHS["task3"] / "rebalances" / f"{split_name}_{model_key}_rebalances.csv"
    elif model_key == "dnn_only":
        path = PATHS["task4_1"] / "rebalances" / f"{split_name}_dnn_only_rebalances.csv"
    elif model_key == "dnn_rl":
        path = PATHS["task4_2"] / "rebalances" / f"{split_name}_dnn_rl_rebalances.csv"
    else:
        return pd.DataFrame()
    return read_csv(str(path))



def read_final_weights(split_name: str, model_key: str) -> Dict[str, float]:
    if model_key in {"equal_weight", "markowitz", "black_litterman"}:
        path = PATHS["task3"] / "weights" / f"{split_name}_{model_key}_final.json"
    elif model_key == "dnn_only":
        path = PATHS["task4_1"] / "weights" / f"{split_name}_dnn_only_final.json"
    elif model_key == "dnn_rl":
        path = PATHS["task4_2"] / "weights" / f"{split_name}_dnn_rl_final.json"
    else:
        return {}
    return read_json(str(path))



def line_chart(df: pd.DataFrame, title: str, ylabel: str = "") -> None:
    if df.empty:
        st.info("No data available for this chart yet.")
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            ax.plot(df.index, series, label=str(col))
    ax.set_title(title)
    ax.set_xlabel("Date")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)



def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, rotate: int = 0) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data available for this chart yet.")
        return
    plot_df = df[[x, y]].copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[y])
    if plot_df.empty:
        st.info("No numeric data available for this chart yet.")
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(plot_df[x].astype(str), plot_df[y].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)



def grouped_bar_from_pivot(df: pd.DataFrame, index_col: str, column_col: str, value_col: str, title: str) -> None:
    if df.empty:
        st.info("No data available for this chart yet.")
        return
    tmp = df[[index_col, column_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        st.info("No numeric data available for this chart yet.")
        return
    pivot = tmp.pivot(index=index_col, columns=column_col, values=value_col).fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(index_col.replace("_", " ").title())
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.legend(title="")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)



def scorecard(label: str, value: str, help_text: Optional[str] = None) -> None:
    st.metric(label=label, value=value, help=help_text)



def section_missing(paths: Iterable[Path]) -> None:
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        st.warning("Missing artefacts for this section: " + ", ".join(missing))


# Sidebar controls
st.sidebar.title("Demo controls")
available_splits = list_splits()
selected_split = st.sidebar.selectbox(
    "Validated split",
    options=available_splits or ["No splits found"],
    index=(len(available_splits) - 1 if available_splits else 0),
)
metric_options = list(METRIC_LABELS.values())
metric_default = METRIC_LABELS.get("net_sharpe", metric_options[0]) if metric_options else None
metric_choice = st.sidebar.selectbox(
    "Comparison metric",
    options=metric_options,
    index=(metric_options.index(metric_default) if metric_options and metric_default in metric_options else 0),
)
selected_metric = LABEL_TO_METRIC[metric_choice]

model_options = list(MODEL_DISPLAY.values())
model_default = MODEL_DISPLAY.get("dnn_rl", model_options[0]) if model_options else None
model_choice = st.sidebar.selectbox(
    "Deep-dive model",
    options=model_options,
    index=(model_options.index(model_default) if model_options and model_default in model_options else 0),
)
selected_model = DISPLAY_TO_MODEL[model_choice]

st.sidebar.markdown("---")
st.sidebar.subheader("Methodology status")
st.sidebar.success("Walk-forward evaluation")
st.sidebar.success("Train-only scaling")
st.sidebar.success("Transaction costs enabled")
if available_splits:
    st.sidebar.info(f"Validated split selected: {selected_split}")
else:
    st.sidebar.warning("Run Tasks 2 to 5 to populate the validated split views.")

st.title("Adaptive and Explainable Portfolio Optimisation")
st.caption(
    "A demo-ready evidence browser for the final year project: leakage-safe walk-forward evaluation, classical baselines, modular DNN + RL, SHAP, LIME, and feature ablation."
)

split_row_df = DATA["task5_common"]
if not split_row_df.empty and "split" in split_row_df.columns and selected_split in split_row_df["split"].astype(str).values:
    split_row = split_row_df.loc[split_row_df["split"].astype(str) == selected_split].iloc[0]
    st.write(
        f"**Selected split:** {selected_split}  |  **Train:** {split_row['train_start']} to {split_row['train_end']}  |  **Test:** {split_row['test_start']} to {split_row['test_end']}"
    )

overview_tab, compare_tab, split_tab, explain_tab, ablation_tab = st.tabs(TAB_NAMES)

with overview_tab:
    st.subheader("Project overview")
    cfg_df = DATA["prep_config"]
    schedule_df = DATA["task2_split_schedule"]
    overview_df = DATA["task2_overview"]

    c1, c2, c3, c4 = st.columns(4)
    prep_cfg = cfg_df.iloc[0].to_dict() if not cfg_df.empty else {}
    with c1:
        scorecard("Assets", str(len(prep_cfg.get("tickers", []))) if prep_cfg else "n/a")
    with c2:
        scorecard("Train days", str(prep_cfg.get("train_days", "n/a")))
    with c3:
        scorecard("Test days", str(prep_cfg.get("test_days", "n/a")))
    with c4:
        scorecard("Step days", str(prep_cfg.get("step_days", "n/a")))

    st.markdown("### Pipeline A at a glance")
    pipeline_cols = st.columns(5)
    pipeline_steps = [
        "**Task 2**\nMarket data, features, walk-forward splits",
        "**Task 3**\nEqual-weight, Markowitz, Black-Litterman baselines",
        "**Task 4.1**\nSupervised DNN predicts next-period signals",
        "**Task 4.2**\nPPO agent consumes features + cached DNN signals",
        "**Tasks 5-6.3**\nEvaluation, SHAP/LIME, feature ablation",
    ]
    for col, text in zip(pipeline_cols, pipeline_steps):
        with col:
            st.info(text)

    st.markdown("### Why this interface is structured this way")
    st.write(
        "The formal project conclusions come from validated walk-forward outputs, so this app is designed as a guided research dashboard rather than a generic backtesting product."
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Dataset and split summary")
        if not overview_df.empty:
            st.dataframe(overview_df, width="stretch")
        else:
            st.info("Task 2 dataset overview is not available yet.")

        if not schedule_df.empty:
            st.markdown("### Walk-forward split schedule")
            st.dataframe(schedule_df.tail(10), width="stretch", hide_index=True)
        else:
            st.info("Task 2 split schedule is not available yet.")

    with right:
        st.markdown("### Task 2 report figures")
        figure_paths = [
            PATHS["task2"] / "figures" / "task2_normalised_price_paths.png",
            PATHS["task2"] / "figures" / "task2_ann_return_vs_volatility.png",
            PATHS["task2"] / "figures" / "task2_return_correlation_heatmap.png",
        ]
        section_missing(figure_paths)
        for path in figure_paths:
            img = load_image_bytes(str(path))
            if img:
                st.image(img, caption=path.stem.replace("_", " "), width="stretch")

with compare_tab:
    st.subheader("Comparative model results across common validated splits")
    section_missing([
        PATHS["task5"] / "task5_common_splits_long.csv",
        PATHS["task5"] / "tables" / "task5_win_counts.csv",
        PATHS["task5"] / "tables" / "task5_average_ranks.csv",
    ])

    common_df = DATA["task5_common"]
    win_df = DATA["task5_win_counts"]
    rank_df = DATA["task5_average_ranks"]
    rl_dnn_df = DATA["task5_rl_vs_dnn"]
    rl_bl_df = DATA["task5_rl_vs_bl"]
    summary_df = build_summary_table(common_df)

    st.write("Use this tab to show the headline evidence first: all five model families compared on the same validated splits.")

    card_df = build_metric_cards(common_df, selected_metric)
    if not card_df.empty:
        cards = st.columns(max(1, len(card_df)))
        for col, (_, row) in zip(cards, card_df.iterrows()):
            with col:
                st.metric(str(row["model_display"]), fmt_metric(row[selected_metric], selected_metric))
    else:
        st.info("Task 5 common-split results are not available yet.")

    left, right = st.columns([1.3, 1])
    with left:
        chart_df = build_metric_means(common_df, selected_metric)
        if not chart_df.empty:
            bar_chart(chart_df, "model_display", selected_metric, f"Mean {metric_label(selected_metric)} by model", rotate=20)
        else:
            st.info("No common-split metric data is available for the selected metric.")

    with right:
        ranks = prepare_ranks_df(rank_df, selected_metric)
        if not ranks.empty:
            rank_value_col = ranks.attrs.get("value_col", "rank")
            metric_title = metric_label(selected_metric) if "metric" in rank_df.columns else "metrics"
            bar_chart(ranks, "model_display", rank_value_col, f"Average rank for {metric_title}", rotate=20)
        else:
            st.info("Task 5 average ranks are not available yet.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        if not win_df.empty:
            tmp = win_df.copy()
            if "metric" in tmp.columns and selected_metric in tmp["metric"].astype(str).values:
                tmp_metric = tmp.loc[tmp["metric"].astype(str) == selected_metric].copy()
                if not tmp_metric.empty:
                    if "model_display" not in tmp_metric.columns and "model" in tmp_metric.columns:
                        tmp_metric["model_display"] = tmp_metric["model"].map(display_model)
                    bar_chart(tmp_metric, "model_display", "win_count", f"Win counts for {metric_label(selected_metric)}", rotate=20)
                else:
                    grouped_bar_from_pivot(tmp, "metric_label", "model_display", "win_count", "Win counts by metric")
            else:
                if "model_display" not in tmp.columns and "model" in tmp.columns:
                    tmp["model_display"] = tmp["model"].map(display_model)
                grouped_bar_from_pivot(tmp, "metric_label", "model_display", "win_count", "Win counts by metric")
        else:
            st.info("Task 5 win counts are not available yet.")

    with bottom_right:
        st.markdown("### DNN+RL delta summaries")
        delta_cols = st.columns(2)
        with delta_cols[0]:
            if not rl_dnn_df.empty:
                st.markdown("**DNN+RL minus DNN-only**")
                st.dataframe(summarise_delta_table(rl_dnn_df), width="stretch", hide_index=True)
            else:
                st.info("No DNN+RL minus DNN-only delta table found.")
        with delta_cols[1]:
            if not rl_bl_df.empty:
                st.markdown("**DNN+RL minus Black-Litterman**")
                st.dataframe(summarise_delta_table(rl_bl_df), width="stretch", hide_index=True)
            else:
                st.info("No DNN+RL minus Black-Litterman delta table found.")

    if not summary_df.empty:
        st.markdown("### Summary statistics by model")
        st.dataframe(summary_df, width="stretch", hide_index=True)

with split_tab:
    st.subheader("Selected split deep dive")
    eq_df = read_equity_comparison(selected_split) if available_splits else pd.DataFrame()
    metrics_df = DATA["task5_common"]
    dnn_diag_df = DATA["task4_1_diag"]
    dnn_rl_train_df = DATA["task4_2_traininfo"]

    line_chart(eq_df, f"Comparative net equity curve: {selected_split}", ylabel="Net equity")

    if not metrics_df.empty and available_splits and "split" in metrics_df.columns:
        split_metrics = metrics_df.loc[metrics_df["split"].astype(str) == selected_split].copy()
        if not split_metrics.empty:
            split_metrics["model_display"] = split_metrics["model"].map(display_model)
            st.markdown("### Split-level metrics")
            metric_cards = st.columns(min(5, len(split_metrics)))
            for col, (_, row) in zip(metric_cards, split_metrics.iterrows()):
                with col:
                    st.metric(str(row["model_display"]), fmt_metric(row[selected_metric], selected_metric))
            show_cols = [
                "model_display", "net_ann_return", "net_ann_vol", "net_sharpe", "net_sortino",
                "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total", "n_reb",
                "train_start", "train_end", "test_start", "test_end",
            ]
            show_cols = [c for c in show_cols if c in split_metrics.columns]
            st.dataframe(split_metrics[show_cols], width="stretch", hide_index=True)
        else:
            st.info("No Task 5 split metrics found for the selected split.")

    st.markdown(f"### {model_choice} allocation behaviour")
    reb_df = read_rebalances(selected_split, selected_model) if available_splits else pd.DataFrame()
    final_weights = read_final_weights(selected_split, selected_model) if available_splits else {}

    left, right = st.columns([1.2, 1])
    with left:
        if not reb_df.empty:
            st.dataframe(reb_df, width="stretch", hide_index=True)
        else:
            st.info("No rebalance trace found for the selected model and split.")

    with right:
        if final_weights:
            weights_df = pd.DataFrame(
                {"ticker": list(final_weights.keys()), "final_weight": list(final_weights.values())}
            ).sort_values("final_weight", ascending=False)
            bar_chart(weights_df, "ticker", "final_weight", f"Final weights: {model_choice}")
            st.dataframe(weights_df, width="stretch", hide_index=True)
        else:
            st.info("No saved final weights found for the selected model and split.")

    extras_left, extras_right = st.columns(2)
    with extras_left:
        if selected_model == "dnn_only" and not dnn_diag_df.empty and "split" in dnn_diag_df.columns:
            row = dnn_diag_df.loc[dnn_diag_df["split"].astype(str) == selected_split]
            if not row.empty:
                st.markdown("### DNN-only forecast diagnostics")
                st.dataframe(row, width="stretch", hide_index=True)
        elif selected_model == "dnn_rl" and not dnn_rl_train_df.empty and "split" in dnn_rl_train_df.columns:
            row = dnn_rl_train_df.loc[dnn_rl_train_df["split"].astype(str) == selected_split]
            if not row.empty:
                st.markdown("### DNN+RL training information")
                st.dataframe(row, width="stretch", hide_index=True)

    with extras_right:
        model_fig_paths = {
            "dnn_only": PATHS["task4_1"] / "figures" / f"{selected_split}_weights.png",
            "dnn_rl": PATHS["task4_2"] / "figures" / f"{selected_split}_ppo_rewards.png",
        }
        fig_path = model_fig_paths.get(selected_model)
        if fig_path is not None:
            img = load_image_bytes(str(fig_path))
            if img:
                st.image(img, caption=fig_path.stem.replace("_", " "), width="stretch")

with explain_tab:
    st.subheader("Explainability: SHAP and LIME")
    st.write("These explanations target the supervised DNN signal generator feeding the allocation stage, not the PPO policy directly.")

    shap_summary_df = DATA["task6_1_summary"]
    shap_global_text_df = DATA["task6_1_global_text"]
    shap_local_df = DATA["task6_1_local"]
    shap_trends_df = DATA["task6_1_feature_trends"]
    shap_top1_df = DATA["task6_1_top1"]
    shap_top3_df = DATA["task6_1_top3"]
    lime_local_df = DATA["task6_2_local"]
    lime_agg_df = DATA["task6_2_agg"]

    split_for_explain = selected_split if not shap_summary_df.empty and "split" in shap_summary_df.columns and selected_split in shap_summary_df["split"].astype(str).tolist() else None
    if split_for_explain is None and not shap_summary_df.empty and "split" in shap_summary_df.columns:
        split_for_explain = str(shap_summary_df["split"].iloc[-1])

    st.markdown(f"### Active explainability split: {split_for_explain or 'n/a'}")

    top_left, top_right = st.columns([1.15, 1])
    with top_left:
        shap_img_path = PATHS["task6_1"] / "figures" / f"{split_for_explain}_shap_summary.png"
        img = load_image_bytes(str(shap_img_path)) if split_for_explain else None
        if img:
            st.image(img, caption="Global SHAP summary", width="stretch")
        else:
            st.info("SHAP summary plot not found for the selected explainability split.")

    with top_right:
        if not shap_global_text_df.empty and split_for_explain and "split" in shap_global_text_df.columns:
            row = shap_global_text_df.loc[shap_global_text_df["split"].astype(str) == split_for_explain]
            if not row.empty:
                record = row.iloc[0].to_dict()
                st.markdown("### Plain-English SHAP summary")
                st.success(str(record.get("summary_text", "No plain-English summary found.")))
        if not shap_summary_df.empty and split_for_explain and "split" in shap_summary_df.columns:
            row = shap_summary_df.loc[shap_summary_df["split"].astype(str) == split_for_explain]
            if not row.empty:
                st.dataframe(row, width="stretch", hide_index=True)

    mid_left, mid_right = st.columns(2)
    with mid_left:
        st.markdown("### Cross-split SHAP feature trends")
        if not shap_trends_df.empty:
            st.dataframe(shap_trends_df.head(10), width="stretch", hide_index=True)
        else:
            st.info("Task 6.1 cross-split SHAP trends are not available yet.")

    with mid_right:
        if not shap_top1_df.empty:
            bar_chart(shap_top1_df.head(10), "feature_label", "top1_count", "Top SHAP features by top-1 count", rotate=35)
        elif not shap_top3_df.empty:
            bar_chart(shap_top3_df.head(10), "feature_label", "top3_count", "Top SHAP features by top-3 count", rotate=35)
        else:
            st.info("Task 6.1 SHAP feature-count tables are not available yet.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("### Local SHAP cases")
        if not shap_local_df.empty and split_for_explain and "split" in shap_local_df.columns:
            local_cases = shap_local_df.loc[shap_local_df["split"].astype(str) == split_for_explain].copy()
            if not local_cases.empty:
                case_labels = [f"Case {int(r['case_rank'])}: {r['ticker']} on {r['date']}" for _, r in local_cases.iterrows()]
                selected_case_label = st.selectbox("SHAP case", case_labels)
                case_rank = int(selected_case_label.split(":")[0].replace("Case", "").strip())
                case_row = local_cases.loc[local_cases["case_rank"] == case_rank].iloc[0]
                st.info(str(case_row["summary_text"]))
                local_img = PATHS["task6_1"] / "local_cases" / f"{split_for_explain}_case_{case_rank}_local_bar.png"
                img = load_image_bytes(str(local_img))
                if img:
                    st.image(img, caption=local_img.stem.replace("_", " "), width="stretch")
                st.dataframe(local_cases, width="stretch", hide_index=True)
            else:
                st.info("No local SHAP cases found for this split.")
        else:
            st.info("Task 6.1 local SHAP summaries are not available yet.")

    with bottom_right:
        st.markdown("### LIME case study")
        if not lime_local_df.empty:
            case_col = "case_id" if "case_id" in lime_local_df.columns else "case_rank"
            split_col = "split_name" if "split_name" in lime_local_df.columns else ("split" if "split" in lime_local_df.columns else None)
            if split_col and selected_split in lime_local_df[split_col].astype(str).tolist():
                active_lime_df = lime_local_df.loc[lime_local_df[split_col].astype(str) == selected_split].copy()
            else:
                active_lime_df = lime_local_df.copy()
            lime_case_labels = [f"Case {int(r[case_col])}: {r['ticker']} on {r['date']}" for _, r in active_lime_df.iterrows()]
            selected_lime_case = st.selectbox("LIME case", lime_case_labels)
            lime_case_id = int(selected_lime_case.split(":")[0].replace("Case", "").strip())
            lime_case_row = active_lime_df.loc[active_lime_df[case_col] == lime_case_id].iloc[0]
            st.info(str(lime_case_row.get("plain_english", "No plain-English LIME summary found.")))
            lime_split_name = str(lime_case_row.get(split_col, "")).strip() if split_col else ""
            lime_img = PATHS["task6_2"] / "figures" / f"{lime_split_name}_case_{lime_case_id}_{lime_case_row['ticker']}_{lime_case_row['date']}_lime.png"
            img = load_image_bytes(str(lime_img))
            if img:
                st.image(img, caption=lime_img.stem.replace("_", " "), width="stretch")
            if not lime_agg_df.empty:
                bar_chart(lime_agg_df.head(10), "feature", "mean_abs_weight", "LIME aggregate local importance", rotate=35)
            st.dataframe(active_lime_df, width="stretch", hide_index=True)
        else:
            st.info("Task 6.2 LIME case study outputs are not available yet.")

with ablation_tab:
    st.subheader("Feature ablation")
    summary_df = DATA["task6_3_summary"]
    plain_df = DATA["task6_3_plain"]
    delta_df = DATA["task6_3_delta"]

    top_left, top_right = st.columns([1.05, 1])
    with top_left:
        img = load_image_bytes(str(PATHS["task6_3"] / "figures" / "task6_3_family_delta_sharpe.png"))
        if img:
            st.image(img, caption="Mean Sharpe degradation after removing a feature family", width="stretch")
        else:
            st.info("Task 6.3 family Sharpe degradation figure is not available yet.")

    with top_right:
        if not summary_df.empty:
            st.dataframe(summary_df, width="stretch", hide_index=True)
        else:
            st.info("Task 6.3 family summary is not available yet.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("### Plain-English family summaries")
        if not plain_df.empty:
            for _, row in plain_df.iterrows():
                label = row.get("family_label", row.get("family", "Feature family"))
                st.write(f"**{label}**")
                st.write(str(row.get("summary", "")))
        else:
            st.info("Task 6.3 plain-English summary is not available yet.")

    with lower_right:
        st.markdown("### Split-level ablation deltas")
        if not delta_df.empty and "family_label" in delta_df.columns:
            family_options = sorted(delta_df["family_label"].dropna().astype(str).unique().tolist())
            family_choice = st.selectbox("Feature family", family_options)
            family_df = delta_df.loc[delta_df["family_label"].astype(str) == family_choice].copy()
            if not family_df.empty:
                bar_chart(family_df, "split", "delta_net_sharpe", f"Sharpe delta by split: {family_choice}", rotate=45)
                st.dataframe(family_df, width="stretch", hide_index=True)
        else:
            st.info("Task 6.3 split-level deltas are not available yet.")

st.markdown("---")
st.caption(
    "This app is intended for validated demo artefacts produced by Tasks 2 to 6.3. If a section is empty, run the corresponding task script first so the saved outputs exist on disk."
)
