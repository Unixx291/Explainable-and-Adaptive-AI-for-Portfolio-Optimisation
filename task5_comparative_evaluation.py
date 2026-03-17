from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    # Inputs from previous stages
    "task3_outdir": "results_task3_classical",
    "task4_1_outdir": "results_task4_1_dnn_only",
    "task4_2_outdir": "results_task4_2_dnn_rl",

    # Output for comparative evaluation
    "task5_outdir": "results_task5_comparative_evaluation",

    # Metrics to include in tables/plots
    "metrics": [
        "net_ann_return",
        "net_ann_vol",
        "net_sharpe",
        "net_sortino",
        "net_cumulative_return",
        "net_max_drawdown",
        "turnover_total",
        "tc_total",
    ],

    # Equity comparison plots
    "make_equity_comparison_plots": True,
    "equity_compare_all_common_splits": True,
    "equity_compare_recent_n": 6,
}


DISPLAY_NAMES = {
    "markowitz": "Markowitz",
    "black_litterman": "Black-Litterman",
    "dnn_only": "DNN-only",
    "dnn_rl": "DNN+RL",
}

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

# True means higher is better, False means lower is better.
METRIC_DIRECTIONS = {
    "net_ann_return": True,
    "net_ann_vol": False,
    "net_sharpe": True,
    "net_sortino": True,
    "net_cumulative_return": True,
    "net_max_drawdown": True,  # closer to zero is better
    "turnover_total": False,
    "tc_total": False,
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _standardise_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model" not in df.columns or "split" not in df.columns:
        raise ValueError("Result file must contain 'split' and 'model' columns.")

    keep_cols = [
        c for c in [
            "split", "model", "train_start", "train_end", "test_start", "test_end", "n_reb",
            "turnover_total", "tc_total", "net_ann_return", "net_ann_vol", "net_sharpe",
            "net_sortino", "net_cumulative_return", "net_max_drawdown"
        ] if c in df.columns
    ]
    df = df[keep_cols].copy()
    return df


def load_all_results(task3_outdir: Path, task4_1_outdir: Path, task4_2_outdir: Path) -> pd.DataFrame:
    task3 = _standardise_results(_load_csv(task3_outdir / "task3_split_results.csv"))
    task4_1 = _standardise_results(_load_csv(task4_1_outdir / "task4_1_split_results.csv"))
    task4_2 = _standardise_results(_load_csv(task4_2_outdir / "task4_2_split_results.csv"))

    all_results = pd.concat([task3, task4_1, task4_2], axis=0, ignore_index=True)
    all_results["model_display"] = all_results["model"].map(DISPLAY_NAMES).fillna(all_results["model"])
    return all_results


def filter_common_splits(all_results: pd.DataFrame, required_models: Iterable[str]) -> pd.DataFrame:
    required_models = list(required_models)
    counts = all_results.groupby("split")["model"].apply(lambda s: set(s.tolist()))
    common_splits = sorted([split for split, models in counts.items() if all(m in models for m in required_models)])
    return all_results[all_results["split"].isin(common_splits)].copy()


def make_wide_metrics(results_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    wide = results_df.pivot(index="split", columns="model", values=metrics)
    wide = wide.sort_index(axis=0)
    return wide


def summary_by_model(results_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    grouped = results_df.groupby(["model", "model_display"])[metrics].agg(["mean", "std", "median", "min", "max"])
    return grouped


def compute_win_counts(results_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        higher_is_better = METRIC_DIRECTIONS.get(metric, True)
        win_counts = {m: 0 for m in sorted(results_df["model"].unique())}
        for _, g in results_df.groupby("split"):
            vals = g[["model", metric]].dropna().copy()
            if vals.empty:
                continue
            best_val = vals[metric].max() if higher_is_better else vals[metric].min()
            winners = vals.loc[np.isclose(vals[metric], best_val, rtol=1e-10, atol=1e-12), "model"].tolist()
            for w in winners:
                win_counts[w] += 1
        for model, count in win_counts.items():
            rows.append({
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "model": model,
                "model_display": DISPLAY_NAMES.get(model, model),
                "win_count": int(count),
            })
    return pd.DataFrame(rows)


def compute_average_ranks(results_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rank_rows = []
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        higher_is_better = METRIC_DIRECTIONS.get(metric, True)
        for split, g in results_df.groupby("split"):
            vals = g[["model", metric]].dropna().copy()
            if vals.empty:
                continue
            vals["rank"] = vals[metric].rank(ascending=not higher_is_better, method="average")
            vals["split"] = split
            vals["metric"] = metric
            rank_rows.append(vals[["split", "metric", "model", "rank"]])
    if not rank_rows:
        return pd.DataFrame()
    ranks = pd.concat(rank_rows, axis=0, ignore_index=True)
    avg_ranks = ranks.groupby(["metric", "model"], as_index=False)["rank"].mean()
    avg_ranks["model_display"] = avg_ranks["model"].map(DISPLAY_NAMES).fillna(avg_ranks["model"])
    avg_ranks["metric_label"] = avg_ranks["metric"].map(METRIC_LABELS).fillna(avg_ranks["metric"])
    return avg_ranks


def compute_rl_deltas(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wide = results_df.pivot(index="split", columns="model")

    def _delta_against(base_model: str, compare_model: str) -> pd.DataFrame:
        rows = []
        metrics = [
            "net_ann_return", "net_ann_vol", "net_sharpe", "net_sortino",
            "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total"
        ]
        for split in wide.index:
            if compare_model not in wide.columns.get_level_values(1) or base_model not in wide.columns.get_level_values(1):
                continue
            row = {"split": split, "base_model": base_model, "compare_model": compare_model}
            valid_any = False
            for metric in metrics:
                try:
                    comp_val = wide.loc[split, (metric, compare_model)]
                    base_val = wide.loc[split, (metric, base_model)]
                except Exception:
                    continue
                row[f"delta_{metric}"] = comp_val - base_val
                valid_any = True
            if valid_any:
                rows.append(row)
        return pd.DataFrame(rows)

    rl_vs_dnn = _delta_against("dnn_only", "dnn_rl")
    rl_vs_bl = _delta_against("black_litterman", "dnn_rl")
    return rl_vs_dnn, rl_vs_bl


def _read_equity_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_equity_comparison(split_name: str, task3_outdir: Path, task4_1_outdir: Path, task4_2_outdir: Path) -> pd.DataFrame:
    task3_eq = _read_equity_csv(task3_outdir / "equity_curves" / f"{split_name}_equity.csv")
    task4_1_eq = _read_equity_csv(task4_1_outdir / "equity_curves" / f"{split_name}_equity.csv")
    task4_2_eq = _read_equity_csv(task4_2_outdir / "equity_curves" / f"{split_name}_equity.csv")

    merged = pd.concat([
        task3_eq[[c for c in task3_eq.columns if c.endswith("_net")]],
        task4_1_eq[[c for c in task4_1_eq.columns if c.endswith("_net")]],
        task4_2_eq[[c for c in task4_2_eq.columns if c.endswith("_net")]],
    ], axis=1).sort_index()

    rename_map = {
        "markowitz_net": "Markowitz",
        "black_litterman_net": "Black-Litterman",
        "dnn_only_net": "DNN-only",
        "dnn_rl_net": "DNN+RL",
    }
    merged = merged.rename(columns=rename_map)
    return merged


def save_equity_comparison_plot(eq_df: pd.DataFrame, split_name: str, outpath: Path) -> None:
    if eq_df.empty:
        return
    plt.figure(figsize=(9, 5))
    for col in eq_df.columns:
        plt.plot(eq_df.index, eq_df[col], label=col)
    plt.title(f"Comparative net equity curve: {split_name}")
    plt.xlabel("Date")
    plt.ylabel("Net equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_boxplot(results_df: pd.DataFrame, metric: str, outpath: Path) -> None:
    if results_df.empty or metric not in results_df.columns:
        return
    model_order = [m for m in ["markowitz", "black_litterman", "dnn_only", "dnn_rl"] if m in results_df["model"].unique()]
    data = [results_df.loc[results_df["model"] == m, metric].dropna().values for m in model_order]
    if not any(len(x) > 0 for x in data):
        return
    labels = [DISPLAY_NAMES.get(m, m) for m in model_order]
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data, tick_labels=labels)
    plt.title(f"Comparison of {METRIC_LABELS.get(metric, metric)}")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_mean_barplot(results_df: pd.DataFrame, metric: str, outpath: Path) -> None:
    if results_df.empty or metric not in results_df.columns:
        return
    model_order = [m for m in ["markowitz", "black_litterman", "dnn_only", "dnn_rl"] if m in results_df["model"].unique()]
    summary = results_df.groupby("model")[metric].mean().reindex(model_order)
    if summary.empty:
        return
    plt.figure(figsize=(7.5, 4.6))
    plt.bar([DISPLAY_NAMES.get(m, m) for m in summary.index], summary.values)
    plt.title(f"Mean {METRIC_LABELS.get(metric, metric)} by model")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_win_counts_plot(win_counts_df: pd.DataFrame, outpath: Path) -> None:
    if win_counts_df.empty:
        return
    plot_df = win_counts_df.pivot(index="metric_label", columns="model_display", values="win_count").fillna(0)
    plot_df = plot_df.sort_index()
    plt.figure(figsize=(11, 5.8))
    plot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Win counts by metric across common splits")
    plt.xlabel("Metric")
    plt.ylabel("Win count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def run_task5() -> None:
    task3_outdir = Path(CONFIG["task3_outdir"]).resolve()
    task4_1_outdir = Path(CONFIG["task4_1_outdir"]).resolve()
    task4_2_outdir = Path(CONFIG["task4_2_outdir"]).resolve()
    task5_outdir = Path(CONFIG["task5_outdir"]).resolve()

    _ensure_dir(task5_outdir)
    _ensure_dir(task5_outdir / "tables")
    _ensure_dir(task5_outdir / "figures")
    _ensure_dir(task5_outdir / "equity_comparisons")

    with open(task5_outdir / "task5_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    all_results = load_all_results(task3_outdir, task4_1_outdir, task4_2_outdir)
    all_results.to_csv(task5_outdir / "task5_all_results_long.csv", index=False)

    required_models = ["markowitz", "black_litterman", "dnn_only", "dnn_rl"]
    common_results = filter_common_splits(all_results, required_models=required_models)
    common_results.to_csv(task5_outdir / "task5_common_splits_long.csv", index=False)

    if common_results.empty:
        raise RuntimeError("No common splits found across Task 3, Task 4.1, and Task 4.2 outputs.")

    metrics = [m for m in CONFIG["metrics"] if m in common_results.columns]

    wide = make_wide_metrics(common_results, metrics)
    wide.to_csv(task5_outdir / "task5_common_splits_wide.csv")

    summary = summary_by_model(common_results, metrics)
    summary.to_csv(task5_outdir / "tables" / "task5_summary_by_model.csv")

    win_counts = compute_win_counts(common_results, metrics)
    win_counts.to_csv(task5_outdir / "tables" / "task5_win_counts.csv", index=False)

    avg_ranks = compute_average_ranks(common_results, metrics)
    avg_ranks.to_csv(task5_outdir / "tables" / "task5_average_ranks.csv", index=False)

    rl_vs_dnn, rl_vs_bl = compute_rl_deltas(common_results)
    rl_vs_dnn.to_csv(task5_outdir / "tables" / "task5_dnn_rl_minus_dnn_only.csv", index=False)
    rl_vs_bl.to_csv(task5_outdir / "tables" / "task5_dnn_rl_minus_black_litterman.csv", index=False)

    # Plot comparison metrics.
    for metric in metrics:
        save_boxplot(common_results, metric, task5_outdir / "figures" / f"task5_boxplot_{metric}.png")
        save_mean_barplot(common_results, metric, task5_outdir / "figures" / f"task5_meanbar_{metric}.png")
    save_win_counts_plot(win_counts, task5_outdir / "figures" / "task5_win_counts.png")

    # Equity comparisons.
    if bool(CONFIG["make_equity_comparison_plots"]):
        common_splits = sorted(common_results["split"].unique())
        if not bool(CONFIG["equity_compare_all_common_splits"]):
            n = int(max(1, CONFIG["equity_compare_recent_n"]))
            common_splits = common_splits[-n:]

        for split_name in common_splits:
            try:
                eq_df = load_equity_comparison(split_name, task3_outdir, task4_1_outdir, task4_2_outdir)
            except Exception:
                continue
            eq_df.to_csv(task5_outdir / "equity_comparisons" / f"{split_name}_equity_comparison.csv")
            save_equity_comparison_plot(
                eq_df,
                split_name,
                task5_outdir / "equity_comparisons" / f"{split_name}_equity_comparison.png",
            )

    print("\nDone.")
    print(f"Saved all results:      {task5_outdir / 'task5_all_results_long.csv'}")
    print(f"Saved common results:   {task5_outdir / 'task5_common_splits_long.csv'}")
    print(f"Saved summary tables:   {task5_outdir / 'tables'}")
    print(f"Saved metric figures:   {task5_outdir / 'figures'}")
    print(f"Saved equity compare:   {task5_outdir / 'equity_comparisons'}")


if __name__ == "__main__":
    run_task5()
