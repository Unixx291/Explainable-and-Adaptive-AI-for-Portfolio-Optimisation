"""
task3_classical_baselines.py

Task 3: Classical model design (baseline framework)
- Loads Task 2 walk-forward splits (train/test) from disk
- Implements classical baselines with consistent constraints:
  1) Mean-Variance (Markowitz) via EfficientFrontier (max_sharpe)
  2) Black-Litterman (equilibrium prior + limited views + Idzorek confidence)
- Leakage-safe walk-forward evaluation:
  - Fit weights using only data available up to each rebalance date
  - Rebalance inside the test window every N trading days (default 21)
- Adds realism:
  - Transaction costs applied at each rebalance based on turnover
  - Reports turnover_total and tc_total, plus net_* metrics (after costs)
- Saves:
  - Per-split equity curves (gross + net)
  - Per-rebalance weights debug CSVs for each model
  - Per-split summary CSV and aggregated summary CSV

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# PyPortfolioOpt imports
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
except Exception as e:
    raise SystemExit(
        "PyPortfolioOpt is required.\n"
        "Install with: pip install PyPortfolioOpt cvxpy\n"
        f"Original error: {e}"
    )


# =========================
# CONFIG (edit this)
# =========================
CONFIG = {
    # Path to Task 2 output directory (contains walk_forward_meta.json and splits/)
    "task2_outdir": "data_prepared_task2",

    # Where to write Task 3 outputs
    "task3_outdir": "results_task3_classical",

    # Portfolio constraints (apply to ALL baselines for fair comparison)
    "weight_bounds": (0.0, 0.60),   # long-only, max 60% per asset
    "risk_free_rate": 0.0,

    # Optimisation objective for both baselines
    "objective": "max_sharpe",      # keep as requested

    # Black-Litterman settings
    "bl_delta": 2.5,                # risk aversion (constant for reproducibility)
    "bl_tau": 0.05,                 # tau parameter

    # New BL realism: limited views + Idzorek confidence
    "bl_view_top_k": 2,             # number of positive views (best assets by train mu)
    "bl_view_bottom_k": 2,          # number of negative views (worst assets by train mu)
    "bl_view_confidence": 0.60,     # same confidence for each view (0 to 1)
    "bl_use_idzorek": True,         # use omega="idzorek" with view_confidences

    # Backtest and realism
    "frequency": 252,               # trading days per year
    "rebalance_every_days": 21,     # approx monthly rebalance inside each test window
    "transaction_cost_rate": 0.001  # 10 bps per unit turnover (0.001 = 0.1%)
}


# =========================
# Helpers: IO and loading Task 2 data
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_panel(path: Path) -> pd.DataFrame:
    """
    Reads a panel dataframe saved by Task 2.
    Supports .parquet and .csv.
    Expected index: (date, ticker)
    """
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

    # Ensure datetime index level
    if not np.issubdtype(df.index.get_level_values("date").dtype, np.datetime64):
        df = df.copy()
        df.index = df.index.set_levels(
            [pd.to_datetime(df.index.levels[0]), df.index.levels[1]],
            level=[0, 1],
        )
    return df.sort_index()


def _panel_to_prices(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Converts panel (date, ticker) with a 'price' column into wide prices (date x ticker).
    """
    if "price" not in panel.columns:
        raise KeyError("Panel is missing required 'price' column.")
    return panel["price"].unstack("ticker").sort_index()


def _load_walk_forward_meta(task2_outdir: Path) -> Dict:
    meta_path = task2_outdir / "walk_forward_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}. Run Task 2 first.")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Optimisers (TRAIN only at each rebalance)
# =========================
def solve_markowitz(
    price_hist: pd.DataFrame,
    weight_bounds: Tuple[float, float],
    objective: str,
    risk_free_rate: float,
    frequency: int,
) -> Dict[str, float]:
    """
    Mean-Variance optimisation using Efficient Frontier.
    Uses price history available up to rebalance time (leakage-safe).
    """
    mu = expected_returns.mean_historical_return(price_hist, frequency=frequency)
    sigma = risk_models.CovarianceShrinkage(price_hist).ledoit_wolf()

    ef = EfficientFrontier(mu, sigma, weight_bounds=weight_bounds)

    try:
        if objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif objective == "min_volatility":
            ef.min_volatility()
        else:
            raise ValueError("CONFIG['objective'] must be 'max_sharpe' or 'min_volatility'")
        return ef.clean_weights()
    except Exception:
        tickers = list(price_hist.columns)
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}


def _select_bl_views(mu_train: pd.Series, top_k: int, bottom_k: int) -> Tuple[Dict[str, float], pd.Series]:
    """
    Select a small number of absolute views:
    - top_k assets by mu_train (positive views)
    - bottom_k assets by mu_train (negative views)
    Returns (views_dict, selected_mu_series_in_view_order).
    """
    mu_sorted = mu_train.sort_values(ascending=False)

    top_k = int(max(0, top_k))
    bottom_k = int(max(0, bottom_k))

    top = mu_sorted.head(top_k)
    bottom = mu_sorted.tail(bottom_k)

    selected = pd.concat([top, bottom])
    selected = selected[~selected.index.duplicated(keep="first")]

    views = {k: float(v) for k, v in selected.items()}
    return views, selected


def solve_black_litterman(
    price_hist: pd.DataFrame,
    weight_bounds: Tuple[float, float],
    objective: str,
    risk_free_rate: float,
    frequency: int,
    delta: float,
    tau: float,
    top_k: int,
    bottom_k: int,
    view_confidence: float,
    use_idzorek: bool,
) -> Dict[str, float]:
    """
    Black-Litterman with:
    - Equilibrium prior (pi) from market-implied returns
    - Equal market caps (controlled, reproducible assumption)
    - Limited views only for a few assets (top_k + bottom_k)
    - Idzorek method for omega via view_confidences (if enabled)

    This avoids BL collapsing into Markowitz due to full-coverage views.
    """
    tickers = list(price_hist.columns)

    sigma = risk_models.CovarianceShrinkage(price_hist).ledoit_wolf()
    mu_train = expected_returns.mean_historical_return(price_hist, frequency=frequency)

    # Controlled assumption: equal market caps -> equal market weights
    market_caps = pd.Series(1.0, index=tickers)
    pi = market_implied_prior_returns(market_caps, delta, sigma)

    # Limited views
    views, selected_mu = _select_bl_views(mu_train, top_k=top_k, bottom_k=bottom_k)

    if not views:
        # No views: pure prior
        try:
            bl = BlackLittermanModel(sigma, pi=pi, tau=tau)
            bl_returns = bl.bl_returns()
            bl_cov = bl.bl_cov()
            ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=weight_bounds)
            if objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == "min_volatility":
                ef.min_volatility()
            else:
                raise ValueError("CONFIG['objective'] must be 'max_sharpe' or 'min_volatility'")
            return ef.clean_weights()
        except Exception:
            w = 1.0 / len(tickers)
            return {t: w for t in tickers}

    # Confidence per view (same value for each view, but you can later make this asset-specific)
    view_confidence = float(np.clip(view_confidence, 0.0, 1.0))
    view_confidences = pd.Series(view_confidence, index=selected_mu.index)

    try:
        if use_idzorek:
            bl = BlackLittermanModel(
                sigma,
                pi=pi,
                absolute_views=views,
                omega="idzorek",
                view_confidences=view_confidences,
                tau=tau,
            )
        else:
            bl = BlackLittermanModel(
                sigma,
                pi=pi,
                absolute_views=views,
                tau=tau,
            )

        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=weight_bounds)
        if objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif objective == "min_volatility":
            ef.min_volatility()
        else:
            raise ValueError("CONFIG['objective'] must be 'max_sharpe' or 'min_volatility'")

        return ef.clean_weights()
    except Exception:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}


# =========================
# Backtesting and metrics
# =========================
def _weights_to_series(weights: Dict[str, float], tickers: List[str]) -> pd.Series:
    w = pd.Series({t: float(weights.get(t, 0.0)) for t in tickers})
    s = float(w.sum())
    if s == 0.0:
        w[:] = 1.0 / len(tickers)
    else:
        w = w / s
    return w


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
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
        }

    equity = (1.0 + port_rets).cumprod()
    ann_return = float((equity.iloc[-1] ** (frequency / len(port_rets))) - 1.0)
    ann_vol = float(port_rets.std(ddof=1) * np.sqrt(frequency))

    rf_daily = (1.0 + risk_free_rate) ** (1.0 / frequency) - 1.0
    excess = port_rets - rf_daily
    sharpe = float(excess.mean() / (port_rets.std(ddof=1) + 1e-12) * np.sqrt(frequency))

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": _max_drawdown(equity),
    }


def _make_rebalance_dates(test_dates: pd.DatetimeIndex, rebalance_every_days: int) -> List[pd.Timestamp]:
    if rebalance_every_days <= 0:
        return [pd.Timestamp(test_dates[0])]

    rebal_dates = []
    i = 0
    n = len(test_dates)
    while i < n:
        rebal_dates.append(pd.Timestamp(test_dates[i]))
        i += rebalance_every_days
    return rebal_dates


def simulate_rebalanced_portfolio(
    full_prices: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    rebalance_every_days: int,
    transaction_cost_rate: float,
    solver: Callable[[pd.DataFrame], pd.Series],
    lookback_days: int,
) -> Tuple[pd.Series, pd.Series, float, float, int, pd.DataFrame]:
    """
    Simulates a portfolio on the test window with periodic rebalancing.
    - solver(price_hist) returns weights (Series indexed by ticker)
    - Uses only data up to the rebalance date (exclusive) to fit weights
    - Applies transaction cost on the first day of each holding period

    Returns:
    - gross_port_rets (daily)
    - net_port_rets (daily, after tc)
    - turnover_total
    - tc_total
    - n_rebalances
    - rebalance_debug_df
    """
    full_prices = full_prices.sort_index()
    all_dates = full_prices.index

    # Safety: ensure boundaries exist in full_prices
    if train_end not in all_dates:
        train_end = all_dates[all_dates.get_indexer([train_end], method="pad")[0]]
    if test_start not in all_dates:
        test_start = all_dates[all_dates.get_indexer([test_start], method="backfill")[0]]
    if test_end not in all_dates:
        test_end = all_dates[all_dates.get_indexer([test_end], method="pad")[0]]

    test_prices = full_prices.loc[test_start:test_end]
    test_dates = test_prices.index
    if len(test_dates) == 0:
        return (pd.Series(dtype=float), pd.Series(dtype=float), 0.0, 0.0, 0, pd.DataFrame())

    rebalance_dates = _make_rebalance_dates(test_dates, rebalance_every_days)

    # Precompute daily asset returns over the full span (needs boundary day)
    asset_rets = full_prices.pct_change()

    gross_parts: List[pd.Series] = []
    net_parts: List[pd.Series] = []

    prev_w: pd.Series | None = None
    turnover_total = 0.0

    debug_rows = []

    for j, reb_date in enumerate(rebalance_dates):
        # Determine holding period end
        if j < len(rebalance_dates) - 1:
            hold_end = rebalance_dates[j + 1] - pd.Timedelta(days=1)
            hold_end = test_dates[test_dates <= hold_end][-1]  # snap to trading day
        else:
            hold_end = pd.Timestamp(test_dates[-1])

        # Fit window ends at the day before rebalance date (data available at rebalance time)
        hist_end = all_dates[all_dates.get_loc(reb_date) - 1] if all_dates.get_loc(reb_date) > 0 else train_end

        # Use rolling lookback for estimation (realistic and stable)
        hist_end_loc = all_dates.get_loc(hist_end)
        hist_start_loc = max(0, hist_end_loc - lookback_days + 1)
        price_hist = full_prices.iloc[hist_start_loc:hist_end_loc + 1].dropna(how="any")

        # Solve weights
        w = solver(price_hist)
        tickers = list(full_prices.columns)
        w = w.reindex(tickers).fillna(0.0)
        w = w / (w.sum() if float(w.sum()) != 0.0 else 1.0)

        # Turnover and transaction cost at this rebalance
        if prev_w is None:
            turnover = float(np.abs(w).sum())
        else:
            turnover = float(np.abs(w - prev_w).sum())

        tc = float(transaction_cost_rate * turnover)
        turnover_total += turnover

        # Holding period returns
        period_rets = asset_rets.loc[reb_date:hold_end].dropna(how="any")
        if period_rets.empty:
            prev_w = w
            continue

        port_gross = period_rets.dot(w)
        port_net = port_gross.copy()
        port_net.iloc[0] = port_net.iloc[0] - tc

        gross_parts.append(port_gross)
        net_parts.append(port_net)

        # Debug row includes weights
        row = {"rebalance_date": str(pd.Timestamp(reb_date).date()), "turnover": turnover, "tc": tc}
        for t in tickers:
            row[f"w_{t}"] = float(w.get(t, 0.0))
        debug_rows.append(row)

        prev_w = w

    gross = pd.concat(gross_parts).sort_index()
    net = pd.concat(net_parts).sort_index()

    # Keep only test window dates
    gross = gross.loc[test_start:test_end]
    net = net.loc[test_start:test_end]

    tc_total = float(transaction_cost_rate * turnover_total)
    debug_df = pd.DataFrame(debug_rows)

    return gross, net, turnover_total, tc_total, len(rebalance_dates), debug_df


# =========================
# Runner
# =========================
def run_task3() -> None:
    task2_outdir = Path(CONFIG["task2_outdir"]).resolve()
    task3_outdir = Path(CONFIG["task3_outdir"]).resolve()
    _ensure_dir(task3_outdir)

    meta = _load_walk_forward_meta(task2_outdir)
    splits = meta.get("splits", [])
    if not splits:
        raise RuntimeError("walk_forward_meta.json contains no splits. Check Task 2 outputs.")

    weights_dir = task3_outdir / "weights"
    curves_dir = task3_outdir / "equity_curves"
    rebal_dir = task3_outdir / "rebalances"
    _ensure_dir(weights_dir)
    _ensure_dir(curves_dir)
    _ensure_dir(rebal_dir)

    with open(task3_outdir / "task3_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    results_rows = []

    for split in splits:
        split_name = split["name"]
        train_path = Path(split["train_path"])
        test_path = Path(split["test_path"])

        if not train_path.is_absolute():
            train_path = (task2_outdir / train_path).resolve()
        if not test_path.is_absolute():
            test_path = (task2_outdir / test_path).resolve()

        print(f"\nRunning split: {split_name}")

        train_panel = _read_panel(train_path)
        test_panel = _read_panel(test_path)

        train_prices = _panel_to_prices(train_panel)
        test_prices = _panel_to_prices(test_panel)

        # Align tickers
        common = [t for t in train_prices.columns if t in test_prices.columns]
        train_prices = train_prices[common].dropna(how="any")
        test_prices = test_prices[common].dropna(how="any")

        if train_prices.shape[0] < 50 or test_prices.shape[0] < 10:
            print("  Skipping split due to insufficient data after alignment.")
            continue

        # Build full price history across train + test
        full_prices = pd.concat([train_prices, test_prices], axis=0)
        full_prices = full_prices[~full_prices.index.duplicated(keep="first")].sort_index()

        train_end = pd.to_datetime(split["train_end"])
        test_start = pd.to_datetime(split["test_start"])
        test_end = pd.to_datetime(split["test_end"])

        # Lookback window length equals training sample length (rolling estimation)
        lookback_days = int(len(train_prices.index))

        # Define solvers that return Series weights
        def markowitz_solver(price_hist: pd.DataFrame) -> pd.Series:
            w = solve_markowitz(
                price_hist=price_hist,
                weight_bounds=tuple(CONFIG["weight_bounds"]),
                objective=str(CONFIG["objective"]),
                risk_free_rate=float(CONFIG["risk_free_rate"]),
                frequency=int(CONFIG["frequency"]),
            )
            return _weights_to_series(w, list(price_hist.columns))

        def bl_solver(price_hist: pd.DataFrame) -> pd.Series:
            w = solve_black_litterman(
                price_hist=price_hist,
                weight_bounds=tuple(CONFIG["weight_bounds"]),
                objective=str(CONFIG["objective"]),
                risk_free_rate=float(CONFIG["risk_free_rate"]),
                frequency=int(CONFIG["frequency"]),
                delta=float(CONFIG["bl_delta"]),
                tau=float(CONFIG["bl_tau"]),
                top_k=int(CONFIG["bl_view_top_k"]),
                bottom_k=int(CONFIG["bl_view_bottom_k"]),
                view_confidence=float(CONFIG["bl_view_confidence"]),
                use_idzorek=bool(CONFIG["bl_use_idzorek"]),
            )
            return _weights_to_series(w, list(price_hist.columns))

        # Simulate both models
        gross_m, net_m, turnover_m, tc_m, nreb_m, reb_m_df = simulate_rebalanced_portfolio(
            full_prices=full_prices,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            rebalance_every_days=int(CONFIG["rebalance_every_days"]),
            transaction_cost_rate=float(CONFIG["transaction_cost_rate"]),
            solver=markowitz_solver,
            lookback_days=lookback_days,
        )

        gross_bl, net_bl, turnover_bl, tc_bl, nreb_bl, reb_bl_df = simulate_rebalanced_portfolio(
            full_prices=full_prices,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            rebalance_every_days=int(CONFIG["rebalance_every_days"]),
            transaction_cost_rate=float(CONFIG["transaction_cost_rate"]),
            solver=bl_solver,
            lookback_days=lookback_days,
        )

        # Metrics (net is the main reported performance)
        net_m_metrics = compute_metrics(net_m, int(CONFIG["frequency"]), float(CONFIG["risk_free_rate"]))
        net_bl_metrics = compute_metrics(net_bl, int(CONFIG["frequency"]), float(CONFIG["risk_free_rate"]))

        # Save equity curves (gross and net)
        eq_df = pd.DataFrame(
            {
                "markowitz_gross": (1.0 + gross_m).cumprod(),
                "markowitz_net": (1.0 + net_m).cumprod(),
                "black_litterman_gross": (1.0 + gross_bl).cumprod(),
                "black_litterman_net": (1.0 + net_bl).cumprod(),
            }
        )
        eq_df.to_csv(curves_dir / f"{split_name}_equity.csv", index=True)

        # Save rebalance debug weights
        reb_m_df.to_csv(rebal_dir / f"{split_name}_markowitz_rebalances.csv", index=False)
        reb_bl_df.to_csv(rebal_dir / f"{split_name}_black_litterman_rebalances.csv", index=False)

        # Save "final weights" for quick reference (last rebalance weights)
        if not reb_m_df.empty:
            last_m = reb_m_df.iloc[-1].to_dict()
            final_m = {k.replace("w_", ""): float(v) for k, v in last_m.items() if str(k).startswith("w_")}
        else:
            final_m = {}

        if not reb_bl_df.empty:
            last_bl = reb_bl_df.iloc[-1].to_dict()
            final_bl = {k.replace("w_", ""): float(v) for k, v in last_bl.items() if str(k).startswith("w_")}
        else:
            final_bl = {}

        with open(weights_dir / f"{split_name}_markowitz_final.json", "w", encoding="utf-8") as f:
            json.dump(final_m, f, indent=2)
        with open(weights_dir / f"{split_name}_black_litterman_final.json", "w", encoding="utf-8") as f:
            json.dump(final_bl, f, indent=2)

        # Record results
        results_rows.append(
            {
                "split": split_name,
                "model": "markowitz",
                "train_start": split["train_start"],
                "train_end": split["train_end"],
                "test_start": split["test_start"],
                "test_end": split["test_end"],
                "n_reb": int(nreb_m),
                "turnover_total": float(turnover_m),
                "tc_total": float(tc_m),
                "net_ann_return": float(net_m_metrics["ann_return"]),
                "net_ann_vol": float(net_m_metrics["ann_vol"]),
                "net_sharpe": float(net_m_metrics["sharpe"]),
                "net_cumulative_return": float(net_m_metrics["cumulative_return"]),
                "net_max_drawdown": float(net_m_metrics["max_drawdown"]),
            }
        )
        results_rows.append(
            {
                "split": split_name,
                "model": "black_litterman",
                "train_start": split["train_start"],
                "train_end": split["train_end"],
                "test_start": split["test_start"],
                "test_end": split["test_end"],
                "n_reb": int(nreb_bl),
                "turnover_total": float(turnover_bl),
                "tc_total": float(tc_bl),
                "net_ann_return": float(net_bl_metrics["ann_return"]),
                "net_ann_vol": float(net_bl_metrics["ann_vol"]),
                "net_sharpe": float(net_bl_metrics["sharpe"]),
                "net_cumulative_return": float(net_bl_metrics["cumulative_return"]),
                "net_max_drawdown": float(net_bl_metrics["max_drawdown"]),
            }
        )

        print(
            f"  Markowitz: net_ann_return={net_m_metrics['ann_return']:.4f}, "
            f"net_sharpe={net_m_metrics['sharpe']:.3f}, net_mdd={net_m_metrics['max_drawdown']:.3f}, "
            f"turnover_total={turnover_m:.3f}, tc_total={tc_m:.5f}, n_reb={nreb_m}"
        )
        print(
            f"  BL:        net_ann_return={net_bl_metrics['ann_return']:.4f}, "
            f"net_sharpe={net_bl_metrics['sharpe']:.3f}, net_mdd={net_bl_metrics['max_drawdown']:.3f}, "
            f"turnover_total={turnover_bl:.3f}, tc_total={tc_bl:.5f}, n_reb={nreb_bl}"
        )

    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(task3_outdir / "task3_split_results.csv", index=False)

    summary = (
        results_df
        .groupby("model")[["net_ann_return", "net_ann_vol", "net_sharpe", "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(task3_outdir / "task3_summary_by_model.csv")

    print("\nDone.")
    print(f"Saved split results: {task3_outdir / 'task3_split_results.csv'}")
    print(f"Saved summary:       {task3_outdir / 'task3_summary_by_model.csv'}")
    print(f"Final weights in:    {weights_dir}")
    print(f"Equity curves in:    {curves_dir}")
    print(f"Rebalance debug in:  {rebal_dir}")


if __name__ == "__main__":
    run_task3()
