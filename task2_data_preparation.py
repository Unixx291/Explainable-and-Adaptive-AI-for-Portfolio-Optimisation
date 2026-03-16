"""
task2_data_preparation.py

Task 2: Data preparation module (Python)
- Download historical market data (yfinance primary)
- Clean and align (time-ordered, no shuffling)
- Feature engineering (returns, rolling stats, momentum, RSI, MACD, volume features)
- Leakage-safe walk-forward splits
- Train-only scaling per split
- Save panel dataset + per-split train/test + split metadata
- Save report-friendly descriptive tables and figures

Notes
- end date is treated as exclusive by yfinance in many cases, so use a day after your intended last day.
- This is academic tooling, not financial advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG (edit this)
# =========================
CONFIG = {
    # Assets and date range
    "tickers": ["SPY", "IWM", "EFA", "TLT"],
    "start": "2015-01-01",
    "end": "2026-01-01",
    "interval": "1d",

    # Output
    "outdir": "data_prepared_task2",
    "prefer_parquet": True,  # requires pyarrow; will fall back to CSV if missing

    # Cleaning
    "min_non_nan_ratio": 0.98,
    "forward_fill_limit": 3,

    # Features
    "rolling_windows": [5, 10, 20, 60],
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # Walk-forward (trading days)
    "train_days": 504,   # about 2 years
    "test_days": 63,     # about 3 months
    "step_days": 63,     # slide by one test window
}


# =========================
# Data classes
# =========================
@dataclass
class PrepConfig:
    tickers: List[str]
    start: str
    end: str
    interval: str
    outdir: str
    prefer_parquet: bool

    min_non_nan_ratio: float
    forward_fill_limit: int

    rolling_windows: List[int]
    rsi_window: int
    macd_fast: int
    macd_slow: int
    macd_signal: int

    train_days: int
    test_days: int
    step_days: int


# =========================
# IO helpers
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path_no_suffix: Path, prefer_parquet: bool) -> Path:
    """
    Writes Parquet if possible and preferred; otherwise writes CSV.
    Returns the actual path used.
    """
    if prefer_parquet:
        try:
            out = path_no_suffix.with_suffix(".parquet")
            df.to_parquet(out, index=True)
            return out
        except Exception:
            pass

    out = path_no_suffix.with_suffix(".csv")
    df.to_csv(out, index=True)
    return out


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


# =========================
# Step 1: Download data
# =========================
def download_data_yfinance(
    tickers: List[str], start: str, end: str, interval: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns wide prices and volumes:
    - prices columns: tickers, values: Adj Close
    - volumes columns: tickers, values: Volume
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    def extract(field: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            # Usually (Ticker, Field)
            try:
                wide = raw.xs(field, level=1, axis=1)
                return wide.copy()
            except Exception:
                # Sometimes (Field, Ticker)
                wide = raw.xs(field, level=0, axis=1)
                return wide.copy()
        else:
            # Single ticker case
            if field not in raw.columns:
                raise KeyError(f"Field '{field}' not found in yfinance output.")
            df = raw[[field]].copy()
            df.columns = [tickers[0]]
            return df

    prices = extract("Adj Close")
    volumes = extract("Volume")
    return prices, volumes


# =========================
# Step 2: Clean and align
# =========================
def clean_align(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    tickers: List[str],
    min_non_nan_ratio: float,
    forward_fill_limit: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Align indices
    idx = prices.index.union(volumes.index)
    prices = prices.reindex(idx).sort_index()
    volumes = volumes.reindex(idx).sort_index()

    # Fill small gaps using past values only (safe)
    prices = prices.ffill(limit=forward_fill_limit)
    volumes = volumes.ffill(limit=forward_fill_limit)

    # Keep tickers with enough data coverage
    kept = []
    for t in tickers:
        if t not in prices.columns:
            continue
        ratio = float(prices[t].notna().mean())
        if ratio >= min_non_nan_ratio:
            kept.append(t)

    if len(kept) < 2:
        raise RuntimeError(
            f"Too few tickers left after cleaning: {kept}. "
            "Try extending date range or reducing min_non_nan_ratio."
        )

    prices = prices[kept].copy()
    volumes = volumes[kept].copy()

    # Drop any dates with missing prices for any kept ticker
    prices = prices.dropna(how="any")
    volumes = volumes.reindex(prices.index).fillna(0.0)

    return prices, volumes, kept


# =========================
# Step 3-6: Features and panel
# =========================
def compute_rsi(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    prices: pd.DataFrame, fast: int, slow: int, signal: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def build_panel(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    rolling_windows: List[int],
    rsi_window: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
) -> pd.DataFrame:
    # Base series
    simple_ret = prices.pct_change()
    log_ret = np.log(prices / prices.shift(1))

    feats: Dict[str, pd.DataFrame] = {
        "log_return_1": log_ret,
        "simple_return_1": simple_ret,
    }

    # Rolling features and momentum
    for w in rolling_windows:
        feats[f"ret_mean_{w}"] = log_ret.rolling(w, min_periods=w).mean()
        feats[f"ret_std_{w}"] = log_ret.rolling(w, min_periods=w).std()
        feats[f"mom_{w}"] = (prices / prices.shift(w)) - 1.0

    # RSI and MACD
    feats[f"rsi_{rsi_window}"] = compute_rsi(prices, rsi_window)

    macd, macd_sig, macd_hist = compute_macd(prices, macd_fast, macd_slow, macd_signal)
    feats[f"macd_{macd_fast}_{macd_slow}"] = macd
    feats[f"macd_signal_{macd_signal}"] = macd_sig
    feats["macd_hist"] = macd_hist

    # Volume features
    feats["volume"] = volumes
    feats["volume_chg"] = volumes.pct_change()
    feats["volume_roll_mean_20"] = volumes.rolling(20, min_periods=20).mean()

    # Target: next-period log return
    target_next = log_ret.shift(-1).rename_axis("date")

    # Build panel (date, ticker)
    parts = []
    parts.append(prices.stack().rename("price"))

    for name, wide in feats.items():
        parts.append(wide.stack().rename(name))

    parts.append(target_next.stack().rename("target_next_log_return"))

    panel = pd.concat(parts, axis=1)
    panel.index.set_names(["date", "ticker"], inplace=True)

    # Drop NaNs created by rolling windows and shift(-1)
    panel = panel.dropna(how="any")

    # Cast to float32 for compact storage
    for c in panel.columns:
        panel[c] = panel[c].astype(np.float32)

    return panel


# =========================
# Step 7-8: Walk-forward splits + train-only scaling
# =========================
def generate_walk_forward_splits(
    dates: List[pd.Timestamp],
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    n = len(dates)
    max_start = n - (train_days + test_days)
    if max_start < 0:
        raise RuntimeError(
            f"Not enough dates for train_days={train_days} and test_days={test_days}. "
            f"Available dates: {n}."
        )

    splits = []
    start_i = 0
    while start_i <= max_start:
        train_start = dates[start_i]
        train_end = dates[start_i + train_days - 1]
        test_start = dates[start_i + train_days]
        test_end = dates[start_i + train_days + test_days - 1]
        splits.append((train_start, train_end, test_start, test_end))
        start_i += step_days

    return splits


def save_splits_with_scaling(
    panel: pd.DataFrame,
    outdir: Path,
    prefer_parquet: bool,
    train_days: int,
    test_days: int,
    step_days: int,
) -> None:
    dates = sorted(panel.index.get_level_values("date").unique())
    splits = generate_walk_forward_splits(dates, train_days, test_days, step_days)

    target_col = "target_next_log_return"
    feature_cols = [c for c in panel.columns if c != target_col]

    # Do not scale raw price by default (you can change this if you want)
    scale_cols = [c for c in feature_cols if c != "price"]

    splits_dir = outdir / "splits"
    ensure_dir(splits_dir)

    meta = {
        "n_dates_total": len(dates),
        "n_splits": len(splits),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "scaled_cols": scale_cols,
        "splits": [],
    }

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits, start=1):
        split_name = f"split_{i:03d}_{tr_s.date()}_{te_e.date()}"
        split_dir = splits_dir / split_name
        ensure_dir(split_dir)

        train_df = panel.loc[(slice(tr_s, tr_e), slice(None)), :].copy()
        test_df = panel.loc[(slice(te_s, te_e), slice(None)), :].copy()

        scaler = StandardScaler()
        scaler.fit(train_df[scale_cols].values)

        train_df.loc[:, scale_cols] = scaler.transform(train_df[scale_cols].values)
        test_df.loc[:, scale_cols] = scaler.transform(test_df[scale_cols].values)

        train_path = write_df(train_df, split_dir / "train", prefer_parquet)
        test_path = write_df(test_df, split_dir / "test", prefer_parquet)

        scaler_path = split_dir / "scaler.joblib"
        joblib.dump({"scaler": scaler, "scale_cols": scale_cols}, scaler_path)

        meta["splits"].append(
            {
                "name": split_name,
                "train_start": str(tr_s.date()),
                "train_end": str(tr_e.date()),
                "test_start": str(te_s.date()),
                "test_end": str(te_e.date()),
                "train_path": str(train_path),
                "test_path": str(test_path),
                "scaler_path": str(scaler_path),
            }
        )

    with open(outdir / "walk_forward_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# =========================
# Report tables and figures
# =========================
def _max_drawdown_from_returns(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def create_task2_report_outputs(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    panel: pd.DataFrame,
    outdir: Path,
    cfg: PrepConfig,
) -> None:
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    simple_rets = prices.pct_change().dropna(how="any")
    log_rets = np.log(prices / prices.shift(1)).dropna(how="any")

    # Overview table
    overview = pd.DataFrame(
        {
            "value": [
                len(prices.columns),
                str(prices.index.min().date()),
                str(prices.index.max().date()),
                int(len(prices.index)),
                int(panel.index.get_level_values("date").nunique()),
                int(panel.shape[0]),
                int(panel.shape[1] - 1),
                int(cfg.train_days),
                int(cfg.test_days),
                int(cfg.step_days),
            ]
        },
        index=[
            "n_assets",
            "price_start_date",
            "price_end_date",
            "n_price_rows",
            "n_panel_dates",
            "n_panel_rows",
            "n_features",
            "train_days",
            "test_days",
            "step_days",
        ],
    )
    _save_table(overview, tables_dir / "task2_dataset_overview.csv")

    # Per-asset descriptive statistics
    asset_stats = pd.DataFrame(index=prices.columns)
    asset_stats["start_date"] = prices.apply(lambda s: str(s.dropna().index.min().date()))
    asset_stats["end_date"] = prices.apply(lambda s: str(s.dropna().index.max().date()))
    asset_stats["n_obs"] = prices.notna().sum().astype(int)
    asset_stats["start_price"] = prices.iloc[0]
    asset_stats["end_price"] = prices.iloc[-1]
    asset_stats["cumulative_return"] = (prices.iloc[-1] / prices.iloc[0]) - 1.0
    asset_stats["ann_return"] = simple_rets.mean() * 252.0
    asset_stats["ann_vol"] = simple_rets.std(ddof=1) * np.sqrt(252.0)
    asset_stats["sharpe_rf0"] = asset_stats["ann_return"] / (asset_stats["ann_vol"] + 1e-12)
    asset_stats["min_daily_return"] = simple_rets.min()
    asset_stats["max_daily_return"] = simple_rets.max()
    asset_stats["avg_daily_volume"] = volumes.mean()
    asset_stats["max_drawdown"] = pd.Series({t: _max_drawdown_from_returns(simple_rets[t]) for t in simple_rets.columns})
    asset_stats = asset_stats.round(6)
    _save_table(asset_stats, tables_dir / "task2_asset_descriptive_stats.csv")

    # Correlation and feature summary tables
    corr = simple_rets.corr().round(6)
    _save_table(corr, tables_dir / "task2_return_correlation.csv")

    feature_cols = [c for c in panel.columns if c != "target_next_log_return"]
    feature_summary = panel[feature_cols].describe().T[["mean", "std", "min", "max"]].round(6)
    _save_table(feature_summary, tables_dir / "task2_feature_summary.csv")

    # Split schedule preview for easier methodology write-up
    split_dates = sorted(panel.index.get_level_values("date").unique())
    split_windows = generate_walk_forward_splits(split_dates, cfg.train_days, cfg.test_days, cfg.step_days)
    split_df = pd.DataFrame(
        [
            {
                "split": f"split_{i:03d}",
                "train_start": str(tr_s.date()),
                "train_end": str(tr_e.date()),
                "test_start": str(te_s.date()),
                "test_end": str(te_e.date()),
                "train_days": cfg.train_days,
                "test_days": cfg.test_days,
            }
            for i, (tr_s, tr_e, te_s, te_e) in enumerate(split_windows, start=1)
        ]
    )
    split_df.to_csv(tables_dir / "task2_split_schedule.csv", index=False)

    # Figure 1: normalised price paths
    norm_prices = prices / prices.iloc[0]
    plt.figure(figsize=(10, 6))
    for col in norm_prices.columns:
        plt.plot(norm_prices.index, norm_prices[col], label=col, linewidth=1.5)
    plt.title("Normalised asset price paths")
    plt.xlabel("Date")
    plt.ylabel("Growth of 1 unit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "task2_normalised_price_paths.png", dpi=200)
    plt.close()

    # Figure 2: annualised return vs annualised volatility scatter
    plt.figure(figsize=(8, 6))
    x = asset_stats["ann_vol"].astype(float)
    y = asset_stats["ann_return"].astype(float)
    plt.scatter(x, y)
    for ticker, xv, yv in zip(asset_stats.index, x, y):
        plt.annotate(ticker, (xv, yv), textcoords="offset points", xytext=(5, 5))
    plt.title("Per-asset annualised return vs annualised volatility")
    plt.xlabel("Annualised volatility")
    plt.ylabel("Annualised return")
    plt.tight_layout()
    plt.savefig(figures_dir / "task2_ann_return_vs_volatility.png", dpi=200)
    plt.close()

    # Figure 3: return correlation heatmap
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Daily return correlation heatmap")
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig(figures_dir / "task2_return_correlation_heatmap.png", dpi=200)
    plt.close()

    # Figure 4: rolling 60-day volatility
    rolling_vol = log_rets.rolling(60, min_periods=60).std() * np.sqrt(252.0)
    plt.figure(figsize=(10, 6))
    for col in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[col], label=col, linewidth=1.2)
    plt.title("Rolling 60-day annualised volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualised volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "task2_rolling_60d_volatility.png", dpi=200)
    plt.close()


# =========================
# Pipeline runner
# =========================
def run_task2_pipeline(cfg: PrepConfig) -> None:
    outdir = Path(cfg.outdir).resolve()
    ensure_dir(outdir)

    # Save config for reproducibility
    with open(outdir / "prep_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"[1/6] Downloading data (yfinance) for {len(cfg.tickers)} tickers...")
    prices, volumes = download_data_yfinance(cfg.tickers, cfg.start, cfg.end, cfg.interval)
    print(f"      Raw shapes: prices={prices.shape}, volumes={volumes.shape}")

    print("[2/6] Cleaning and aligning...")
    prices, volumes, kept = clean_align(
        prices,
        volumes,
        cfg.tickers,
        cfg.min_non_nan_ratio,
        cfg.forward_fill_limit,
    )
    print(f"      Kept tickers: {kept}")
    print(f"      Clean shapes: prices={prices.shape}, volumes={volumes.shape}")

    print("[3/6] Feature engineering and panel build...")
    panel = build_panel(
        prices,
        volumes,
        cfg.rolling_windows,
        cfg.rsi_window,
        cfg.macd_fast,
        cfg.macd_slow,
        cfg.macd_signal,
    )
    print(f"      Panel shape: {panel.shape}")
    print(f"      Unique dates in panel: {panel.index.get_level_values('date').nunique()}")

    print("[4/6] Saving full panel...")
    panel_path = write_df(panel, outdir / "panel_full", cfg.prefer_parquet)
    print(f"      Saved: {panel_path}")

    print("[5/6] Walk-forward splits + train-only scaling + save...")
    save_splits_with_scaling(
        panel,
        outdir,
        cfg.prefer_parquet,
        cfg.train_days,
        cfg.test_days,
        cfg.step_days,
    )

    print("[6/6] Exporting report-friendly tables and figures...")
    create_task2_report_outputs(prices, volumes, panel, outdir, cfg)
    print(f"      Done. Output folder: {outdir}")


if __name__ == "__main__":
    # Build config from CONFIG dict so you can click Run in PyCharm without arguments
    cfg = PrepConfig(
        tickers=[t.strip().upper() for t in CONFIG["tickers"]],
        start=CONFIG["start"],
        end=CONFIG["end"],
        interval=CONFIG["interval"],
        outdir=CONFIG["outdir"],
        prefer_parquet=bool(CONFIG["prefer_parquet"]),
        min_non_nan_ratio=float(CONFIG["min_non_nan_ratio"]),
        forward_fill_limit=int(CONFIG["forward_fill_limit"]),
        rolling_windows=[int(x) for x in CONFIG["rolling_windows"]],
        rsi_window=int(CONFIG["rsi_window"]),
        macd_fast=int(CONFIG["macd_fast"]),
        macd_slow=int(CONFIG["macd_slow"]),
        macd_signal=int(CONFIG["macd_signal"]),
        train_days=int(CONFIG["train_days"]),
        test_days=int(CONFIG["test_days"]),
        step_days=int(CONFIG["step_days"]),
    )

    run_task2_pipeline(cfg)
