"""
task4_2_dnn_rl.py

Task 4.2: DNN + RL portfolio optimisation (Pipeline A)
- Loads leakage-safe walk-forward splits from Task 2
- Loads cached DNN prediction streams from Task 4.1
- Builds a custom Gym-style environment where observations contain:
  - engineered market features per asset
  - cached DNN signals aligned to each timestep
  - previous portfolio weights
- Trains PPO on the training window only
- Evaluates the trained policy on the out-of-sample test window
- Saves comparable portfolio metrics, equity curves, rebalance weights, and training logs

Important
- This stage depends on Task 2 and Task 4.1 outputs already being present on disk.
- By default, PPO model binaries are NOT saved to avoid repo bloat.
- This is academic tooling, not financial advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    gym = None
    spaces = None
    GYM_IMPORT_ERROR = e
else:
    GYM_IMPORT_ERROR = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as e:
    PPO = None
    Monitor = None
    DummyVecEnv = None
    SB3_IMPORT_ERROR = e
else:
    SB3_IMPORT_ERROR = None


CONFIG = {
    # Inputs from previous stages
    "task2_outdir": "data_prepared_task2",
    "task4_1_outdir": "results_task4_1_dnn_only",

    # Output for Task 4.2
    "task4_2_outdir": "results_task4_2_dnn_rl",

    # Scope: keep pilot_mode=True first for RL cost, then switch to False later
    "pilot_mode": True,
    "pilot_recent_splits": 3,

    # Feature selection and observation design
    "exclude_feature_cols": ["price"],
    "target_col": "target_next_log_return",
    "prediction_signal_col": "signal_for_trade",
    "use_prev_weights_in_obs": True,

    # Shared portfolio/backtest assumptions
    "weight_bounds": (0.0, 0.60),
    "rebalance_every_days": 21,
    "transaction_cost_rate": 0.001,
    "risk_free_rate": 0.0,
    "frequency": 252,

    # Reward shaping
    "downside_penalty": 1.0,
    "reward_scale": 100.0,

    # PPO settings
    "seed": 42,
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 64,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "total_timesteps": 5000,
    "verbose_learn": 0,
    "device": "auto",

    # Artifact saving
    "save_models": False,
}


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


def _load_prediction_file(task4_1_outdir: Path, split_name: str) -> pd.DataFrame:
    pred_path = task4_1_outdir / "predictions" / f"{split_name}_combined_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing cached DNN prediction file for {split_name}: {pred_path}. Run Task 4.1 first."
        )
    raw = pd.read_csv(pred_path)
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
    return df.sort_index()


def _panel_to_prices(panel: pd.DataFrame) -> pd.DataFrame:
    if "price" not in panel.columns:
        raise KeyError("Panel is missing required 'price' column.")
    return panel["price"].unstack("ticker").sort_index()


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


def _make_rebalance_dates(dates: pd.DatetimeIndex, rebalance_every_days: int) -> List[pd.Timestamp]:
    if len(dates) == 0:
        return []
    if rebalance_every_days <= 0:
        return [pd.Timestamp(dates[0])]
    out: List[pd.Timestamp] = []
    i = 0
    n = len(dates)
    while i < n:
        out.append(pd.Timestamp(dates[i]))
        i += rebalance_every_days
    return out


def _cap_and_normalise_weights(w: pd.Series, max_weight: float) -> pd.Series:
    w = w.clip(lower=0.0).astype(float)
    if float(w.sum()) <= 0.0:
        w[:] = 1.0 / len(w)
        return w

    w = w / w.sum()
    max_weight = float(max_weight)

    for _ in range(20):
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


def action_to_weights(action: np.ndarray, tickers: Sequence[str], max_weight: float) -> pd.Series:
    action = np.asarray(action, dtype=float).reshape(-1)
    if len(action) != len(tickers):
        raise ValueError(f"Action length {len(action)} does not match n_assets {len(tickers)}")
    z = action - np.max(action)
    exp_z = np.exp(np.clip(z, -20, 20))
    probs = exp_z / (exp_z.sum() + 1e-12)
    w = pd.Series(probs, index=list(tickers), dtype=float)
    return _cap_and_normalise_weights(w, max_weight=max_weight)


@dataclass
class RebalanceRecord:
    rebalance_date: pd.Timestamp
    hold_end: pd.Timestamp
    features: np.ndarray
    signals: np.ndarray
    period_rets: pd.DataFrame


def build_rebalance_records(
    panel_window: pd.DataFrame,
    pred_df: pd.DataFrame,
    prices_full: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    feature_cols: List[str],
    tickers: List[str],
    rebalance_every_days: int,
    signal_col: str,
) -> List[RebalanceRecord]:
    panel_window = panel_window.sort_index()
    prices_window = prices_full.loc[start_date:end_date, tickers].dropna(how="any")
    test_dates = prices_window.index
    if len(test_dates) == 0:
        return []

    asset_rets = prices_full[tickers].pct_change()
    rebalance_dates = _make_rebalance_dates(test_dates, rebalance_every_days)
    records: List[RebalanceRecord] = []

    for j, reb_date in enumerate(rebalance_dates):
        if j < len(rebalance_dates) - 1:
            hold_end = rebalance_dates[j + 1] - pd.Timedelta(days=1)
            elig = test_dates[test_dates <= hold_end]
            if len(elig) == 0:
                continue
            hold_end = pd.Timestamp(elig[-1])
        else:
            hold_end = pd.Timestamp(test_dates[-1])

        period_rets = asset_rets.loc[reb_date:hold_end, tickers].dropna(how="any")
        if period_rets.empty:
            continue

        try:
            date_block = panel_window.xs(reb_date, level="date").reindex(tickers)
        except Exception:
            continue
        if date_block[feature_cols].isna().any().any():
            continue
        features = date_block[feature_cols].astype(np.float32).values

        try:
            sig_today = pred_df.xs(reb_date, level="date")[signal_col].reindex(tickers)
        except Exception:
            sig_today = pd.Series(index=tickers, data=np.nan)
        sig_today = sig_today.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

        records.append(
            RebalanceRecord(
                rebalance_date=pd.Timestamp(reb_date),
                hold_end=hold_end,
                features=features,
                signals=sig_today.values.astype(np.float32),
                period_rets=period_rets.astype(np.float32),
            )
        )

    return records


if gym is not None:
    class PortfolioRLRebalanceEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            records: Sequence[RebalanceRecord],
            tickers: Sequence[str],
            max_weight: float,
            transaction_cost_rate: float,
            downside_penalty: float,
            reward_scale: float,
            use_prev_weights_in_obs: bool,
        ) -> None:
            super().__init__()
            self.records = list(records)
            self.tickers = list(tickers)
            self.n_assets = len(self.tickers)
            self.max_weight = float(max_weight)
            self.transaction_cost_rate = float(transaction_cost_rate)
            self.downside_penalty = float(downside_penalty)
            self.reward_scale = float(reward_scale)
            self.use_prev_weights_in_obs = bool(use_prev_weights_in_obs)

            if len(self.records) == 0:
                raise ValueError("Environment received no rebalance records.")

            feat_dim = int(np.prod(self.records[0].features.shape))
            sig_dim = int(len(self.records[0].signals))
            prev_dim = self.n_assets if self.use_prev_weights_in_obs else 0
            obs_dim = feat_dim + sig_dim + prev_dim

            self.observation_space = spaces.Box(
                low=-1e6,
                high=1e6,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(self.n_assets,),
                dtype=np.float32,
            )

            self.idx = 0
            self.prev_weights = np.repeat(1.0 / self.n_assets, self.n_assets).astype(np.float32)

        def _obs_for_index(self, idx: int) -> np.ndarray:
            rec = self.records[idx]
            parts = [rec.features.reshape(-1), rec.signals.reshape(-1)]
            if self.use_prev_weights_in_obs:
                parts.append(self.prev_weights.reshape(-1))
            obs = np.concatenate(parts).astype(np.float32)
            return obs

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            self.idx = 0
            self.prev_weights = np.repeat(1.0 / self.n_assets, self.n_assets).astype(np.float32)
            obs = self._obs_for_index(self.idx)
            info = {
                "rebalance_date": str(self.records[self.idx].rebalance_date.date()),
            }
            return obs, info

        def step(self, action: np.ndarray):
            rec = self.records[self.idx]
            w = action_to_weights(action, self.tickers, max_weight=self.max_weight)
            prev_w = pd.Series(self.prev_weights, index=self.tickers, dtype=float)
            turnover = float(np.abs(w - prev_w).sum())
            tc = float(self.transaction_cost_rate * turnover)

            daily_port_gross = rec.period_rets.dot(w).astype(float)
            gross_period_return = float((1.0 + daily_port_gross).prod() - 1.0)
            net_period_return = gross_period_return - tc
            downside_term = abs(min(net_period_return, 0.0))
            reward = (net_period_return - self.downside_penalty * downside_term) * self.reward_scale

            self.prev_weights = w.values.astype(np.float32)
            self.idx += 1
            terminated = self.idx >= len(self.records)
            truncated = False
            if not terminated:
                obs = self._obs_for_index(self.idx)
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)

            info = {
                "rebalance_date": str(rec.rebalance_date.date()),
                "hold_end": str(rec.hold_end.date()),
                "gross_period_return": gross_period_return,
                "net_period_return": net_period_return,
                "turnover": turnover,
                "tc": tc,
            }
            return obs, float(reward), terminated, truncated, info


else:
    PortfolioRLRebalanceEnv = object


def simulate_trained_policy(
    model: "PPO",
    records: Sequence[RebalanceRecord],
    tickers: Sequence[str],
    max_weight: float,
    transaction_cost_rate: float,
    use_prev_weights_in_obs: bool,
) -> Tuple[pd.Series, pd.Series, float, float, int, pd.DataFrame]:
    if len(records) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0.0, 0.0, 0, pd.DataFrame()

    prev_weights = pd.Series(1.0 / len(tickers), index=list(tickers), dtype=float)
    gross_parts: List[pd.Series] = []
    net_parts: List[pd.Series] = []
    debug_rows: List[Dict[str, float]] = []
    turnover_total = 0.0

    for rec in records:
        obs_parts = [rec.features.reshape(-1), rec.signals.reshape(-1)]
        if use_prev_weights_in_obs:
            obs_parts.append(prev_weights.values.reshape(-1))
        obs = np.concatenate(obs_parts).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        w = action_to_weights(action, tickers, max_weight=max_weight)

        turnover = float(np.abs(w - prev_weights).sum())
        tc = float(transaction_cost_rate * turnover)
        turnover_total += turnover

        port_gross = rec.period_rets.dot(w).astype(float)
        port_net = port_gross.copy()
        port_net.iloc[0] = port_net.iloc[0] - tc

        gross_parts.append(port_gross)
        net_parts.append(port_net)

        row = {
            "rebalance_date": str(rec.rebalance_date.date()),
            "hold_end": str(rec.hold_end.date()),
            "turnover": turnover,
            "tc": tc,
        }
        for i, t in enumerate(tickers):
            row[f"signal_{t}"] = float(rec.signals[i])
            row[f"w_{t}"] = float(w.iloc[i])
        debug_rows.append(row)
        prev_weights = w

    gross = pd.concat(gross_parts).sort_index() if gross_parts else pd.Series(dtype=float)
    net = pd.concat(net_parts).sort_index() if net_parts else pd.Series(dtype=float)
    tc_total = float(transaction_cost_rate * turnover_total)
    debug_df = pd.DataFrame(debug_rows)
    return gross, net, turnover_total, tc_total, len(debug_rows), debug_df


def save_equity_plot(eq_df: pd.DataFrame, outpath: Path, split_name: str) -> None:
    if eq_df.empty:
        return
    plt.figure(figsize=(8, 4.5))
    for col in eq_df.columns:
        plt.plot(eq_df.index, eq_df[col], label=col)
    plt.title(f"Task 4.2 DNN+RL equity curve: {split_name}")
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
    plt.title(f"Task 4.2 DNN+RL weights by rebalance: {split_name}")
    plt.xlabel("Rebalance date")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_monitor_plot(monitor_csv: Path, outpath: Path, split_name: str) -> None:
    if not monitor_csv.exists():
        return
    try:
        df = pd.read_csv(monitor_csv, comment="#")
    except Exception:
        return
    if df.empty or "r" not in df.columns:
        return
    rewards = df["r"].astype(float)
    rolling = rewards.rolling(10, min_periods=1).mean()
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(rewards) + 1), rewards, alpha=0.35, label="Episode reward")
    plt.plot(np.arange(1, len(rolling) + 1), rolling, label="Rolling mean (10)")
    plt.title(f"Task 4.2 PPO training reward: {split_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_metric_bar(results_df: pd.DataFrame, outpath: Path) -> None:
    if results_df.empty:
        return
    metric_cols = ["net_ann_return", "net_sharpe", "net_sortino", "net_max_drawdown"]
    plot_df = results_df[["split"] + metric_cols].copy().set_index("split")
    plt.figure(figsize=(10, 5.5))
    plot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Task 4.2 DNN+RL split metrics")
    plt.xlabel("Split")
    plt.ylabel("Metric value")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_summary_tables(results_df: pd.DataFrame, traininfo_df: pd.DataFrame, outdir: Path) -> None:
    tables_dir = outdir / "tables"
    _ensure_dir(tables_dir)

    if not results_df.empty:
        summary = results_df[[
            "net_ann_return", "net_ann_vol", "net_sharpe", "net_sortino",
            "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total"
        ]].agg(["mean", "std"]).T
        summary.columns = [f"portfolio_{c}" for c in summary.columns]
        summary.to_csv(tables_dir / "task4_2_portfolio_summary.csv")
        results_df.to_csv(tables_dir / "task4_2_split_portfolio_metrics.csv", index=False)

    if not traininfo_df.empty:
        traininfo_df.to_csv(tables_dir / "task4_2_training_info.csv", index=False)


def select_splits(meta_splits: List[Dict], pilot_mode: bool, pilot_recent_splits: int) -> List[Dict]:
    if not pilot_mode:
        return meta_splits
    n = int(max(1, pilot_recent_splits))
    return meta_splits[-n:]


def run_task4_2() -> None:
    if gym is None:
        raise SystemExit(
            "Gymnasium is required for Task 4.2.\n"
            "Install with: pip install gymnasium\n"
            f"Original error: {GYM_IMPORT_ERROR}"
        )
    if PPO is None:
        raise SystemExit(
            "Stable-Baselines3 is required for Task 4.2.\n"
            "Install with: pip install stable-baselines3\n"
            f"Original error: {SB3_IMPORT_ERROR}"
        )

    task2_outdir = Path(CONFIG["task2_outdir"]).resolve()
    task4_1_outdir = Path(CONFIG["task4_1_outdir"]).resolve()
    task4_2_outdir = Path(CONFIG["task4_2_outdir"]).resolve()
    _ensure_dir(task4_2_outdir)
    _ensure_dir(task4_2_outdir / "figures")
    _ensure_dir(task4_2_outdir / "equity_curves")
    _ensure_dir(task4_2_outdir / "rebalances")
    _ensure_dir(task4_2_outdir / "weights")
    _ensure_dir(task4_2_outdir / "training")
    _ensure_dir(task4_2_outdir / "models")

    with open(task4_2_outdir / "task4_2_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    np.random.seed(int(CONFIG["seed"]))

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
    if base_feature_cols is None:
        raise RuntimeError("feature_cols missing from walk_forward_meta.json")
    feature_cols = [c for c in base_feature_cols if c != target_col and c not in set(CONFIG["exclude_feature_cols"])]

    results_rows: List[Dict] = []
    train_rows: List[Dict] = []

    mode_label = "pilot" if bool(CONFIG["pilot_mode"]) else "full"
    print(f"Selected {len(selected_splits)} split(s) for Task 4.2 ({mode_label} mode).")

    for split in selected_splits:
        split_name = split["name"]
        train_path = Path(split["train_path"])
        test_path = Path(split["test_path"])
        if not train_path.is_absolute():
            train_path = (task2_outdir / train_path).resolve()
        if not test_path.is_absolute():
            test_path = (task2_outdir / test_path).resolve()

        print(f"\nRunning Task 4.2 on split: {split_name}")

        train_df = _read_panel(train_path)
        test_df = _read_panel(test_path)
        pred_df = _load_prediction_file(task4_1_outdir, split_name)

        train_prices = _panel_to_prices(train_df)
        test_prices = _panel_to_prices(test_df)
        common = [t for t in train_prices.columns if t in test_prices.columns]
        common = sorted(common)
        if not common:
            raise RuntimeError(f"No common tickers found for {split_name}")

        train_prices = train_prices[common].dropna(how="any")
        test_prices = test_prices[common].dropna(how="any")
        full_prices = pd.concat([train_prices, test_prices], axis=0)
        full_prices = full_prices[~full_prices.index.duplicated(keep="first")].sort_index()
        full_panel = pd.concat([train_df, test_df], axis=0).sort_index()

        train_start = pd.to_datetime(split["train_start"])
        train_end = pd.to_datetime(split["train_end"])
        test_start = pd.to_datetime(split["test_start"])
        test_end = pd.to_datetime(split["test_end"])

        train_records = build_rebalance_records(
            panel_window=full_panel.loc[(slice(train_start, train_end), slice(None)), :].copy(),
            pred_df=pred_df,
            prices_full=full_prices,
            start_date=train_start,
            end_date=train_end,
            feature_cols=feature_cols,
            tickers=common,
            rebalance_every_days=int(CONFIG["rebalance_every_days"]),
            signal_col=str(CONFIG["prediction_signal_col"]),
        )
        test_records = build_rebalance_records(
            panel_window=full_panel.loc[(slice(test_start, test_end), slice(None)), :].copy(),
            pred_df=pred_df,
            prices_full=full_prices,
            start_date=test_start,
            end_date=test_end,
            feature_cols=feature_cols,
            tickers=common,
            rebalance_every_days=int(CONFIG["rebalance_every_days"]),
            signal_col=str(CONFIG["prediction_signal_col"]),
        )

        if len(train_records) < 2:
            raise RuntimeError(f"Not enough train records for PPO on {split_name}")
        if len(test_records) < 1:
            raise RuntimeError(f"Not enough test records for evaluation on {split_name}")

        monitor_csv = task4_2_outdir / "training" / f"{split_name}_monitor.csv"

        def make_train_env() -> Monitor:
            env = PortfolioRLRebalanceEnv(
                records=train_records,
                tickers=common,
                max_weight=float(CONFIG["weight_bounds"][1]),
                transaction_cost_rate=float(CONFIG["transaction_cost_rate"]),
                downside_penalty=float(CONFIG["downside_penalty"]),
                reward_scale=float(CONFIG["reward_scale"]),
                use_prev_weights_in_obs=bool(CONFIG["use_prev_weights_in_obs"]),
            )
            return Monitor(env, filename=str(monitor_csv))

        vec_env = DummyVecEnv([make_train_env])
        model = PPO(
            policy=str(CONFIG["policy"]),
            env=vec_env,
            learning_rate=float(CONFIG["learning_rate"]),
            n_steps=int(CONFIG["n_steps"]),
            batch_size=int(CONFIG["batch_size"]),
            n_epochs=int(CONFIG["n_epochs"]),
            gamma=float(CONFIG["gamma"]),
            gae_lambda=float(CONFIG["gae_lambda"]),
            clip_range=float(CONFIG["clip_range"]),
            ent_coef=float(CONFIG["ent_coef"]),
            vf_coef=float(CONFIG["vf_coef"]),
            max_grad_norm=float(CONFIG["max_grad_norm"]),
            seed=int(CONFIG["seed"]),
            verbose=int(CONFIG["verbose_learn"]),
            device=str(CONFIG["device"]),
        )
        model.learn(total_timesteps=int(CONFIG["total_timesteps"]))

        if bool(CONFIG["save_models"]):
            model.save(task4_2_outdir / "models" / f"{split_name}_ppo")

        gross, net, turnover_total, tc_total, n_reb, reb_df = simulate_trained_policy(
            model=model,
            records=test_records,
            tickers=common,
            max_weight=float(CONFIG["weight_bounds"][1]),
            transaction_cost_rate=float(CONFIG["transaction_cost_rate"]),
            use_prev_weights_in_obs=bool(CONFIG["use_prev_weights_in_obs"]),
        )

        net_metrics = compute_metrics(net, int(CONFIG["frequency"]), float(CONFIG["risk_free_rate"]))

        eq_df = pd.DataFrame({
            "dnn_rl_gross": (1.0 + gross).cumprod(),
            "dnn_rl_net": (1.0 + net).cumprod(),
        })
        eq_df.to_csv(task4_2_outdir / "equity_curves" / f"{split_name}_equity.csv", index=True)
        reb_df.to_csv(task4_2_outdir / "rebalances" / f"{split_name}_dnn_rl_rebalances.csv", index=False)

        final_weights: Dict[str, float]
        if not reb_df.empty:
            last_row = reb_df.iloc[-1].to_dict()
            final_weights = {k.replace("w_", ""): float(v) for k, v in last_row.items() if k.startswith("w_")}
        else:
            final_weights = {}
        with open(task4_2_outdir / "weights" / f"{split_name}_dnn_rl_final.json", "w", encoding="utf-8") as f:
            json.dump(final_weights, f, indent=2)

        save_equity_plot(eq_df, task4_2_outdir / "figures" / f"{split_name}_equity.png", split_name)
        save_weights_plot(reb_df, task4_2_outdir / "figures" / f"{split_name}_weights.png", split_name)
        save_monitor_plot(monitor_csv, task4_2_outdir / "figures" / f"{split_name}_ppo_rewards.png", split_name)

        monitor_episodes = np.nan
        if monitor_csv.exists():
            try:
                monitor_df = pd.read_csv(monitor_csv, comment="#")
                monitor_episodes = int(len(monitor_df))
            except Exception:
                monitor_episodes = np.nan

        results_rows.append({
            "split": split_name,
            "model": "dnn_rl",
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

        train_rows.append({
            "split": split_name,
            "model": "dnn_rl",
            "n_train_records": int(len(train_records)),
            "n_test_records": int(len(test_records)),
            "n_assets": int(len(common)),
            "n_features_per_asset": int(len(feature_cols)),
            "ppo_total_timesteps": int(CONFIG["total_timesteps"]),
            "monitor_episodes": monitor_episodes,
            "reward_scale": float(CONFIG["reward_scale"]),
            "downside_penalty": float(CONFIG["downside_penalty"]),
            "transaction_cost_rate": float(CONFIG["transaction_cost_rate"]),
        })

        print(
            f"  DNN+RL: ann_return={net_metrics['ann_return']:.4f}, "
            f"sharpe={net_metrics['sharpe']:.3f}, sortino={net_metrics['sortino']:.3f}, "
            f"mdd={net_metrics['max_drawdown']:.3f}, n_reb={n_reb}"
        )

    results_df = pd.DataFrame(results_rows)
    traininfo_df = pd.DataFrame(train_rows)

    results_df.to_csv(task4_2_outdir / "task4_2_split_results.csv", index=False)
    traininfo_df.to_csv(task4_2_outdir / "task4_2_training_info.csv", index=False)

    save_summary_tables(results_df, traininfo_df, task4_2_outdir)
    save_metric_bar(results_df, task4_2_outdir / "figures" / "task4_2_split_metrics.png")

    if not results_df.empty:
        summary = (
            results_df
            .groupby("model")[[
                "net_ann_return", "net_ann_vol", "net_sharpe", "net_sortino",
                "net_cumulative_return", "net_max_drawdown", "turnover_total", "tc_total"
            ]]
            .agg(["mean", "std"])
        )
        summary.to_csv(task4_2_outdir / "task4_2_summary_by_model.csv")

    print("\nDone.")
    print(f"Saved split results:  {task4_2_outdir / 'task4_2_split_results.csv'}")
    print(f"Saved training info:  {task4_2_outdir / 'task4_2_training_info.csv'}")
    print(f"Saved figures in:     {task4_2_outdir / 'figures'}")
    print(f"Saved rebalances in:  {task4_2_outdir / 'rebalances'}")
    print(f"Saved weights in:     {task4_2_outdir / 'weights'}")


if __name__ == "__main__":
    run_task4_2()
