# Explainable and Adaptive AI for Portfolio Optimisation

A final year project that compares adaptive and explainable AI methods with strong classical portfolio construction baselines under a leakage-safe walk-forward evaluation framework.

## Overview

This project implements a modular research pipeline for portfolio optimisation using daily market data for four exchange-traded funds: **SPY, IWM, EFA, and TLT**. The workflow combines:

- leakage-safe data preparation and walk-forward splitting
- classical baselines: **Equal-weight**, **Markowitz**, and **Black-Litterman**
- a supervised **DNN** predictor for next-period log returns
- a modular **DNN + PPO RL** allocation stage (**Pipeline A**)
- **SHAP** and **LIME** explainability for the DNN stage
- feature-family ablation analysis
- a **Streamlit** demo app for browsing saved artefacts and validated results

The project is designed as an **offline academic research prototype**, not a live trading or financial advice system.

## Key Results

Evaluation was performed on daily data for four ETFs (**SPY, IWM, EFA, TLT**) using a leakage-safe walk-forward setup with **34 common splits**.

Main findings:

- **Markowitz** was the strongest aggregate performer on the main risk-adjusted metrics.
- **Black-Litterman** also remained highly competitive across the common-split evaluation.
- **Equal-weight** was the most cost-efficient approach because it produced the lowest turnover and transaction cost totals.
- The **DNN-only** stage was sometimes competitive on return, but it also produced the highest turnover and transaction costs.
- The **DNN + RL** pipeline produced a functioning adaptive allocation system and improved trading efficiency relative to DNN-only, but it did **not** outperform the strongest classical benchmark on average risk-adjusted performance.
- On selected windows, the adaptive pipeline could be competitive or even strongest on raw return, but it was not consistently strongest on Sharpe, Sortino, or drawdown.
- **SHAP** analysis suggested that the DNN forecasting stage relied mainly on medium-horizon return, volatility, momentum, and MACD-related features.

Overall, the project contributes a disciplined comparative framework for evaluating adaptive and explainable portfolio optimisation rather than claiming a decisive AI performance advantage over strong classical baselines.

## Pipeline

The code is organised as a staged workflow. Each stage writes outputs that are used by later stages.

1. **Task 2 - Data preparation**  
   Downloads historical data, cleans and aligns it, engineers features, creates leakage-safe walk-forward splits, applies train-only scaling, and saves report-friendly tables and figures.

2. **Task 3 - Classical baselines**  
   Evaluates Equal-weight, Markowitz, and Black-Litterman using the same shared constraints and transaction-cost assumptions.

3. **Task 4.1 - DNN-only**  
   Trains a supervised DNN per split, produces forecasts, converts signals into portfolio weights, and saves diagnostics and model artefacts.

4. **Task 4.2 - DNN + RL**  
   Loads cached DNN signals from Task 4.1, builds a custom Gymnasium environment, trains a PPO agent on the training window only, and evaluates out of sample.

5. **Task 5 - Comparative evaluation**  
   Combines Task 3, Task 4.1, and Task 4.2 results into common-split comparison tables, figures, delta summaries, and equity comparisons.

6. **Task 6.1 - SHAP explainability**  
   Explains the exact saved DNN models from Task 4.1 and exports global, local, and plain-English explanation artefacts.

7. **Task 6.2 - LIME explainability**  
   Runs a focused local case-study explanation for a selected saved DNN split.

8. **Task 6.3 - Feature ablation**  
   Removes feature families from the DNN-only stage and measures the effect on forecasting and portfolio performance.

9. **Demo app**  
   Loads the saved artefacts from the stages above and presents them in a structured Streamlit interface.

## Repository files

Core scripts:

- `task2_data_preparation.py`
- `task3_classical_baselines.py`
- `task4_1_dnn_only.py`
- `task4_2_dnn_rl.py`
- `task5_comparative_evaluation.py`
- `task6_1_shap_explainability.py`
- `task6_2_lime_explainability.py`
- `task6_3_feature_ablation.py`
- `demo_app.py`

Supporting files:

- `requirements.txt`
- `README.md`
- `.gitignore`
- `LICENSE`

## Environment setup

Recommended Python version: **3.10**.

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit pyarrow
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit pyarrow
```

### Dependency notes

- `requirements.txt` contains the core project dependencies.
- `streamlit` is needed for the demo app.
- `pyarrow` is recommended because Task 2 prefers Parquet output, although it will fall back to CSV if Parquet support is unavailable.

## How to run

Each script uses a local `CONFIG` dictionary near the top of the file. Edit those values if you want to change paths, dates, tickers, hyperparameters, or runtime scope.

Run the pipeline in this order:

```bash
python task2_data_preparation.py
python task3_classical_baselines.py
python task4_1_dnn_only.py
python task4_2_dnn_rl.py
python task5_comparative_evaluation.py
python task6_1_shap_explainability.py
python task6_2_lime_explainability.py
python task6_3_feature_ablation.py
```

Then launch the demo app:

```bash
streamlit run demo_app.py
```

## Output folders

By default, the pipeline writes to the following folders in the project root:

- `data_prepared_task2`
- `results_task3_classical`
- `results_task4_1_dnn_only`
- `results_task4_2_dnn_rl`
- `results_task5_comparative_evaluation`
- `results_task6_1_shap_explainability`
- `results_task6_2_lime_explainability`
- `results_task6_3_feature_ablation`

Typical saved artefacts include:

- split-level train/test panels and scaling metadata
- equity curves, rebalance logs, and final weights
- split-level and aggregate performance summaries
- DNN diagnostics and saved models
- PPO training information
- SHAP and LIME figures, tables, and plain-English summaries
- feature ablation deltas and summary tables
- comparative evaluation tables and equity comparison plots

## Demo app

The Streamlit app is a lightweight evidence browser for the completed project. It reads saved artefacts directly from the output folders above and provides five main views:

1. Overview  
2. Comparative results  
3. Split deep dive  
4. Explainability  
5. Feature ablation

The app is intentionally designed around **precomputed results** rather than live retraining.

## Practical notes

- Run **Task 2** first. All later stages depend on the saved walk-forward metadata and split files.
- **Task 4.2** depends on both **Task 2** and **Task 4.1** outputs.
- **Task 6.1** and **Task 6.2** rely on the saved DNN artefacts from **Task 4.1**.
- **Task 5** assumes that Task 3, Task 4.1, and Task 4.2 have all produced valid results.
- If you want a clean rerun, delete or archive old result folders before starting again, or change the configured output directories.

## What the project demonstrates

This repository is intended to show a full, reproducible comparative workflow for evaluating adaptive portfolio optimisation under realistic academic constraints, including:

- strict temporal separation between training and testing
- transaction costs and turnover-aware evaluation
- comparison against strong classical baselines
- modular prediction and allocation stages
- post-hoc explainability for the supervised forecasting component

## Limitations

- This is an **offline academic research prototype**, not a live trading system.
- The core evaluation uses **four daily ETFs**, so the asset universe is intentionally narrow.
- Results are based on **historical backtesting** under a constrained walk-forward setting and do not guarantee live performance.
- Explainability is strongest for the **supervised DNN forecasting stage** rather than the PPO policy itself.
- The repository is intended to support research, reproducibility, and demonstration, not financial decision-making in practice.

## Disclaimer

This repository was created for academic research and demonstration. It is **not** financial advice, a production trading system, or a recommendation engine.
