import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

from sklearn.metrics import accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
PRICES_PATH   = "data/raw/price_data.csv"
FEATURES_PATH = "data/processed/features.csv"
TARGET_PATH   = "data/processed/target.csv"
LR_MODEL_PATH = "models/lr_tuned.pkl"
RF_MODEL_PATH = "models/rf_tuned.pkl"
PLOT_PATH     = "outputs/backtest_report.png"
RESULTS_PATH  = "data/processed/backtest_results.csv"

TRAIN_RATIO  = 0.75
RISK_FREE    = 0.0

# Allocation: regime → (SPY weight, IEF weight)
ALLOCATION = {
    0: (0.90, 0.10),   # Low vol  → risk-on
    1: (0.60, 0.40),   # Med vol  → balanced
    2: (0.20, 0.80),   # High vol → defensive
}

REGIME_NAMES = {0: "Low", 1: "Medium", 2: "High"}
BG_DARK      = "#0d1117"
BG_PANEL     = "#161b22"
SPINE_COL    = "#30363d"
TEXT_COL     = "white"
COLORS = {
    "ML Strategy" : "#f39c12",
    "100% SPY"    : "#58a6ff",
    "60/40"       : "#2ecc71",
}


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_all():
    print("Loading data...")
    prices   = pd.read_csv(PRICES_PATH,   parse_dates=["Date"], index_col="Date")
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"], index_col="Date")
    target   = pd.read_csv(TARGET_PATH,   parse_dates=["Date"], index_col="Date").squeeze()
    target.name = "regime"
    print(f"  Prices   : {prices.shape[0]} rows")
    print(f"  Features : {features.shape[0]} rows, {features.shape[1]} columns")
    print(f"  Target   : {target.shape[0]} rows")
    return prices, features, target


# ── 2. Load best model automatically ─────────────────────────────────────────

def load_best_model(X_train, X_test, y_train, y_test):
    """
    Load both tuned models, evaluate on test set,
    and automatically return the better one.
    """
    print("\nSelecting best model...")
    results = {}
    for name, path in [("Logistic Regression", LR_MODEL_PATH),
                       ("Random Forest",        RF_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        with open(path, "rb") as f:
            model = pickle.load(f)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {"model": model, "accuracy": acc}
        print(f"  {name:<26} test accuracy: {acc:.4f}")

    if not results:
        raise FileNotFoundError(
            "No tuned models found. Run hyperparameter_tuning.py first."
        )

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n  Selected: {best_name} (accuracy: {results[best_name]['accuracy']:.4f})")
    return results[best_name]["model"], best_name


# ── 3. Prepare test set ───────────────────────────────────────────────────────

def prepare_test_set(prices, features, target):
    """
    Align features and target, shift target +1 day,
    and return only the test portion (last 25%).
    """
    target_next = target.shift(-1).dropna().astype(int)
    common      = features.index.intersection(target_next.index)
    X           = features.loc[common]
    y           = target_next.loc[common]

    cut      = int(len(X) * TRAIN_RATIO)
    X_train  = X.iloc[:cut]
    X_test   = X.iloc[cut:]
    y_train  = y.iloc[:cut]
    y_test   = y.iloc[cut:]

    # Price data aligned to test dates
    price_test = prices.reindex(X_test.index).ffill()

    return X_train, X_test, y_train, y_test, price_test


# ── 4. Generate predictions ───────────────────────────────────────────────────

def get_predictions(model, X_test):
    preds = model.predict(X_test)
    return pd.Series(preds, index=X_test.index, name="predicted_regime")


# ── 5. Daily returns ──────────────────────────────────────────────────────────

def get_daily_returns(price_test):
    return price_test.pct_change().fillna(0)


# ── 6. Run strategies ─────────────────────────────────────────────────────────

def run_ml_strategy(pred_regimes, daily_ret):
    port = []
    for date in daily_ret.index:
        if date not in pred_regimes.index:
            port.append(0.0)
            continue
        regime       = pred_regimes.loc[date]
        w_spy, w_ief = ALLOCATION[regime]
        r            = w_spy * daily_ret.loc[date, "SPY"] + \
                       w_ief * daily_ret.loc[date, "IEF"]
        port.append(r)
    return pd.Series(port, index=daily_ret.index, name="ML Strategy")


def run_spy(daily_ret):
    return daily_ret["SPY"].rename("100% SPY")


def run_6040(daily_ret):
    return (0.60 * daily_ret["SPY"] + 0.40 * daily_ret["IEF"]).rename("60/40")


# ── 7. Performance metrics ────────────────────────────────────────────────────

def compute_metrics(returns, rf=RISK_FREE):
    r   = returns.dropna()
    n   = len(r)
    ann = 252

    total_ret = (1 + r).prod() - 1
    ann_ret   = (1 + total_ret) ** (ann / n) - 1
    ann_vol   = r.std() * np.sqrt(ann)
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    win_rate  = (r > 0).mean()

    cum      = (1 + r).cumprod()
    peak     = cum.cummax()
    max_dd   = ((cum - peak) / peak).min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Total Return"   : round(total_ret * 100, 2),
        "Ann. Return %"  : round(ann_ret   * 100, 2),
        "Ann. Vol %"     : round(ann_vol   * 100, 2),
        "Sharpe Ratio"   : round(sharpe,          3),
        "Max Drawdown %" : round(max_dd    * 100, 2),
        "Calmar Ratio"   : round(calmar,          3),
        "Win Rate %"     : round(win_rate  * 100, 2),
    }


def print_metrics(metrics_dict):
    print("\n" + "=" * 68)
    print("PERFORMANCE METRICS — TEST PERIOD")
    print("=" * 68)
    col_w    = 16
    strategies = list(metrics_dict.keys())
    header   = f"  {'Metric':<20}" + "".join(f"{s:>{col_w}}" for s in strategies)
    print(header)
    print("  " + "-" * (20 + col_w * len(strategies)))
    for metric in list(metrics_dict[strategies[0]].keys()):
        row = f"  {metric:<20}"
        for s in strategies:
            row += f"{str(metrics_dict[s][metric]):>{col_w}}"
        print(row)


# ── 8. Save results ───────────────────────────────────────────────────────────

def save_results(all_returns, metrics_dict):
    os.makedirs("data/processed", exist_ok=True)
    pd.concat(all_returns.values(), axis=1).to_csv(RESULTS_PATH)
    print(f"\n  Daily returns saved → {RESULTS_PATH}")

    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_path = "data/processed/metrics_summary.csv"
    metrics_df.to_csv(metrics_path)
    print(f"  Metrics summary  saved → {metrics_path}")


# ── 9. Plot ───────────────────────────────────────────────────────────────────

def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TEXT_COL, fontsize=11,
                 pad=8, fontweight="semibold")
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    return ax


def fmt_dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_backtest(all_returns, pred_regimes, metrics_dict):
    os.makedirs("outputs", exist_ok=True)

    fig = plt.figure(figsize=(20, 26), facecolor=BG_DARK)
    fig.suptitle(
        "Backtest Report — ML Regime Strategy vs Benchmarks (Test Period)",
        fontsize=18, fontweight="bold", color=TEXT_COL, y=0.985
    )
    gs = GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.30)

    # ── A: Cumulative returns ─────────────────────────────────────────────────
    ax_a = style_ax(fig.add_subplot(gs[0, :]),
                    "Cumulative Returns — ML Strategy vs Benchmarks")
    for name, ret in all_returns.items():
        cum = (1 + ret).cumprod()
        ax_a.plot(cum.index, cum.values,
                  color=COLORS[name], lw=2.2, label=name)
    ax_a.axhline(1.0, color="white", ls="--", lw=0.8, alpha=0.4)
    ax_a.set_ylabel("Portfolio Value (start = 1.0)")
    ax_a.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=10)
    fmt_dates(ax_a)

    # ── B: Drawdown ───────────────────────────────────────────────────────────
    ax_b = style_ax(fig.add_subplot(gs[1, :]),
                    "Drawdown — ML Strategy vs Benchmarks")
    for name, ret in all_returns.items():
        cum = (1 + ret).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax_b.fill_between(dd.index, dd.values, 0,
                          alpha=0.30, color=COLORS[name])
        ax_b.plot(dd.index, dd.values,
                  color=COLORS[name], lw=1.2, label=name)
    ax_b.set_ylabel("Drawdown")
    ax_b.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=10)
    fmt_dates(ax_b)

    # ── C: Rolling 21-day volatility ──────────────────────────────────────────
    ax_c = style_ax(fig.add_subplot(gs[2, 0]),
                    "Rolling 21-Day Volatility (Ann.)")
    for name, ret in all_returns.items():
        rv = ret.rolling(21).std() * np.sqrt(252)
        ax_c.plot(rv.index, rv.values,
                  color=COLORS[name], lw=1.5, label=name)
    ax_c.set_ylabel("Ann. Volatility")
    ax_c.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)
    fmt_dates(ax_c)

    # ── D: SPY allocation over time ───────────────────────────────────────────
    ax_d = style_ax(fig.add_subplot(gs[2, 1]),
                    "SPY Allocation Over Time (ML Strategy)")
    spy_alloc = pred_regimes.map(lambda r: ALLOCATION[r][0])
    ax_d.fill_between(spy_alloc.index, spy_alloc.values,
                      color="#58a6ff", alpha=0.65, step="post",
                      label="ML SPY weight")
    ax_d.axhline(0.60, color="#2ecc71", ls="--", lw=1.5,
                 label="60/40 benchmark (60%)")
    ax_d.axhline(0.90, color="#f39c12", ls=":", lw=1,
                 label="Max weight (90%)")
    ax_d.axhline(0.20, color="#e74c3c", ls=":", lw=1,
                 label="Min weight (20%)")
    ax_d.set_ylabel("SPY Weight")
    ax_d.set_ylim(0, 1.1)
    ax_d.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=8)
    fmt_dates(ax_d)

    # ── E: Key metrics bar chart ──────────────────────────────────────────────
    ax_e = style_ax(fig.add_subplot(gs[3, 0]),
                    "Annualised Return vs Volatility")
    strategies = list(metrics_dict.keys())
    x          = np.arange(len(strategies))
    width      = 0.35
    ret_vals   = [metrics_dict[s]["Ann. Return %"]  for s in strategies]
    vol_vals   = [metrics_dict[s]["Ann. Vol %"]     for s in strategies]
    b1 = ax_e.bar(x - width/2, ret_vals, width,
                  color=[COLORS[s] for s in strategies],
                  alpha=0.9, label="Ann. Return %", edgecolor="white", lw=0.5)
    b2 = ax_e.bar(x + width/2, vol_vals, width,
                  color=[COLORS[s] for s in strategies],
                  alpha=0.45, label="Ann. Vol %", edgecolor="white", lw=0.5)
    for bar, val in zip(list(b1) + list(b2), ret_vals + vol_vals):
        ax_e.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.3,
                  f"{val:.1f}%", ha="center", va="bottom",
                  color=TEXT_COL, fontsize=8)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(strategies, color=TEXT_COL, fontsize=9)
    ax_e.set_ylabel("%")
    ax_e.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)

    # ── F: Sharpe and Calmar ──────────────────────────────────────────────────
    ax_f = style_ax(fig.add_subplot(gs[3, 1]),
                    "Sharpe Ratio vs Calmar Ratio")
    sharpe_vals = [metrics_dict[s]["Sharpe Ratio"] for s in strategies]
    calmar_vals = [metrics_dict[s]["Calmar Ratio"] for s in strategies]
    b3 = ax_f.bar(x - width/2, sharpe_vals, width,
                  color=[COLORS[s] for s in strategies],
                  alpha=0.9, label="Sharpe Ratio", edgecolor="white", lw=0.5)
    b4 = ax_f.bar(x + width/2, calmar_vals, width,
                  color=[COLORS[s] for s in strategies],
                  alpha=0.45, label="Calmar Ratio", edgecolor="white", lw=0.5)
    for bar, val in zip(list(b3) + list(b4), sharpe_vals + calmar_vals):
        ax_f.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.02,
                  f"{val:.2f}", ha="center", va="bottom",
                  color=TEXT_COL, fontsize=8)
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(strategies, color=TEXT_COL, fontsize=9)
    ax_f.axhline(0, color="white", lw=0.5)
    ax_f.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {PLOT_PATH}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("WEEK 4 — BACKTESTING & STRATEGY IMPLEMENTATION")
    print("=" * 65)

    # Load data
    prices, features, target = load_all()

    # Prepare test set
    X_train, X_test, y_train, y_test, price_test = prepare_test_set(
        prices, features, target
    )
    print(f"\n  Train period : {X_train.index[0].date()} → {X_train.index[-1].date()} ({len(X_train)} rows)")
    print(f"  Test period  : {X_test.index[0].date()}  → {X_test.index[-1].date()}  ({len(X_test)} rows)")

    # Pick best model automatically
    model, model_name = load_best_model(X_train, X_test, y_train, y_test)

    # Predictions
    pred_regimes = get_predictions(model, X_test)
    print(f"\n  Predicted regime distribution:")
    for r, cnt in pred_regimes.value_counts().sort_index().items():
        w_spy, w_ief = ALLOCATION[r]
        print(f"    {REGIME_NAMES[r]:>6}  ({cnt:3d} days) "
              f"→ SPY {int(w_spy*100)}% / IEF {int(w_ief*100)}%")

    # Daily returns
    daily_ret = get_daily_returns(price_test)

    # Run strategies
    print("\n" + "=" * 65)
    print("RUNNING STRATEGIES")
    print("=" * 65)
    all_returns = {
        "ML Strategy" : run_ml_strategy(pred_regimes, daily_ret),
        "100% SPY"    : run_spy(daily_ret),
        "60/40"       : run_6040(daily_ret),
    }

    # Metrics
    metrics_dict = {name: compute_metrics(r) for name, r in all_returns.items()}
    print_metrics(metrics_dict)

    # Save
    print("\n" + "=" * 65)
    print("SAVING RESULTS")
    print("=" * 65)
    save_results(all_returns, metrics_dict)

    # Plot
    print("\n" + "=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)
    plot_backtest(all_returns, pred_regimes, metrics_dict)

    print("\nBacktest complete!")
    return all_returns, metrics_dict


if __name__ == "__main__":
    main()
