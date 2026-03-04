import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.csv"
TARGET_PATH   = "data/processed/target.csv"
PRICES_PATH   = "data/raw/price_data.csv"
OUTPUT_PATH   = "outputs/eda_report.png"

REGIME_NAMES  = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
PALETTE       = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
BG_DARK       = "#0d1117"
BG_PANEL      = "#161b22"
SPINE_COL     = "#30363d"
TEXT_COL      = "white"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all():
    prices   = pd.read_csv(PRICES_PATH,   parse_dates=["Date"], index_col="Date")
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"], index_col="Date")
    target   = pd.read_csv(TARGET_PATH,   parse_dates=["Date"], index_col="Date").squeeze()
    target.name = "regime"

    # Align all on common dates
    common   = features.index.intersection(target.index)
    features = features.loc[common]
    target   = target.loc[common]
    return prices, features, target


def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TEXT_COL, fontsize=11,
                 pad=8, fontweight="semibold")
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    return ax


def fmt_dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def regime_legend(ax):
    patches = [mpatches.Patch(color=PALETTE[k], label=REGIME_NAMES[k])
               for k in sorted(PALETTE)]
    ax.legend(handles=patches, facecolor=BG_PANEL,
              labelcolor=TEXT_COL, fontsize=8, loc="upper left")

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_eda(prices, features, target, out_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(20, 28), facecolor=BG_DARK)
    fig.suptitle("Exploratory Data Analysis — SPY / IEF Market Regime Pipeline",
                 fontsize=20, fontweight="bold", color=TEXT_COL, y=0.985)

    gs = GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.30)

    colors_reg = [PALETTE[r] for r in target.values]

    # ── 1. SPY & IEF price history ────────────────────────────────────────────
    ax1 = style_ax(fig.add_subplot(gs[0, :]),
                   "SPY & IEF Adjusted Close Price (2022–2024)")
    ax1b = ax1.twinx()
    ax1.plot(prices.index,  prices["SPY"], color="#58a6ff", lw=1.5, label="SPY")
    ax1b.plot(prices.index, prices["IEF"], color="#f39c12", lw=1.5,
              linestyle="--", label="IEF")
    ax1.set_ylabel("SPY (USD)", color="#58a6ff")
    ax1b.set_ylabel("IEF (USD)", color="#f39c12")
    ax1b.tick_params(colors=TEXT_COL, labelsize=8)
    ax1b.yaxis.label.set_color(TEXT_COL)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2,
               facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)
    fmt_dates(ax1)

    # ── 2. SPY daily returns ──────────────────────────────────────────────────
    ax2 = style_ax(fig.add_subplot(gs[1, 0]),
                   "SPY Daily Returns")
    ret = features["r1"]
    ax2.bar(ret.index, ret.values,
            color=["#e74c3c" if r < 0 else "#2ecc71" for r in ret.values],
            width=1, alpha=0.8)
    ax2.axhline(0, color="white", lw=0.5)
    ax2.set_ylabel("Return")
    fmt_dates(ax2)
    # ── 3. Realized vol + regime shading ─────────────────────────────────────
    ax3 = style_ax(fig.add_subplot(gs[1, 1]),
                   "21-Day Realized Volatility (Ann.) with Regimes")
    vol = features["vol_21"]
    ax3.plot(vol.index, vol.values, color="white", lw=1, zorder=3)
    for rid, col in PALETTE.items():
        mask = target == rid
        ax3.fill_between(vol.index,
                         vol.min() * 0.95, vol.max() * 1.05,
                         where=mask.reindex(vol.index).fillna(False).values,
                         alpha=0.25, color=col, label=REGIME_NAMES[rid])
    ax3.set_ylabel("Ann. Vol")
    ax3.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=8)
    fmt_dates(ax3)

    # ── 4. Regime distribution ────────────────────────────────────────────────
    ax4 = style_ax(fig.add_subplot(gs[2, 0]),
                   "Regime Distribution")
    counts = target.value_counts().sort_index()
    total  = len(target)
    bars   = ax4.bar([REGIME_NAMES[i] for i in counts.index],
                     counts.values,
                     color=[PALETTE[i] for i in counts.index],
                     edgecolor="white", lw=0.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{val}\n({100*val/total:.1f}%)",
                 ha="center", va="bottom", color=TEXT_COL, fontsize=9)
    ax4.set_ylabel("Trading Days")
    ax4.set_ylim(0, counts.max() * 1.2)

    # ── 5. Regime timeline ────────────────────────────────────────────────────
    ax5 = style_ax(fig.add_subplot(gs[2, 1]),
                   "Regime Timeline (2022–2024)")
    for rid, col in PALETTE.items():
        mask = (target == rid)
        ax5.fill_between(target.index, rid, rid + 0.8,
                         where=mask.values,
                         color=col, alpha=0.85, step="post")
    ax5.set_yticks([0.4, 1.4, 2.4])
    ax5.set_yticklabels(["Low", "Medium", "High"], color=TEXT_COL, fontsize=9)
    ax5.set_ylabel("Regime")
    fmt_dates(ax5)

    # ── 6. Feature correlation heatmap ────────────────────────────────────────
    ax6 = style_ax(fig.add_subplot(gs[3, 0]),
                   "Feature Correlation Heatmap")
    corr = features.corr()
    im   = ax6.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ticks = range(len(corr.columns))
    ax6.set_xticks(ticks); ax6.set_yticks(ticks)
    ax6.set_xticklabels(corr.columns, rotation=45, ha="right",
                        color=TEXT_COL, fontsize=8)
    ax6.set_yticklabels(corr.columns, color=TEXT_COL, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax6.text(j, i, f"{corr.values[i, j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COL)
# ── 7. vol_21 distribution per regime ─────────────────────────────────────
    ax7 = style_ax(fig.add_subplot(gs[3, 1]),
                   "vol_21 Distribution by Regime")
    for rid, col in PALETTE.items():
        subset = features.loc[target == rid, "vol_21"]
        ax7.hist(subset, bins=30, color=col, alpha=0.6,
                 label=REGIME_NAMES[rid], edgecolor="none")
    ax7.set_xlabel("21-Day Realized Vol (Ann.)")
    ax7.set_ylabel("Frequency")
    ax7.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"EDA report saved → {out_path}")
    plt.close()


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(prices, features, target):
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  Price data   : {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} rows)")
    print(f"  Features     : {features.shape[1]} columns, {len(features)} rows")
    print(f"  Target rows  : {len(target)}")
    print()

    print("  Regime distribution:")
    for rid, cnt in target.value_counts().sort_index().items():
        print(f"    {REGIME_NAMES[rid]:>10}: {cnt:4d}  ({100*cnt/len(target):.1f}%)")
    print()

    print("  Feature statistics:")
    print(features.describe().round(4).to_string())
    print()

    print("  Feature means per regime:")
    df_combined = features.copy()
    df_combined["regime"] = target
    print(df_combined.groupby("regime")[features.columns].mean().round(4).to_string())
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices, features, target = load_all()
    print_summary(prices, features, target)
    plot_eda(prices, features, target)
    print("\nEDA complete.")
