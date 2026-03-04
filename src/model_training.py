import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.csv"
TARGET_PATH   = "data/processed/target.csv"
MODELS_DIR    = "models"
OUTPUTS_DIR   = "outputs"
PREDS_PATH    = "data/processed/predictions.csv"
PLOT_PATH     = "outputs/model_evaluation.png"

TRAIN_RATIO   = 0.75
N_CV_SPLITS   = 5
RANDOM_STATE  = 42

REGIME_NAMES  = {0: "Low", 1: "Medium", 2: "High"}
PALETTE       = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
BG_DARK       = "#0d1117"
BG_PANEL      = "#161b22"
SPINE_COL     = "#30363d"
TEXT_COL      = "white"


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data():
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"], index_col="Date")
    target   = pd.read_csv(TARGET_PATH,   parse_dates=["Date"], index_col="Date").squeeze()
    target.name = "regime"
    return features, target
# ── 2. Align & shift ──────────────────────────────────────────────────────────

def align_and_shift(features, target):
    """
    Shift target +1 day: today's features predict tomorrow's regime.
    This is the correct framing for a real trading system.
    """
    target_next = target.shift(-1).dropna().astype(int)
    common      = features.index.intersection(target_next.index)
    return features.loc[common], target_next.loc[common]


# ── 3. Train / test split ─────────────────────────────────────────────────────

def ts_split(X, y, ratio=TRAIN_RATIO):
    cut = int(len(X) * ratio)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

# ── 4. Models ─────────────────────────────────────────────────────────────────

def build_models():
    """
    Both models wrapped in a Pipeline with StandardScaler.
    Scaler is fitted on train set only inside cross_val_score — no leakage.
    """
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    return {"Logistic Regression": lr, "Random Forest": rf}

# ── 5 & 6. Train & evaluate ───────────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    tscv    = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    results = {}

    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, pipe in models.items():
        print(f"\n── {name} ──")

        # Cross-validation (temporal order preserved)
        cv_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=tscv, scoring="accuracy", n_jobs=-1
        )
        print(f"  CV  accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        print(f"  Test accuracy: {acc:.4f}\n")
        print(classification_report(
            y_test, y_pred,
            target_names=["Low", "Medium", "High"],
            digits=3,
        ))

        results[name] = dict(
            pipeline  = pipe,
            cv_scores = cv_scores,
            y_pred    = y_pred,
            y_test    = y_test,
            accuracy  = acc,
        )

    return results


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(rf_pipeline, feature_names):
    imp = rf_pipeline.named_steps["clf"].feature_importances_
    return (pd.DataFrame({"feature": feature_names, "importance": imp})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))

# ── 7. Summary ────────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<26} {'CV Acc':>8} {'CV Std':>8} {'Test Acc':>10}")
    print("  " + "-" * 55)
    for name, res in results.items():
        mu  = res["cv_scores"].mean()
        sig = res["cv_scores"].std()
        print(f"  {name:<26} {mu:>8.4f} {sig:>8.4f} {res['accuracy']:>10.4f}")


# ── 8. Save artefacts ─────────────────────────────────────────────────────────

def save_artefacts(results, X_test):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING ARTEFACTS")
    print("=" * 60)

    for name, res in results.items():
        fname = "lr_model.pkl" if "Logistic" in name else "rf_model.pkl"
        path  = os.path.join(MODELS_DIR, fname)
        with open(path, "wb") as f:
            pickle.dump(res["pipeline"], f)
        print(f"  Saved → {path}")

    # Predictions CSV
    rows = []
    for name, res in results.items():
        for date, true_v, pred_v in zip(X_test.index,
                                        res["y_test"], res["y_pred"]):
            rows.append({"Date": date, "model": name,
                         "y_true": true_v, "y_pred": pred_v})
    (pd.DataFrame(rows)
       .set_index("Date")
       .to_csv(PREDS_PATH))
    print(f"  Predictions → {PREDS_PATH}")
    # ── 9. Plots ──────────────────────────────────────────────────────────────────

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


def plot_evaluation(results, X_test, feature_names):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    lr_res = results["Logistic Regression"]
    rf_res = results["Random Forest"]

    fig = plt.figure(figsize=(20, 22), facecolor=BG_DARK)
    fig.suptitle("Model Evaluation Report — Logistic Regression vs Random Forest",
                 fontsize=18, fontweight="bold", color=TEXT_COL, y=0.985)
    gs = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    # ── A: Confusion matrix — LR ──────────────────────────────────────────────
    ax_a = style_ax(fig.add_subplot(gs[0, 0]),
                    "Confusion Matrix — Logistic Regression")
    cm_lr = confusion_matrix(lr_res["y_test"], lr_res["y_pred"])
    ConfusionMatrixDisplay(cm_lr, display_labels=["Low", "Med", "High"]).plot(
        ax=ax_a, colorbar=False, cmap="Blues")
    ax_a.set_title("Confusion Matrix — Logistic Regression",
                   color=TEXT_COL, fontsize=11, pad=8, fontweight="semibold")
    for txt in ax_a.texts:
        txt.set_color(TEXT_COL)

    # ── B: Confusion matrix — RF ──────────────────────────────────────────────
    ax_b = style_ax(fig.add_subplot(gs[0, 1]),
                    "Confusion Matrix — Random Forest")
    cm_rf = confusion_matrix(rf_res["y_test"], rf_res["y_pred"])
    ConfusionMatrixDisplay(cm_rf, display_labels=["Low", "Med", "High"]).plot(
        ax=ax_b, colorbar=False, cmap="Oranges")
    ax_b.set_title("Confusion Matrix — Random Forest",
                   color=TEXT_COL, fontsize=11, pad=8, fontweight="semibold")
    for txt in ax_b.texts:
        txt.set_color(TEXT_COL)

    # ── C: CV accuracy per fold ───────────────────────────────────────────────
    ax_c = style_ax(fig.add_subplot(gs[1, 0]),
                    "Cross-Validation Accuracy per Fold (TimeSeriesSplit)")
    model_colors = {"Logistic Regression": "#58a6ff", "Random Forest": "#f39c12"}
    for name, col in model_colors.items():
        sc = results[name]["cv_scores"]
        ax_c.plot(range(1, len(sc) + 1), sc, "o-",
                  color=col, lw=2, markersize=8, label=name)
        ax_c.axhline(sc.mean(), color=col, ls=":", lw=1.2, alpha=0.6)
    ax_c.set_xlabel("Fold")
    ax_c.set_ylabel("Accuracy")
    ax_c.set_xticks(range(1, N_CV_SPLITS + 1))
    ax_c.set_ylim(0.3, 1.0)
    ax_c.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)
# ── D: Test accuracy comparison bar ──────────────────────────────────────
    ax_d = style_ax(fig.add_subplot(gs[1, 1]),
                    "Test Set Accuracy Comparison")
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    bars  = ax_d.bar(names, accs,
                     color=[model_colors[n] for n in names],
                     edgecolor="white", lw=0.5, width=0.4)
    for bar, val in zip(bars, accs):
        ax_d.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.005,
                  f"{val:.4f}",
                  ha="center", va="bottom", color=TEXT_COL, fontsize=11,
                  fontweight="bold")
    ax_d.set_ylabel("Accuracy")
    ax_d.set_ylim(0, 1.0)
    ax_d.axhline(1/3, color="white", ls="--", lw=1,
                 label="Random baseline (33%)")
    ax_d.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)

    # ── E: Feature importance — RF ────────────────────────────────────────────
    ax_e = style_ax(fig.add_subplot(gs[2, 0]),
                    "Feature Importances — Random Forest")
    fi = get_feature_importance(rf_res["pipeline"], feature_names)
    colors_fi = ["#f39c12" if "vol" in f else "#58a6ff" for f in fi["feature"][::-1]]
    ax_e.barh(fi["feature"][::-1], fi["importance"][::-1],
              color=colors_fi, edgecolor="white", lw=0.3)
    ax_e.set_xlabel("Importance")

    # ── F: Predicted probabilities — RF ──────────────────────────────────────
    ax_f = style_ax(fig.add_subplot(gs[2, 1]),
                    "Predicted Class Probabilities — Random Forest (Test Set)")
    rf_proba = rf_res["pipeline"].predict_proba(X_test)
    proba_df = pd.DataFrame(rf_proba, index=X_test.index,
                            columns=["Low Vol", "Med Vol", "High Vol"])
    ax_f.stackplot(proba_df.index,
                   proba_df["Low Vol"], proba_df["Med Vol"], proba_df["High Vol"],
                   labels=["Low Vol", "Med Vol", "High Vol"],
                   colors=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.80)
    ax_f.set_ylabel("Probability")
    ax_f.set_ylim(0, 1)
    ax_f.legend(loc="upper left", facecolor=BG_PANEL,
                labelcolor=TEXT_COL, fontsize=9)
    import matplotlib.dates as mdates
    ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax_f.get_xticklabels(), rotation=30, ha="right")

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {PLOT_PATH}")
    plt.close()
# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load
    features, target = load_data()

    # Align & shift
    X, y = align_and_shift(features, target)
    print(f"Dataset        : {len(X)} rows, {X.shape[1]} features")
    print(f"Date range     : {X.index[0].date()} → {X.index[-1].date()}")
    print(f"Regime counts  :")
    for k, v in y.value_counts().sort_index().items():
        print(f"  {REGIME_NAMES[k]:>8}: {v:4d}  ({100*v/len(y):.1f}%)")
    print()

    # Split
    X_train, X_test, y_train, y_test = ts_split(X, y)
    print(f"Train : {len(X_train)} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"Test  : {len(X_test)}  rows  ({X_test.index[0].date()} → {X_test.index[-1].date()})\n")

    # Train & evaluate
    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # Summary
    print_summary(results)

    # Save
    save_artefacts(results, X_test)

    # Plot
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_evaluation(results, X_test, X.columns.tolist())

    print("\nWeek 2 complete.")
    return results


if __name__ == "__main__":
    main()
