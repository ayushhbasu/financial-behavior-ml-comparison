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

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (TimeSeriesSplit, GridSearchCV,
                                     cross_val_score)
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection      import permutation_importance
# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.csv"
TARGET_PATH   = "data/processed/target.csv"
MODELS_DIR    = "models"
PLOT_PATH     = "outputs/tuning_report.png"

TRAIN_RATIO  = 0.75
N_CV_SPLITS  = 5
RANDOM_STATE = 42

REGIME_NAMES = {0: "Low", 1: "Medium", 2: "High"}
PALETTE      = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
BG_DARK      = "#0d1117"
BG_PANEL     = "#161b22"
SPINE_COL    = "#30363d"
TEXT_COL     = "white"

# ── Hyperparameter grids ──────────────────────────────────────────────────────
LR_GRID = {
    "clf__C"            : [0.01, 0.1, 0.5, 1.0, 5.0],
    "clf__solver"       : ["lbfgs", "saga"],
    "clf__class_weight" : ["balanced"],
}

RF_GRID = {
    "clf__n_estimators"    : [100, 200, 300],
    "clf__max_depth"       : [4, 6, 8, None],
    "clf__min_samples_leaf": [5, 10, 20],
}

# ── Load ──────────────────────────────────────────────────────────────────────

def load_data():
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"], index_col="Date")
    target   = pd.read_csv(TARGET_PATH,   parse_dates=["Date"], index_col="Date").squeeze()
    target.name = "regime"
    target_next = target.shift(-1).dropna().astype(int)
    common      = features.index.intersection(target_next.index)
    return features.loc[common], target_next.loc[common]


def load_baseline_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ts_split(X, y, ratio=TRAIN_RATIO):
    cut = int(len(X) * ratio)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# ── Base pipelines (for tuning) ───────────────────────────────────────────────

def base_lr_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

def base_rf_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)),
    ])



# ── GridSearchCV ──────────────────────────────────────────────────────────────

def tune_model(name, pipeline, param_grid, X_train, y_train):
    print(f"\n── Tuning {name} ──")
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv           = tscv,
        scoring      = "accuracy",
        n_jobs       = -1,
        verbose      = 0,
        refit        = True,
    )
    grid.fit(X_train, y_train)

    print(f"  Best params   : {grid.best_params_}")
    print(f"  Best CV acc   : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    tscv      = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=tscv, scoring="accuracy", n_jobs=-1)
    y_pred    = pipeline.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)

    print(f"\n  {name}")
    print(f"    CV  accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Low", "Medium", "High"], digits=3))

    return dict(pipeline=pipeline, cv_scores=cv_scores,
                y_pred=y_pred, y_test=y_test, accuracy=acc)
# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    tscv      = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=tscv, scoring="accuracy", n_jobs=-1)
    y_pred    = pipeline.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)

    print(f"\n  {name}")
    print(f"    CV  accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Low", "Medium", "High"], digits=3))

    return dict(pipeline=pipeline, cv_scores=cv_scores,
                y_pred=y_pred, y_test=y_test, accuracy=acc)


# ── Feature importance ────────────────────────────────────────────────────────

def compute_importances(rf_pipeline, X_test, y_test, feature_names):
    # Built-in impurity importance
    rf_clf   = rf_pipeline.named_steps["clf"]
    builtin  = pd.Series(rf_clf.feature_importances_, index=feature_names)

    # Permutation importance on test set (more reliable)
    X_scaled = rf_pipeline.named_steps["scaler"].transform(X_test)
    perm     = permutation_importance(
        rf_pipeline.named_steps["clf"],
        X_scaled, y_test,
        n_repeats    = 20,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    perm_mean = pd.Series(perm.importances_mean, index=feature_names)
    perm_std  = pd.Series(perm.importances_std,  index=feature_names)

    return builtin.sort_values(ascending=False), perm_mean, perm_std

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    tscv      = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=tscv, scoring="accuracy", n_jobs=-1)
    y_pred    = pipeline.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)

    print(f"\n  {name}")
    print(f"    CV  accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Low", "Medium", "High"], digits=3))

    return dict(pipeline=pipeline, cv_scores=cv_scores,
                y_pred=y_pred, y_test=y_test, accuracy=acc)


# ── Feature importance ────────────────────────────────────────────────────────

def compute_importances(rf_pipeline, X_test, y_test, feature_names):
    # Built-in impurity importance
    rf_clf   = rf_pipeline.named_steps["clf"]
    builtin  = pd.Series(rf_clf.feature_importances_, index=feature_names)

    # Permutation importance on test set (more reliable)
    X_scaled = rf_pipeline.named_steps["scaler"].transform(X_test)
    perm     = permutation_importance(
        rf_pipeline.named_steps["clf"],
        X_scaled, y_test,
        n_repeats    = 20,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    perm_mean = pd.Series(perm.importances_mean, index=feature_names)
    perm_std  = pd.Series(perm.importances_std,  index=feature_names)

    return builtin.sort_values(ascending=False), perm_mean, perm_std

# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(results, X_test, feature_names):
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS — where do models fail?")
    print("=" * 60)

    for name, res in results.items():
        errors  = X_test.copy()
        errors["y_true"] = res["y_test"].values
        errors["y_pred"] = res["y_pred"]
        errors["correct"]= errors["y_true"] == errors["y_pred"]

        wrong = errors[~errors["correct"]]
        print(f"\n  {name}  —  {len(wrong)} misclassifications")
        print("  Most confused pairs (true → predicted):")
        confused = (pd.DataFrame({"true": res["y_test"].values,
                                  "pred": res["y_pred"]})
                      .groupby(["true", "pred"])
                      .size()
                      .reset_index(name="count")
                      .query("true != pred")
                      .sort_values("count", ascending=False)
                      .head(4))
        for _, row in confused.iterrows():
            print(f"    {REGIME_NAMES[row['true']]:>8} → "
                  f"{REGIME_NAMES[row['pred']]:<8}  ({int(row['count'])} times)")

        print(f"  Mean vol_21 when wrong : "
              f"{wrong['vol_21'].mean():.4f}")
        print(f"  Mean vol_21 overall    : "
              f"{X_test['vol_21'].mean():.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(pipeline, path):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Saved → {path}")
# ── Plots ─────────────────────────────────────────────────────────────────────

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


def plot_tuning_report(baseline, tuned, X_test, y_test,
                       builtin_imp, perm_mean, perm_std, feature_names):
    os.makedirs("outputs", exist_ok=True)

    fig = plt.figure(figsize=(20, 24), facecolor=BG_DARK)
    fig.suptitle("Week 3 — Hyperparameter Tuning & Feature Analysis",
                 fontsize=18, fontweight="bold", color=TEXT_COL, y=0.985)
    gs = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.32)

    model_colors = {
        "LR Baseline" : "#4a90d9",
        "LR Tuned"    : "#58a6ff",
        "RF Baseline" : "#c0862a",
        "RF Tuned"    : "#f39c12",
    }
# ── A: Baseline vs tuned accuracy comparison ──────────────────────────────
    ax_a = style_ax(fig.add_subplot(gs[0, :]),
                    "Baseline vs Tuned — Test Accuracy & CV Accuracy")
    labels = list(model_colors.keys())
    test_accs = [
        baseline["Logistic Regression"]["accuracy"],
        tuned["Logistic Regression"]["accuracy"],
        baseline["Random Forest"]["accuracy"],
        tuned["Random Forest"]["accuracy"],
    ]
    cv_accs = [
        baseline["Logistic Regression"]["cv_scores"].mean(),
        tuned["Logistic Regression"]["cv_scores"].mean(),
        baseline["Random Forest"]["cv_scores"].mean(),
        tuned["Random Forest"]["cv_scores"].mean(),
    ]
    x     = np.arange(len(labels))
    width = 0.35
    bars1 = ax_a.bar(x - width/2, cv_accs,  width, label="CV Accuracy",
                     color=[model_colors[l] for l in labels],
                     alpha=0.6, edgecolor="white", lw=0.5)
    bars2 = ax_a.bar(x + width/2, test_accs, width, label="Test Accuracy",
                     color=[model_colors[l] for l in labels],
                     edgecolor="white", lw=0.5)
    for bar, val in zip(list(bars1) + list(bars2),
                        cv_accs + test_accs):
        ax_a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.005,
                  f"{val:.3f}", ha="center", va="bottom",
                  color=TEXT_COL, fontsize=9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, color=TEXT_COL)
    ax_a.set_ylabel("Accuracy")
    ax_a.set_ylim(0, 1.0)
    ax_a.axhline(1/3, color="white", ls="--", lw=1, alpha=0.5,
                 label="Random baseline (33%)")
    ax_a.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)
 # ── B: Confusion matrix — best tuned model ────────────────────────────────
    best_name = ("Logistic Regression"
                 if tuned["Logistic Regression"]["accuracy"]
                 >= tuned["Random Forest"]["accuracy"]
                 else "Random Forest")
    best_res  = tuned[best_name]

    ax_b = style_ax(fig.add_subplot(gs[1, 0]),
                    f"Confusion Matrix — Tuned {best_name}")
    cm = confusion_matrix(best_res["y_test"], best_res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Low", "Med", "High"]).plot(
        ax=ax_b, colorbar=False,
        cmap="Blues" if "Logistic" in best_name else "Oranges")
    ax_b.set_title(f"Confusion Matrix — Tuned {best_name}",
                   color=TEXT_COL, fontsize=11, pad=8, fontweight="semibold")
    for txt in ax_b.texts:
        txt.set_color(TEXT_COL)

    # ── C: CV scores — tuned models ───────────────────────────────────────────
    ax_c = style_ax(fig.add_subplot(gs[1, 1]),
                    "CV Accuracy per Fold — Tuned Models")
    for name, col in [("Logistic Regression", "#58a6ff"),
                      ("Random Forest",       "#f39c12")]:
        sc = tuned[name]["cv_scores"]
        ax_c.plot(range(1, len(sc)+1), sc, "o-",
                  color=col, lw=2, markersize=8, label=f"Tuned {name}")
        ax_c.axhline(sc.mean(), color=col, ls=":", lw=1.2, alpha=0.5)
    ax_c.set_xlabel("Fold")
    ax_c.set_ylabel("Accuracy")
    ax_c.set_xticks(range(1, N_CV_SPLITS+1))
    ax_c.set_ylim(0.3, 1.0)
    ax_c.legend(facecolor=BG_PANEL, labelcolor=TEXT_COL, fontsize=9)
 # ── D: Built-in feature importance ────────────────────────────────────────
    ax_d = style_ax(fig.add_subplot(gs[2, 0]),
                    "Feature Importance — RF (Impurity-based)")
    colors_fi = ["#f39c12" if "vol" in f else "#58a6ff"
                 for f in builtin_imp.index[::-1]]
    ax_d.barh(builtin_imp.index[::-1], builtin_imp.values[::-1],
              color=colors_fi, edgecolor="white", lw=0.3)
    ax_d.set_xlabel("Mean Decrease in Impurity")

    # ── E: Permutation importance ─────────────────────────────────────────────
    ax_e = style_ax(fig.add_subplot(gs[2, 1]),
                    "Permutation Importance — RF (Test Set)")
    order  = perm_mean.sort_values(ascending=True)
    colors_p = ["#f39c12" if "vol" in f else "#58a6ff" for f in order.index]
    ax_e.barh(order.index, order.values,
              xerr=perm_std.loc[order.index].values,
              color=colors_p, edgecolor="white", lw=0.3,
              error_kw=dict(ecolor="white", lw=1, capsize=3))
    ax_e.axvline(0, color="white", lw=0.8, ls="--")
    ax_e.set_xlabel("Mean Accuracy Decrease")

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {PLOT_PATH}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load
    X, y = load_data()
    X_train, X_test, y_train, y_test = ts_split(X, y)
    feature_names = X.columns.tolist()

    print("=" * 60)
    print("WEEK 3 — HYPERPARAMETER TUNING & FEATURE ANALYSIS")
    print("=" * 60)
    print(f"  Train : {len(X_train)} rows  |  Test : {len(X_test)} rows\n")

    # ── Load baseline models ──────────────────────────────────────────────────
    print("Loading baseline models...")
    bl_lr = load_baseline_model("models/lr_model.pkl")
    bl_rf = load_baseline_model("models/rf_model.pkl")

    baseline = {
        "Logistic Regression": evaluate("LR Baseline", bl_lr,
                                        X_train, X_test, y_train, y_test),
        "Random Forest"      : evaluate("RF Baseline", bl_rf,
                                        X_train, X_test, y_train, y_test),
    }
 # ── Tune models ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRID SEARCH — HYPERPARAMETER TUNING")
    print("=" * 60)

    tuned_lr_pipe = tune_model("Logistic Regression",
                               base_lr_pipeline(), LR_GRID,
                               X_train, y_train)
    tuned_rf_pipe = tune_model("Random Forest",
                               base_rf_pipeline(), RF_GRID,
                               X_train, y_train)

    print("\n" + "=" * 60)
    print("TUNED MODEL EVALUATION")
    print("=" * 60)

    tuned = {
        "Logistic Regression": evaluate("LR Tuned", tuned_lr_pipe,
                                        X_train, X_test, y_train, y_test),
        "Random Forest"      : evaluate("RF Tuned", tuned_rf_pipe,
                                        X_train, X_test, y_train, y_test),
    }
 # ── Improvement summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE vs TUNED — IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22} {'Baseline':>10} {'Tuned':>10} {'Delta':>8}")
    print("  " + "-" * 52)
    for name in ["Logistic Regression", "Random Forest"]:
        bl_acc = baseline[name]["accuracy"]
        tu_acc = tuned[name]["accuracy"]
        delta  = tu_acc - bl_acc
        arrow  = "▲" if delta >= 0 else "▼"
        print(f"  {name:<22} {bl_acc:>10.4f} {tu_acc:>10.4f} "
              f"{arrow}{abs(delta):>6.4f}")

    # ── Feature importance ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    builtin_imp, perm_mean, perm_std = compute_importances(
        tuned_rf_pipe, X_test, y_test, feature_names
    )
    print("\n  Impurity-based importance (RF):")
    for feat, imp in builtin_imp.items():
        print(f"    {feat:<20} {imp:.4f}")
    print("\n  Permutation importance (test set):")
    for feat in perm_mean.sort_values(ascending=False).index:
        print(f"    {feat:<20} {perm_mean[feat]:.4f} ± {perm_std[feat]:.4f}")

    # ── Error analysis ────────────────────────────────────────────────────────
    error_analysis(tuned, X_test, feature_names)

    # ── Model selection ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    best_name = ("Logistic Regression"
                 if tuned["Logistic Regression"]["accuracy"]
                 >= tuned["Random Forest"]["accuracy"]
                 else "Random Forest")
    print(f"  Selected model : {best_name}")
    print(f"  Test accuracy  : {tuned[best_name]['accuracy']:.4f}")
# ── Save tuned models ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING TUNED MODELS")
    print("=" * 60)
    save_model(tuned_lr_pipe, "models/lr_tuned.pkl")
    save_model(tuned_rf_pipe, "models/rf_tuned.pkl")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_tuning_report(baseline, tuned, X_test, y_test,
                       builtin_imp, perm_mean, perm_std, feature_names)
    return tuned, best_name


if __name__ == "__main__":
    main()
