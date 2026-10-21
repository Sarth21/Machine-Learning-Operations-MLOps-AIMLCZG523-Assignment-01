"""Matplotlib / seaborn figures for EDA and model evaluation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve  # noqa: E402

from heartml import config as C  # noqa: E402

PALETTE = {0: "#4C78A8", 1: "#E4572E"}
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def class_balance(df, path):
    counts = df[C.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values, color=[PALETTE[0], PALETTE[1]])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    return _save(fig, path)


def missing_values(df, path):
    miss = df[C.FEATURES].isna().sum()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(miss.index, miss.values, color="#7F7F7F")
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values per feature (after cleaning)")
    for i, v in enumerate(miss.values):
        ax.text(v + 0.05, i, str(int(v)), va="center", fontsize=8)
    ax.invert_yaxis()
    return _save(fig, path)


def numeric_histograms(df, path):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), C.NUMERIC_FEATURES):
        for cls in (0, 1):
            sns.histplot(df.loc[df[C.TARGET] == cls, col].dropna(), bins=18, kde=(col != "ca"),
                         color=PALETTE[cls], alpha=0.45, ax=ax, stat="density",
                         label="Disease" if cls else "No disease")
        ax.set_title(col)
        ax.set_xlabel("")
    axes[0, 0].legend()
    fig.suptitle("Numeric feature distributions by class", y=1.01, fontsize=13)
    return _save(fig, path)


def categorical_counts(df, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for ax, col in zip(axes.ravel(), C.CATEGORICAL_FEATURES):
        sns.countplot(data=df.assign(**{col: df[col].astype("Int64")}), x=col, hue=C.TARGET,
                      palette=PALETTE, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.get_legend().remove()
    last = axes.ravel()[-1]
    last.axis("off")
    last.legend([plt.Rectangle((0, 0), 1, 1, color=PALETTE[0]), plt.Rectangle((0, 0), 1, 1, color=PALETTE[1])],
                ["No disease", "Disease"], loc="center", fontsize=11)
    fig.suptitle("Categorical features by class", y=1.01, fontsize=13)
    return _save(fig, path)


def boxplots(df, path):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8))
    for ax, col in zip(axes, ["age", "trestbps", "chol", "thalach", "oldpeak"]):
        sns.boxplot(data=df, x=C.TARGET, y=col, hue=C.TARGET, palette=PALETTE, ax=ax, legend=False)
        ax.set_title(col)
    fig.suptitle("Outlier check: numeric features vs target", y=1.03, fontsize=13)
    return _save(fig, path)


def correlation_heatmap(df, path):
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.4))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 7.5})
    ax.set_title("Correlation heatmap (Pearson)")
    return _save(fig, path)


def target_correlation(df, path):
    corr = df.corr(numeric_only=True)[C.TARGET].drop(C.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.barh(corr.index, corr.values, color=[PALETTE[1] if v > 0 else PALETTE[0] for v in corr.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Correlation of each feature with the target")
    ax.set_xlabel("Pearson r")
    return _save(fig, path)


def roc_curves(curves, path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - hold-out test set")
    ax.legend(loc="lower right", fontsize=8.5)
    return _save(fig, path)


def pr_curves(curves, path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, lw=2, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - hold-out test set")
    ax.legend(loc="lower left", fontsize=8.5)
    return _save(fig, path)


def confusion(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["No disease", "Disease"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    ax.grid(False)
    return _save(fig, path)


def model_comparison(cv_table, path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = list(cv_table)
    x = np.arange(len(metrics))
    w = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for i, n in enumerate(names):
        ax.bar(x + i * w - 0.4 + w / 2, [cv_table[n][m][0] for m in metrics], w,
               yerr=[cv_table[n][m][1] for m in metrics], capsize=3, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    lo = min(cv_table[n][m][0] for n in names for m in metrics)
    ax.set_ylim(max(0, lo - 0.12), 1.02)
    ax.set_title("Repeated stratified 5-fold CV (mean +/- std), tuned models")
    ax.legend(fontsize=8.5, loc="lower right")
    return _save(fig, path)


def importance(names, values, title, path, top=15):
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#2F6690")
    ax.set_title(title)
    ax.set_xlabel("Mean decrease in ROC-AUC when shuffled")
    return _save(fig, path)


def calibration(y_true, y_score, title, path):
    frac, mean_pred = calibration_curve(y_true, y_score, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(mean_pred, frac, "o-", color="#E4572E", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)
