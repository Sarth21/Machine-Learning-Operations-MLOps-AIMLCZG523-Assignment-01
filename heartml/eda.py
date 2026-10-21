"""Exploratory data analysis: figures + machine-readable summary."""
import json
import logging

from heartml import config as C
from heartml import data as D
from heartml import plots as P

log = logging.getLogger(__name__)


def run_eda(df=None, fig_dir=None):
    df = D.load_clean() if df is None else df
    fig_dir = fig_dir or C.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "class_balance": P.class_balance(df, fig_dir / "eda_class_balance.png"),
        "missing_values": P.missing_values(df, fig_dir / "eda_missing_values.png"),
        "numeric_histograms": P.numeric_histograms(df, fig_dir / "eda_numeric_histograms.png"),
        "categorical_counts": P.categorical_counts(df, fig_dir / "eda_categorical_counts.png"),
        "boxplots": P.boxplots(df, fig_dir / "eda_boxplots.png"),
        "correlation_heatmap": P.correlation_heatmap(df, fig_dir / "eda_correlation_heatmap.png"),
        "target_correlation": P.target_correlation(df, fig_dir / "eda_target_correlation.png"),
    }
    corr_t = df.corr(numeric_only=True)[C.TARGET].drop(C.TARGET).sort_values(ascending=False)
    desc = df[C.NUMERIC_FEATURES].describe().T.round(2)

    def rate(col):
        return {str(int(k)): round(float(v), 3) for k, v in df.groupby(col)[C.TARGET].mean().items()}

    summary = {
        "data_source": D.data_source(), "n_rows": int(len(df)), "n_features": len(C.FEATURES),
        "class_counts": {str(k): int(v) for k, v in df[C.TARGET].value_counts().sort_index().items()},
        "positive_rate": round(float(df[C.TARGET].mean()), 4),
        "missing": {k: int(v) for k, v in df[C.FEATURES].isna().sum().items() if v > 0},
        "male_share": round(float((df["sex"] == 1).mean()), 4),
        "target_correlation": {k: round(float(v), 3) for k, v in corr_t.items()},
        "numeric_describe": desc.reset_index().rename(columns={"index": "feature"}).to_dict("records"),
        "disease_rate_by_cp": rate("cp"), "disease_rate_by_sex": rate("sex"),
        "disease_rate_by_exang": rate("exang"),
        "figures": {k: str(v.relative_to(C.ROOT)) for k, v in figs.items()},
    }
    C.EDA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    log.info("EDA complete: %d figures", len(figs))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_eda()
