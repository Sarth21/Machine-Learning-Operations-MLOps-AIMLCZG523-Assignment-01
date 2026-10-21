"""End-to-end training pipeline:  python -m heartml.train

load+clean -> split -> tune 3 models (grid search, stratified CV) -> repeated CV of tuned models
-> pick best -> hold-out evaluation -> log to MLflow/local tracker -> refit winner on all data
-> save model + metadata.
"""
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from heartml import __version__
from heartml import config as C
from heartml import data as D
from heartml import modeling as M
from heartml import plots as P
from heartml.features import build_pipeline, feature_names
from heartml.tracking import Tracker

log = logging.getLogger("heartml.train")
_CM = ("tn", "fp", "fn", "tp")


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_training(df=None, fig_dir=None, model_path=None, metadata_path=None, metrics_path=None,
                 models=None, tracker=None, seed=C.RANDOM_STATE, quiet=False, specs=None):
    t0 = time.time()
    df = D.load_clean() if df is None else df
    fig_dir = Path(fig_dir or C.FIG_DIR)
    model_path = Path(model_path or C.MODEL_PATH)
    metadata_path = Path(metadata_path or C.METADATA_PATH)
    metrics_path = Path(metrics_path or C.METRICS_PATH)
    for d in (fig_dir, model_path.parent, metrics_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    tracker = tracker or Tracker()

    source = D.data_source()
    if D.is_synthetic():
        log.warning("*** DATA SOURCE IS SYNTHETIC - metrics are NOT clinically meaningful ***")
    X_train, X_test, y_train, y_test = D.train_test(df, seed=seed)
    log.info("train=%d test=%d positive-rate(train)=%.3f", len(X_train), len(X_test), y_train.mean())

    specs = specs or M.model_specs(seed)
    names = models or list(specs)
    data_hash = D.file_sha256(C.DATA_CLEAN) if C.DATA_CLEAN.exists() else "n/a"
    cv_table, test_table, best_params, fitted, curves, csv_files = {}, {}, {}, {}, {}, {}

    with tracker.run("training-pipeline", tags={"data_source": source, "git_sha": _git_sha()}) as parent:
        parent.log_params({"n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
                           "test_size": C.TEST_SIZE, "cv_folds": C.CV_FOLDS, "cv_repeats": C.CV_REPEATS,
                           "random_state": seed, "data_sha256": data_hash[:16], "data_source": source,
                           "sklearn": sklearn.__version__, "python": platform.python_version(),
                           "models_compared": ",".join(names)})
        for name in names:
            log.info("Tuning %s ...", name)
            with tracker.run(name, nested=True, tags={"model": name}) as run:
                search = M.tune(name, X_train, y_train, seed=seed, specs=specs)
                best = search.best_estimator_
                best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                cv_table[name] = M.cross_validate_pipeline(best, X_train, y_train, seed=seed)
                proba = best.predict_proba(X_test)[:, 1]
                test_table[name] = M.evaluate(y_test, proba)
                fitted[name] = best
                curves[name] = (y_test.values, proba)

                cvres = pd.DataFrame(search.cv_results_)
                keep = [c for c in cvres.columns if c.startswith("param_")
                        or c in ("mean_test_score", "std_test_score", "rank_test_score")]
                csv_path = fig_dir / f"cv_results_{name}.csv"
                cvres[keep].sort_values("rank_test_score").to_csv(csv_path, index=False)
                csv_files[name] = csv_path

                run.log_params({"model": name, **{f"best_{k}": v for k, v in best_params[name].items()},
                                "grid_size": len(search.cv_results_["params"]), "search_scoring": "roc_auc"})
                run.log_metrics({f"cv_{m}_mean": v[0] for m, v in cv_table[name].items()})
                run.log_metrics({f"cv_{m}_std": v[1] for m, v in cv_table[name].items()})
                run.log_metrics({"grid_best_cv_roc_auc": search.best_score_})
                run.log_metrics({f"test_{k}": v for k, v in test_table[name].items() if k not in _CM})
                cm = P.confusion(y_test, (proba >= C.DECISION_THRESHOLD).astype(int),
                                 f"{name} - test confusion matrix", fig_dir / f"confusion_{name}.png")
                run.log_artifact(cm, "plots")
                run.log_artifact(csv_path, "tuning")
                run.log_model(best, "model", input_example=X_train.head(3))
                log.info("%s: cv_auc=%.3f test_auc=%.3f", name, cv_table[name]["roc_auc"][0],
                         test_table[name]["roc_auc"])

        best_name = M.pick_best(cv_table)
        log.info("Selected model: %s", best_name)
        figs = [P.roc_curves(curves, fig_dir / "roc_curves.png"),
                P.pr_curves(curves, fig_dir / "pr_curves.png"),
                P.model_comparison(cv_table, fig_dir / "model_comparison.png"),
                P.calibration(*curves[best_name], f"Calibration - {best_name}", fig_dir / "calibration_best.png"),
                P.confusion(curves[best_name][0], (curves[best_name][1] >= C.DECISION_THRESHOLD).astype(int),
                            f"{best_name} (selected) - test confusion matrix", fig_dir / "confusion_best.png")]
        names_, imp_mean, imp_std = M.permutation_importances(fitted[best_name], X_test, y_test, seed=seed)
        figs.append(P.importance(names_, imp_mean, f"Permutation importance - {best_name}",
                                 fig_dir / "feature_importance.png"))
        for f in figs:
            parent.log_artifact(f, "plots")
        parent.log_params({"selected_model": best_name,
                           **{f"selected_{k}": v for k, v in best_params[best_name].items()}})
        parent.log_metrics({f"selected_test_{k}": v for k, v in test_table[best_name].items() if k not in _CM})
        parent.log_metrics({f"selected_cv_{k}": v[0] for k, v in cv_table[best_name].items()})

        # final model: winner refit with tuned params on ALL rows
        X_all, y_all = D.split_xy(df)
        est = specs[best_name][0]
        final = build_pipeline(est.__class__(**{**est.get_params(), **best_params[best_name]}))
        final.fit(X_all, y_all)
        joblib.dump(final, model_path)

        metadata = {
            "model_name": best_name,
            "model_version": f"{__version__}-{time.strftime('%Y%m%d%H%M%S')}",
            "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trained_on": "all rows (hyper-parameters chosen by CV on the training split)",
            "n_rows": int(len(df)), "features": C.FEATURES,
            "engineered_features": C.ENGINEERED_NUMERIC + C.ENGINEERED_CATEGORICAL,
            "transformed_feature_names": feature_names(final),
            "best_params": best_params[best_name],
            "holdout_metrics": test_table[best_name],
            "cv_metrics": {k: {"mean": v[0], "std": v[1]} for k, v in cv_table[best_name].items()},
            "decision_threshold": C.DECISION_THRESHOLD, "data_source": source, "data_sha256": data_hash,
            "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
            "numpy_version": np.__version__, "pandas_version": pd.__version__, "git_sha": _git_sha(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        parent.log_artifact(model_path, "final_model")
        parent.log_artifact(metadata_path, "final_model")
        parent.log_model(final, "final_model_sklearn", input_example=X_all.head(3))

        results = {
            "data_source": source, "synthetic": D.is_synthetic(), "tracking_backend": tracker.backend,
            "tracking_uri": str(tracker.uri), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "selected_model": best_name, "best_params": best_params,
            "grid_sizes": {n: int(len(pd.read_csv(csv_files[n]))) for n in names},
            "cv": {n: {m: {"mean": v[0], "std": v[1]} for m, v in t.items()} for n, t in cv_table.items()},
            "test": test_table,
            "permutation_importance": {"features": names_, "mean": [float(x) for x in imp_mean],
                                       "std": [float(x) for x in imp_std]},
            "seconds": round(time.time() - t0, 1),
        }
        metrics_path.write_text(json.dumps(results, indent=2, default=str))
        parent.log_artifact(metrics_path, "reports")

    if not quiet:
        _print_table(cv_table, test_table, best_name)
    log.info("Model saved to %s (%.1fs total)", model_path, time.time() - t0)
    return results


def _print_table(cv_table, test_table, best):
    print("\n{:<22}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "model", "cv_auc", "cv_rec", "test_auc", "test_acc", "test_rec"))
    for n in cv_table:
        print("{:<22}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.3f}{}".format(
            n, cv_table[n]["roc_auc"][0], cv_table[n]["recall"][0], test_table[n]["roc_auc"],
            test_table[n]["accuracy"], test_table[n]["recall"], " *" if n == best else ""))
    print("(* = selected)\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_training()
