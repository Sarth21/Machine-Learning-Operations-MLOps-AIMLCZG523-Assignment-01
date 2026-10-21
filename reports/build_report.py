#!/usr/bin/env python
"""Build reports/MLOps_Assignment1_Report.pdf from the latest metrics, figures, diagrams and screenshots.

    python reports/build_report.py

Numbers are read from reports/metrics.json + reports/eda_summary.json, so re-running after
`make data && make all` refreshes the report with real-data results. Any PNG/JPG you drop into
screenshots/ (see screenshots/README.md for the expected names) is embedded automatically.
"""
import json
import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from heartml import config as C  # noqa: E402

REPO_URL = "https://github.com/<your-username>/heart-disease-mlops"   # <- edit before submitting
STUDENT = "Sarth"
OUT = ROOT / "reports" / "MLOps_Assignment1_Report.pdf"
FIG, DIA, SHOT, EVI = ROOT / "reports/figures", ROOT / "docs/diagrams", ROOT / "screenshots", ROOT / "docs/evidence"

M = json.loads((ROOT / "reports/metrics.json").read_text())
E = json.loads((ROOT / "reports/eda_summary.json").read_text())
META = json.loads((ROOT / "models/model_metadata.json").read_text())
SYN = bool(M.get("synthetic"))
BEST = M["selected_model"]
PRETTY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
          "gradient_boosting": "Gradient Boosting"}

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.6, leading=13.6, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceBefore=6, spaceAfter=8, textColor=colors.HexColor("#1A365D"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2C5282"))
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.3, leading=10.5, textColor=colors.HexColor("#4A5568"), alignment=TA_CENTER)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.4, leading=10.8)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.4, leading=9.2, backColor=colors.HexColor("#F7FAFC"),
                      borderColor=colors.HexColor("#CBD5E0"), borderWidth=0.5, borderPadding=4, spaceAfter=8)
WARN = ParagraphStyle("warn", parent=BODY, backColor=colors.HexColor("#FFF5F5"), borderColor=colors.HexColor("#C53030"),
                      borderWidth=0.8, borderPadding=6, spaceAfter=10, textColor=colors.HexColor("#742A2A"))
W = A4[0] - 4 * cm
story = []
fig_no = [0]
tab_no = [0]


def P(t, st=BODY):
    story.append(Paragraph(t, st))


def bullets(items):
    for i in items:
        story.append(Paragraph("&bull;&nbsp; " + i, ParagraphStyle("b", parent=BODY, leftIndent=14, firstLineIndent=-10, spaceAfter=2)))
    story.append(Spacer(1, 4))


def code(text):
    story.append(Preformatted(text.strip("\n"), CODE))


def img(path, caption, width=None, maxh=None):
    path = Path(path)
    if not path.exists():
        return
    w0, h0 = PILImage.open(path).size
    width = min(width or W, W)
    h = width * h0 / w0
    if maxh and h > maxh:
        width, h = width * maxh / h, maxh
    fig_no[0] += 1
    story.append(KeepTogether([Image(str(path), width=width, height=h), Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP), Spacer(1, 8)]))


def table(rows, widths=None, header=True, caption=None, font=8.4):
    cell = ParagraphStyle("cell", parent=BODY, fontSize=font, leading=font + 2.4, spaceAfter=0)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    rows = [[Paragraph(c, head if (header and i == 0) else cell) if isinstance(c, str) else c for c in r]
            for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")])]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5282")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    story.append(t)
    if caption:
        tab_no[0] += 1
        story.append(Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", CAP))
    story.append(Spacer(1, 8))


def shot(name, caption):
    """Embed screenshots/<name>.(png|jpg) if present, else a clearly-marked placeholder."""
    for ext in ("png", "jpg", "jpeg"):
        p = SHOT / f"{name}.{ext}"
        if p.exists():
            img(p, caption, width=W * 0.92, maxh=11 * cm)
            return
    box = Table([[Paragraph(f"<b>SCREENSHOT TO ADD:</b> <font face='Courier'>screenshots/{name}.png</font><br/>{caption}", SMALL)]],
                colWidths=[W * 0.92])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#DD6B20")), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFAF0")),
                             ("TOPPADDING", (0, 0), (-1, -1), 14), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
    story.append(box)
    story.append(Spacer(1, 8))


def f3(x):
    return f"{x:.3f}"


def pm(d):
    return f"{d['mean']:.3f} &plusmn; {d['std']:.3f}"


def page_deco(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(2 * cm, 1.2 * cm, "MLOps (AIMLCZG523) - Assignment 01 - Heart Disease Risk Prediction")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# =================================================================== title page
story.append(Spacer(1, 3.2 * cm))
P("BITS Pilani - Machine Learning Operations (MLOps) AIMLCZG523", ParagraphStyle("t0", parent=BODY, fontSize=11, alignment=TA_CENTER))
P("Assignment 01 - End-to-End ML Model Development, CI/CD and Production Deployment",
  ParagraphStyle("t1", parent=H1, fontSize=22, leading=27, alignment=TA_CENTER, spaceBefore=18))
P("Heart Disease Risk Prediction as a Containerised, Monitored API", ParagraphStyle("t2", parent=H2, fontSize=14, alignment=TA_CENTER))
story.append(Spacer(1, 1.2 * cm))
table([["Student", STUDENT], ["Dataset", "UCI Heart Disease (Cleveland), 14 attributes, binary target"],
       ["Stack", "scikit-learn, MLflow, Flask + gunicorn, Docker, Kubernetes, Prometheus, Grafana, GitHub Actions"],
       ["Repository", REPO_URL], ["Report generated from", f"reports/metrics.json (data source: {M['data_source']})"]],
      widths=[4 * cm, W - 4 * cm], header=False, font=9.5)
if SYN:
    P("<b>Data provenance notice.</b> The figures and metrics in this generated copy of the report were produced on a "
      "<b>synthetic stand-in dataset</b> with the same schema as UCI Cleveland, because the build environment had no internet "
      "access. They validate that the pipeline works end to end but are <b>not clinically meaningful</b>. Run "
      "<font face='Courier'>make data &amp;&amp; make all &amp;&amp; make report</font> to regenerate every number, plot and table "
      "from the real UCI data before submission.", WARN)
story.append(PageBreak())

# ================================================================== contents
P("Contents", H1)
toc = ["1. Introduction and objectives", "2. Repository layout and setup instructions", "3. Data acquisition and cleaning",
       "4. Exploratory data analysis", "5. Feature engineering and model development", "6. Experiment tracking (MLflow)",
       "7. Model packaging and reproducibility", "8. Automated testing and CI/CD pipeline", "9. Containerisation (Docker)",
       "10. Production deployment (Kubernetes)", "11. Monitoring and logging", "12. Architecture overview",
       "13. Results, limitations and ethical considerations", "14. Requirement-to-evidence checklist", "Appendix A - API reference, Appendix B - commands"]
bullets(toc)
P("Marks map: EDA 5, modelling 8, tracking 5, packaging 7, CI/CD 8, containerisation 5, deployment 7, monitoring 3, documentation 2 (total 50).", SMALL)
story.append(PageBreak())

# ============================================================== 1 introduction
P("1. Introduction and objectives", H1)
P("Cardiovascular disease is a leading cause of death worldwide, and early risk stratification from routine clinical measurements "
  "can prioritise patients for further testing. This project builds a binary classifier that estimates the risk of heart disease from "
  "13 patient attributes (age, sex, chest-pain type, blood pressure, cholesterol, ECG findings, exercise test results and fluoroscopy "
  "findings) and ships it the way a production ML system should be shipped: reproducible, tested, tracked, containerised, deployed "
  "to Kubernetes and monitored.")
P("<b>Design goals</b>", H2)
bullets(["<b>Reproducibility</b> - one command per stage, pinned dependencies, fixed seeds, data hash and library versions stored with the model.",
         "<b>No train/serve skew</b> - a single scikit-learn Pipeline (feature engineering + imputation + scaling + encoding + classifier) is trained, cross-validated, pickled and served.",
         "<b>No data leakage</b> - imputation and scaling are fitted inside cross-validation folds only.",
         "<b>Fail loudly</b> - schema validation on data and on API requests, tests gate CI, the pipeline stops on any error.",
         "<b>Observability</b> - structured JSON logs, request IDs, Prometheus metrics, Grafana dashboard and alert rules.",
         "<b>Privacy by default</b> - patient feature values are never logged unless explicitly enabled."])
img(DIA / "system_architecture.png", "End-to-end system architecture: GitHub, CI/CD, MLflow, container registry, Kubernetes serving and monitoring.")

# ================================================================ 2 layout / setup
P("2. Repository layout and setup instructions", H1)
code("""
heart-disease-mlops/
  heartml/            package: config, data, features, modeling, tracking, train, eda, plots, predict, schema
  app/                Flask API (main.py) + dependency-free Prometheus registry (metrics.py)
  scripts/            download_data.py, smoke_test.py, load_test.py
  tests/              48 unit tests (data, features, model/training/tracking, API, metrics)
  notebooks/          01_eda, 02_training, 03_inference
  data/raw|processed  downloaded dataset and cleaned dataset (+ DATA_SOURCE.txt provenance marker)
  models/             model.joblib + model_metadata.json
  reports/            metrics.json, eda_summary.json, figures/, build_report.py
  docs/diagrams/      architecture diagrams (+ build script);  docs/evidence/  captured local run outputs
  k8s/                namespace, configmap, deployment, services, ingress, HPA, Prometheus, Grafana, kustomization
  monitoring/         prometheus.yml, alerts.yml, Grafana provisioning + dashboard JSON (docker-compose stack)
  .github/workflows/  ci.yml        Dockerfile  docker-compose.yml  Makefile  requirements*.txt
""")
P("<b>Quick start (clean machine)</b>", H2)
code("""
git clone <repo-url> && cd heart-disease-mlops
python -m venv .venv && source .venv/bin/activate        # Python 3.13
pip install -r requirements-dev.txt

python scripts/download_data.py        # real UCI data (ucimlrepo, falls back to direct HTTPS)
make all                               # clean -> EDA -> train (3 models, MLflow tracked)
make test                              # 48 unit tests with coverage
make run                               # API on http://localhost:8000

docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0
python scripts/smoke_test.py http://localhost:8000

docker compose up --build              # API + Prometheus + Grafana
kubectl apply -k k8s/                  # Kubernetes (Minikube / Docker Desktop / cloud)
""")
P("The download script tries the <font face='Courier'>ucimlrepo</font> package (dataset id 45), then a direct HTTPS download of "
  "<font face='Courier'>processed.cleveland.data</font>. If both fail it prints manual instructions. An explicit "
  "<font face='Courier'>--synthetic</font> flag exists only for offline dry runs; it writes a provenance marker that is propagated to the logs, the tracker, "
  "<font face='Courier'>model_metadata.json</font>, <font face='Courier'>/model-info</font> and this report so synthetic results can never be mistaken for real ones.")
story.append(PageBreak())

# ============================================================== 3 data
P("3. Data acquisition and cleaning", H1)
P("The Cleveland subset of the UCI Heart Disease database has 303 patients and 13 predictors. The original target (<i>num</i>) grades disease "
  "severity 0-4; following common practice it is binarised: 0 = no disease, 1-4 = disease present. Missing values are encoded as '?' in the raw file.")
rows = [["Feature", "Type", "Description"]]
for f in C.FEATURES:
    rows.append([f, "numeric" if f in C.NUMERIC_FEATURES else "categorical", C.FEATURE_DESCRIPTIONS[f]])
table(rows, widths=[2.2 * cm, 2.4 * cm, W - 4.6 * cm], caption="Input features.", font=8)
P("<b>Cleaning steps</b> (<font face='Courier'>heartml/data.py::clean</font>)", H2)
bullets(["Schema validation - fails with <i>DataValidationError</i> if any expected column is absent; columns are re-ordered deterministically.",
         "Coercion - every column is converted to numeric; '?' and any unparseable token become NaN.",
         "Plausibility rules - physiologically impossible values (e.g. cholesterol 0, systolic BP 900, undefined category codes) are set to NaN and logged.",
         "Duplicate rows are dropped; rows without a target are removed; target is binarised.",
         "<b>Missing values are kept as NaN</b> in the cleaned CSV. Imputation (median for numerics, most-frequent for categoricals) is a pipeline step fitted on training folds only, which prevents leakage and guarantees that the API handles missing values exactly as training did."])
P(f"After cleaning the dataset has <b>{E['n_rows']}</b> rows; missing values: "
  + ", ".join(f"<b>{k}</b> = {v}" for k, v in E["missing"].items()) + ".")
img(FIG / "eda_missing_values.png", "Missing values per feature after cleaning.", width=W * 0.6)

# ============================================================== 4 EDA
P("4. Exploratory data analysis", H1)
c0, c1 = E["class_counts"]["0"], E["class_counts"]["1"]
P(f"<b>Class balance.</b> {c0} patients without disease and {c1} with disease ({E['positive_rate']:.1%} positive). The classes are nearly balanced, so accuracy is "
  "a meaningful metric, but because a missed case is more costly than a false alarm, recall and ROC-AUC are monitored as well and used for model selection tie-breaking.")
img(FIG / "eda_class_balance.png", "Class balance.", width=W * 0.5)
P("<b>Distributions.</b> Histograms with kernel density estimates, split by class, show which numeric features separate the classes.")
img(FIG / "eda_numeric_histograms.png", "Numeric feature distributions by class.")
img(FIG / "eda_categorical_counts.png", "Categorical feature counts by class.")
img(FIG / "eda_boxplots.png", "Box plots for outlier inspection. Extreme but plausible values are retained; only impossible values are nulled.")
tc = E["target_correlation"]
top = list(tc.items())[:3]
bot = list(tc.items())[-2:]
P("<b>Correlation.</b> Pearson correlations with the target: strongest positive "
  + ", ".join(f"<b>{k}</b> ({v:+.2f})" for k, v in top) + "; strongest negative "
  + ", ".join(f"<b>{k}</b> ({v:+.2f})" for k, v in bot) + ". No pair of predictors is so collinear that it threatens the linear model; regularisation handles the moderate correlations.")
img(FIG / "eda_correlation_heatmap.png", "Correlation heatmap (lower triangle).", width=W * 0.78)
img(FIG / "eda_target_correlation.png", "Correlation of each feature with the target.", width=W * 0.62)
P(f"<b>Subgroup rates.</b> Disease prevalence by sex: female {E['disease_rate_by_sex']['0']:.1%}, male {E['disease_rate_by_sex']['1']:.1%}; "
  f"with exercise-induced angina {E['disease_rate_by_exang']['1']:.1%} vs {E['disease_rate_by_exang']['0']:.1%} without. "
  f"Males make up {E['male_share']:.0%} of the sample, which is a representativeness limitation discussed in Section 13.")
story.append(PageBreak())

# ============================================================== 5 modelling
P("5. Feature engineering and model development", H1)
P("<b>Preprocessing pipeline.</b> A scikit-learn <i>ColumnTransformer</i> inside a single <i>Pipeline</i>:")
table([["Step", "Applies to", "Operation"],
       ["FeatureAdder", "all rows", "hr_reserve_ratio = thalach / (220 - age); chol_per_age = chol / age; high_bp = (trestbps >= 140)"],
       ["Numeric branch", "age, trestbps, chol, thalach, oldpeak, ca + 2 engineered", "median imputation -> StandardScaler"],
       ["Categorical branch", "sex, cp, fbs, restecg, exang, slope, thal + high_bp", "most-frequent imputation -> OneHotEncoder(handle_unknown='ignore')"],
       ["Classifier", "-", "Logistic Regression / Random Forest / Gradient Boosting"]],
      widths=[3.2 * cm, 5.6 * cm, W - 8.8 * cm], caption="Pipeline stages.")
img(DIA / "ml_pipeline.png", "Training pipeline from raw data to a saved, tracked model.", width=W * 0.42)
P("The engineered features encode clinical knowledge: the heart-rate reserve ratio compares the achieved maximum heart rate with the age-predicted maximum, "
  "cholesterol is normalised by age, and a blood-pressure flag marks the stage-2 hypertension threshold.")
P("<b>Models and tuning.</b> Three classifiers with different inductive biases were compared. Each was tuned with an exhaustive grid search using "
  "5-fold stratified cross-validation on the training split (80 %), optimising ROC-AUC. The tuned models were then re-evaluated with <b>repeated stratified 5-fold CV (3 repeats = 15 fits)</b> "
  "to obtain stable mean +/- std estimates, and finally scored once on the 20 % hold-out test set.")
rows = [["Model", "Search space", "Grid size", "Best parameters"]]
space = {"logistic_regression": "C in {0.01..10}, class_weight in {None, balanced}",
         "random_forest": "n_estimators {150,300}, max_depth {3,5,8,None}, min_samples_leaf {1,3,5}",
         "gradient_boosting": "n_estimators {50,100,200}, learning_rate {0.03,0.1}, max_depth {2,3}"}
for n in M["cv"]:
    rows.append([PRETTY[n], Paragraph(space[n], SMALL), str(M["grid_sizes"][n]), Paragraph(", ".join(f"{k}={v}" for k, v in M["best_params"][n].items()), SMALL)])
table(rows, widths=[3.2 * cm, 6.4 * cm, 1.8 * cm, W - 11.4 * cm], caption="Hyper-parameter search and selected values.")
P("<b>Cross-validated results</b> (repeated stratified 5-fold, mean &plusmn; std):")
rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
for n, r in M["cv"].items():
    rows.append([PRETTY[n] + (" *" if n == BEST else "")] + [pm(r[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")])
table(rows, widths=[3.8 * cm] + [(W - 3.8 * cm) / 5] * 5, caption="Cross-validation metrics (* = selected model).")
img(FIG / "model_comparison.png", "Cross-validated metrics of the three tuned models (error bars = std).", width=W * 0.8)
P("<b>Hold-out test results</b> (61 patients, threshold 0.5):")
rows = [["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "Brier"]]
for n, t in M["test"].items():
    rows.append([PRETTY[n] + (" *" if n == BEST else "")] + [f3(t[k]) for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "brier")])
table(rows, widths=[3.6 * cm] + [(W - 3.6 * cm) / 7] * 7, caption="Hold-out test metrics.", font=8)
tb = M["test"][BEST]
P(f"<b>Model selection.</b> <b>{PRETTY[BEST]}</b> was selected: highest cross-validated ROC-AUC ({M['cv'][BEST]['roc_auc']['mean']:.3f}), ties broken by recall because a false negative "
  f"(missed disease) is the costlier error. On the hold-out set it produced {tb['tp']} true positives, {tb['tn']} true negatives, {tb['fp']} false positives and {tb['fn']} false negatives. "
  "Selection used cross-validation only - the test set was never used to choose a model, so its scores are an unbiased estimate.")
img(FIG / "roc_curves.png", "ROC curves on the hold-out test set.", width=W * 0.52)
img(FIG / "pr_curves.png", "Precision-recall curves on the hold-out test set.", width=W * 0.52)
img(FIG / "confusion_best.png", "Confusion matrix of the selected model.", width=W * 0.42)
img(FIG / "calibration_best.png", "Calibration curve of the selected model (reliability diagram). Probabilities reported by the API are used as the confidence value.", width=W * 0.5)
img(FIG / "feature_importance.png", "Permutation importance of the raw input features on the test set (drop in ROC-AUC when shuffled).", width=W * 0.62)
pi = M["permutation_importance"]
order = sorted(zip(pi["features"], pi["mean"]), key=lambda t: -t[1])[:4]
P("The most influential inputs were " + ", ".join(f"<b>{k}</b> ({v:.3f})" for k, v in order) + ".")
P("<b>Notes and caveats.</b> The final deployed model is refit on <i>all</i> rows with the CV-selected hyper-parameters, a standard practice once model selection is complete; its hold-out score therefore "
  "is not independent and the figures above (from the train-only fit) are the ones to quote. With only 303 rows, confidence intervals are wide (see the std columns), so differences of 1-2 points between "
  "models are not significant. If a selected hyper-parameter sits at the edge of its grid (e.g. the largest C), a wider grid should be explored.")
story.append(PageBreak())

# ============================================================== 6 tracking
P("6. Experiment tracking (MLflow)", H1)
P("Every run is tracked through <font face='Courier'>heartml/tracking.py</font>, which uses <b>MLflow</b> when installed (the default via requirements.txt) and a small file-based fallback "
  "with the same API when it is not, so the pipeline never silently loses tracking. The structure is a <b>parent run</b> (<i>training-pipeline</i>) with one <b>nested child run per model</b>.")
table([["Logged item", "Parent run", "Child run (per model)"],
       ["Parameters", "rows, split sizes, seed, CV folds/repeats, data SHA-256, data source, library versions, models compared, selected model + params",
        "model name, every best_* hyper-parameter, grid size, search metric"],
       ["Metrics", "selected_test_* (accuracy, precision, recall, specificity, F1, ROC-AUC, AP, Brier), selected_cv_*",
        "cv_*_mean / cv_*_std for 5 metrics, grid_best_cv_roc_auc, test_* metrics"],
       ["Artifacts / plots", "ROC, PR, comparison, calibration, confusion, importance, final model + metadata, metrics.json", "confusion matrix, full grid-search table (CSV)"],
       ["Models", "final pipeline (MLflow sklearn flavour, with input example)", "tuned pipeline per model"],
       ["Tags", "data_source, git_sha", "model"]],
      widths=[2.8 * cm, (W - 2.8 * cm) * 0.55, (W - 2.8 * cm) * 0.45], caption="What is logged to MLflow.")
P(f"Tracking backend used to generate this report: <b>{M['tracking_backend']}</b> (<font face='Courier'>{Path(M['tracking_uri']).name}</font>). "
  "Browse runs with <font face='Courier'>mlflow ui --backend-store-uri ./mlruns</font> (or <font face='Courier'>make mlflow-ui</font>) at http://localhost:5000.")
shot("mlflow_runs_list", "MLflow UI - experiment 'heart-disease-risk' with the parent run and three nested model runs.")
shot("mlflow_run_detail", "MLflow UI - run detail with parameters, metrics and logged plots/artifacts.")
shot("mlflow_compare", "MLflow UI - comparison of the three model runs.")
P("A summary of the tuning process is saved for every model as <font face='Courier'>reports/figures/cv_results_&lt;model&gt;.csv</font> (all grid points ranked by CV ROC-AUC), making the selection auditable.")

# ============================================================== 7 packaging
P("7. Model packaging and reproducibility", H1)
bullets(["<b>Reusable format.</b> <font face='Courier'>models/model.joblib</font> is the whole fitted Pipeline (feature engineering + preprocessing + classifier), so serving needs no hand-written preprocessing. The same pipeline is also logged to MLflow's sklearn flavour.",
         "<b>Metadata.</b> <font face='Courier'>models/model_metadata.json</font> records model name and version, UTC training time, best parameters, hold-out and CV metrics, decision threshold, data source and SHA-256, git commit, and the exact Python / scikit-learn / NumPy / pandas versions.",
         "<b>Pinned environments.</b> <font face='Courier'>requirements-api.txt</font> (serving, small) and <font face='Courier'>requirements.txt</font> / <font face='Courier'>requirements-dev.txt</font> (full / dev). Serving pins must match the training versions because pickles are version-sensitive; "
         "the container image installs exactly those pins.",
         "<b>Determinism.</b> Fixed random seeds for the split, CV folds, and models; a unit test asserts that tuning twice gives identical results.",
         "<b>Clean-setup execution.</b> Every stage is a single command (Makefile) that works from a fresh virtual environment: download -> clean -> EDA -> train -> test -> serve."])
meta_rows = [["Field", "Value"]] + [[k, str(META[k])] for k in ("model_name", "model_version", "trained_at_utc", "data_source", "sklearn_version", "python_version", "numpy_version", "pandas_version")]
table(meta_rows, widths=[4 * cm, W - 4 * cm], caption="Excerpt of model_metadata.json stored beside the model.")
story.append(PageBreak())

# ============================================================== 8 CI/CD
P("8. Automated testing and CI/CD pipeline", H1)
P("<b>Unit tests</b> (<font face='Courier'>tests/</font>, pytest-compatible, written with unittest classes): <b>48 tests</b> in four modules.")
table([["Module", "Covers"],
       ["test_data.py", "'?' to NaN, target binarisation, impossible-value nulling, duplicate removal, schema errors, column ordering, stratified and reproducible splits"],
       ["test_features.py", "engineered feature values, input not mutated, NaN propagation, type check, no NaNs reach the classifier, scaling and one-hot output, unseen categories, imputation at predict time"],
       ["test_model.py", "metric ranges and perfect-score case, model selection rule, a full training run on small data: artefacts written, tracking params/metrics/plots logged, probabilities valid, determinism"],
       ["test_api.py", "request schema (missing/unknown/type/range/null), Prometheus renderer + label escaping, every endpoint, 200/400/404/405/413/422 paths, batch input, metrics counters"]],
      widths=[3.2 * cm, W - 3.2 * cm], caption="Test suite.")
P("Training tests use tiny search grids so the whole suite finishes in about 12 seconds. Captured output of the last local run:")
ut = (EVI / "unit_tests.txt")
if ut.exists():
    code(ut.read_text().strip())
P("<b>GitHub Actions workflow</b> (<font face='Courier'>.github/workflows/ci.yml</font>) runs on every push and pull request to <i>main</i>:")
img(DIA / "cicd_workflow.png", "CI/CD workflow. Each job depends on the previous one, so any failure stops the pipeline.")
table([["Job", "Steps", "Artifacts / logs"],
       ["lint", "flake8 over the repo (fails on any violation)", "annotated source lines in the log"],
       ["test", "pip install -r requirements-dev.txt; pytest with coverage", "test-results.xml, coverage.xml"],
       ["train", "download UCI data -> clean -> EDA -> train (MLflow); job summary table", "models/, reports/, mlruns/, train.log"],
       ["docker", "download the model, docker build, run container, run smoke test against it; push to GHCR on main only", "container.log"],
       ["k8s-validate", "kustomize render + kubeconform strict schema validation", "log"]],
      widths=[2.4 * cm, W - 6.4 * cm, 4 * cm], caption="Pipeline jobs.")
P("The pipeline <b>fails on code or test errors</b> with clear logs: jobs are chained with <font face='Courier'>needs:</font>, the training step uses "
  "<font face='Courier'>set -o pipefail</font> so a crash is not hidden by <font face='Courier'>tee</font>, and the smoke test exits non-zero on the first failed check.")
shot("github_actions_run", "GitHub Actions - successful run showing all jobs (lint, test, train, docker, k8s-validate).")
shot("github_actions_artifacts", "GitHub Actions - uploaded artifacts (test reports, training outputs, container logs).")
shot("github_actions_failure_demo", "Optional: a deliberately failing run (e.g. a broken test) showing the pipeline stops with a clear error.")
story.append(PageBreak())

# ============================================================== 9 Docker
P("9. Containerisation (Docker)", H1)
P("The API is a Flask application served by gunicorn. The image is based on <i>python:3.13-slim</i>, installs only the small serving requirements, copies the code and trained model, "
  "runs as an <b>unprivileged user</b> (uid 10001) and defines a <b>HEALTHCHECK</b>. gunicorn runs <b>one worker with four threads</b> so the in-process Prometheus registry "
  "sees every request; horizontal scaling is done with Kubernetes replicas.")
code("""
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000 MODEL_PATH=/app/models/model.joblib
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt     # cached layer
COPY heartml/ heartml/   COPY app/ app/   COPY models/ models/
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser
HEALTHCHECK ... GET /health
CMD ["gunicorn","--bind","0.0.0.0:8000","--workers","1","--threads","4","app.main:create_app()"]
""")
P("<b>Endpoints.</b> <font face='Courier'>POST /predict</font> accepts one JSON object (or a list / <i>{\"instances\": [...]}</i>, max 100) and returns the prediction and confidence; "
  "<font face='Courier'>GET /health</font>, <font face='Courier'>/ready</font>, <font face='Courier'>/model-info</font> and <font face='Courier'>/metrics</font> support operations.")
P("<b>Sample request and response</b> (captured from the locally running API):")
code("""
$ curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":63,"sex":1,"cp":1,"trestbps":145,
    "chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
""" + (EVI / "curl_predict.json").read_text().strip() + "\n")
P("Invalid input is rejected with HTTP 422 and a precise list of problems (captured):")
inv = json.loads((EVI / "curl_predict_invalid.json").read_text())
code(json.dumps({"error": inv["error"], "details": [{"index": 0, "errors": inv["details"][0]["errors"][:3] + ["..."]}]}, indent=1))
P("<b>Local smoke test</b> (<font face='Courier'>scripts/smoke_test.py</font>, run against the live server; the same script gates the Docker CI job):")
code((EVI / "smoke_test.txt").read_text().strip().replace('{"prediction"', '{"prediction"')[:1500])
P("<b>Docker proof required by the assignment</b> - to be captured on a machine with Docker:")
shot("docker_build", "Terminal - docker build -t heart-disease-api:1.0.0 . completing successfully.")
shot("docker_run_predict", "Terminal - docker run -p 8000:8000 and a successful curl /predict against the container.")
shot("docker_smoke_test", "Terminal - python scripts/smoke_test.py http://localhost:8000 against the container (all PASS).")
story.append(PageBreak())

# ============================================================== 10 Kubernetes
P("10. Production deployment (Kubernetes)", H1)
P("Manifests in <font face='Courier'>k8s/</font> are applied with <font face='Courier'>kubectl apply -k k8s/</font> (kustomize). They run unchanged on Minikube, Docker Desktop Kubernetes, GKE, EKS or AKS; "
  "only the image reference and the Service type behaviour differ (see below).")
img(DIA / "k8s_deployment.png", "Kubernetes resources in the heart-mlops namespace.", width=W * 0.85)
table([["Resource", "Purpose / key settings"],
       ["Namespace heart-mlops", "isolates all resources"],
       ["ConfigMap heart-api-config", "LOG_LEVEL, LOG_FEATURES=false, MODEL_PATH - configuration separated from the image"],
       ["Deployment heart-api", "2 replicas, rolling update (maxUnavailable 0), readiness /ready, liveness /health, CPU/memory requests+limits, runAsNonRoot, read-only root FS, all capabilities dropped, seccomp RuntimeDefault"],
       ["Service heart-api (LoadBalancer :80)", "public endpoint; cloud provisions a load balancer, Minikube uses 'minikube tunnel'"],
       ["Service heart-api-headless", "per-pod DNS discovery so Prometheus scrapes every replica"],
       ["Ingress heart-api.local", "alternative HTTP entry (nginx ingress class)"],
       ["HorizontalPodAutoscaler", "2-5 replicas at 70 % CPU (needs metrics-server)"],
       ["Prometheus / Grafana", "monitoring stack with provisioned data source and dashboard (Section 11)"]],
      widths=[5 * cm, W - 5 * cm], caption="Kubernetes resources.")
P("<b>Deploy on Minikube</b>", H2)
code("""
minikube start --driver=docker
minikube addons enable metrics-server ingress
docker build -t heart-disease-api:1.0.0 .
minikube image load heart-disease-api:1.0.0          # no registry needed
kubectl apply -k k8s/
kubectl -n heart-mlops rollout status deploy/heart-api
minikube tunnel                                       # separate terminal: gives the LoadBalancer an IP
kubectl -n heart-mlops get svc heart-api              # EXTERNAL-IP
python scripts/smoke_test.py http://<EXTERNAL-IP>
""")
P("<b>Deploy on a public cloud.</b> Push the CI-built image (<font face='Courier'>ghcr.io/&lt;owner&gt;/heart-disease-api:&lt;sha&gt;</font>), set it in <font face='Courier'>k8s/deployment.yaml</font> with "
  "<font face='Courier'>imagePullPolicy: Always</font>, and apply the same manifests to the cluster; the Service of type LoadBalancer yields a public IP.")
shot("k8s_get_pods", "kubectl -n heart-mlops get pods,svc,hpa - pods Running and a LoadBalancer address.")
shot("k8s_rollout", "kubectl rollout status / describe deployment.")
shot("k8s_predict_via_lb", "curl POST /predict through the LoadBalancer / Ingress address.")
story.append(PageBreak())

# ============================================================== 11 Monitoring
P("11. Monitoring and logging", H1)
P("<b>Structured request logging.</b> Every request produces one JSON log line (timestamp, level, request_id, method, path, status, latency in ms, client address) and every prediction produces a second line "
  "(prediction, confidence, model_version). The <font face='Courier'>X-Request-ID</font> header is honoured or generated and returned, allowing a client report to be traced to log lines. "
  "Raw patient features are <b>not</b> logged unless <font face='Courier'>LOG_FEATURES=true</font>. Output goes to stdout (collected by Docker/Kubernetes) and optionally to <font face='Courier'>LOG_FILE</font>.")
code(
    '{"ts":"2026-10-20T05:16:12.948Z","level":"INFO","logger":"heart-api","msg":"request","request_id":"8abfa35976b0",\n'
    ' "method":"POST","path":"/predict","status":200,"latency_ms":10.4,"remote_addr":"127.0.0.1"}')
P("<b>Prometheus metrics</b> (<font face='Courier'>GET /metrics</font>, a dependency-free implementation of the text exposition format):")
table([["Metric", "Type", "Use"],
       ["http_requests_total{method,endpoint,status}", "counter", "traffic and error rate"],
       ["http_request_duration_seconds{method,endpoint}", "histogram", "latency percentiles (p50/p95/p99)"],
       ["heart_inference_seconds", "histogram", "model inference time"],
       ["heart_predictions_total{label}", "counter", "predicted-class mix (input/prediction drift signal)"],
       ["heart_prediction_confidence", "histogram", "confidence distribution shift"],
       ["prediction_errors_total{reason}", "counter", "validation / bad JSON / inference failures"],
       ["heart_model_info{version,model}", "gauge", "which model version is live"],
       ["process_uptime_seconds", "gauge", "restarts"]], widths=[7.4 * cm, 2.2 * cm, W - 9.6 * cm], caption="Exported metrics.", font=8)
P("Excerpt scraped from the live local server after the load test (<font face='Courier'>scripts/load_test.py</font>, 120 requests including 5 % deliberately invalid):")
code((EVI / "metrics_excerpt.txt").read_text().strip())
P("<b>Dashboards and alerts.</b> <font face='Courier'>docker compose up</font> starts Prometheus (:9090) and Grafana (:3000, admin/admin) with the data source and the "
  "<i>Heart Disease API - Monitoring</i> dashboard provisioned automatically (request rate, p95 latency, 5xx ratio, predictions by class, mean confidence, rejected requests, totals, uptime). "
  "On Kubernetes the same dashboard is mounted from a ConfigMap and Prometheus discovers all pods through the headless Service. "
  "<font face='Courier'>monitoring/alerts.yml</font> defines alerts for target down, &gt;5 % 5xx, p95 latency &gt;500 ms and a share of positive predictions above 80 % (possible drift).")
shot("grafana_dashboard", "Grafana dashboard 'Heart Disease API - Monitoring' after running scripts/load_test.py.")
shot("prometheus_targets", "Prometheus - Status > Targets showing heart-api UP.")
shot("api_logs", "kubectl logs / docker logs showing structured JSON request and prediction logs.")
story.append(PageBreak())

# ============================================================== 12 Architecture
P("12. Architecture overview", H1)
P("<b>Request path.</b> A client sends JSON to the LoadBalancer/Ingress, which routes to one of the API pods. The service validates the payload, runs the same pipeline used in training, "
  "returns the prediction with its confidence, and records logs and metrics. Prometheus scrapes each pod; Grafana visualises the metrics.")
img(DIA / "serving_flow.png", "Inside the API: validation, preprocessing, inference, logging and metrics.", width=W * 0.7)
P("<b>Key design decisions</b>", H2)
table([["Decision", "Rationale", "Alternative considered"],
       ["Single sklearn Pipeline artefact", "eliminates train/serve skew; one file to version", "separate scaler/encoder pickles"],
       ["Imputation inside the pipeline", "no leakage in CV; API accepts missing ca/thal exactly as training", "impute in the cleaning script"],
       ["Flask + gunicorn (1 worker, threads)", "simple, small image; consistent in-process metrics", "FastAPI (automatic schema docs); multi-worker with prometheus multiprocess mode"],
       ["Hand-rolled validation module", "framework-free, unit-testable, precise error messages", "pydantic models"],
       ["Dependency-free /metrics", "no extra runtime dependency, easy to read and test", "prometheus_client"],
       ["MLflow with local fallback", "assignment tracking requirement plus robustness", "Weights & Biases"],
       ["Kustomize for manifests", "single `kubectl apply -k`; no templating engine needed", "Helm chart"],
       ["Model baked into the image", "immutable, reproducible deployment; image tag = model version", "model pulled from registry at start-up"]],
      widths=[4.2 * cm, 6.8 * cm, W - 11 * cm], caption="Architecture decisions.", font=8)

# ============================================================== 13 results / limits
P("13. Results, limitations and ethical considerations", H1)
P(f"<b>Outcome.</b> The selected {PRETTY[BEST]} achieves a cross-validated ROC-AUC of {pm(M['cv'][BEST]['roc_auc'])} and a hold-out ROC-AUC of {f3(tb['roc_auc'])} "
  f"(accuracy {f3(tb['accuracy'])}, recall {f3(tb['recall'])}, specificity {f3(tb['specificity'])}).")
if SYN:
    P("These numbers come from the synthetic stand-in and only demonstrate that the pipeline works. Replace them with the real-data numbers (rerun <font face='Courier'>make data all report</font>) before submitting.", WARN)
P("<b>Limitations</b>", H2)
bullets(["Small sample (303 patients) - high variance; no external validation; single-site data from the 1980s.",
         "Sample skew toward male patients limits generalisation to women; subgroup performance should be audited before any real use.",
         "The binary label merges all disease grades; the model predicts presence of angiographic disease, not future events.",
         "The decision threshold (0.5) was not optimised for a clinical cost trade-off.",
         "Monitoring detects traffic, latency and prediction-mix shifts but not label-based accuracy decay (labels are not available online).",
         "Grafana credentials in the demo manifests are placeholders; use Kubernetes Secrets, TLS and authentication for any real deployment."])
P("<b>Ethics and safety.</b> This is an educational system and <b>not a medical device</b>. Predictions must not be used for diagnosis or treatment decisions. The service never logs patient features by default, "
  "validates all inputs, and exposes the model version so every prediction is attributable.")
P("<b>Future work.</b> Data and prediction drift detection (PSI / KS tests), automated retraining and model registry promotion, canary rollouts, TLS and authentication at the Ingress, calibrated thresholds by cost analysis, and fairness metrics by sex and age group.")

# ============================================================== 14 checklist
P("14. Requirement-to-evidence checklist", H1)
table([["Task (marks)", "Evidence in the repository / report"],
       ["1. Data acquisition & EDA (5)", "scripts/download_data.py; heartml/data.py; heartml/eda.py; notebooks/01_eda.ipynb; Sections 3-4; reports/figures/eda_*.png"],
       ["2. Feature engineering & models (8)", "heartml/features.py, modeling.py, train.py; 3 models, grid search, repeated CV, accuracy/precision/recall/F1/ROC-AUC; Section 5"],
       ["3. Experiment tracking (5)", "heartml/tracking.py (MLflow); parent + nested runs with params, metrics, plots, models; Section 6; mlruns/"],
       ["4. Packaging & reproducibility (7)", "models/model.joblib + metadata; requirements*.txt; sklearn Pipeline; seeds; Section 7"],
       ["5. CI/CD & testing (8)", "tests/ (48); .github/workflows/ci.yml (lint, test, train, docker, k8s-validate); Section 8"],
       ["6. Containerisation (5)", "Dockerfile, .dockerignore, /predict returns prediction + confidence; scripts/smoke_test.py; Section 9"],
       ["7. Production deployment (7)", "k8s/*.yaml (Deployment, LoadBalancer, Ingress, HPA, kustomization); Section 10"],
       ["8. Monitoring & logging (3)", "JSON logs, /metrics, Prometheus + Grafana (compose and k8s), alerts; Section 11"],
       ["9. Documentation (2)", "README.md and this report: setup, EDA/modelling, tracking, architecture diagrams, CI/CD + deployment screenshots, repository link"]],
      widths=[5.2 * cm, W - 5.2 * cm], caption="Mapping of assignment tasks to deliverables.", font=8)
P("<b>Production-readiness requirements.</b> (1) Scripts run from a clean setup using the requirements file - yes, via Makefile targets. (2) Model serves correctly in an isolated environment - Dockerfile + smoke test (screenshots in Section 9). "
  "(3) Pipeline fails on code or test errors with clear logs - job dependencies, pipefail and non-zero smoke-test exit.")

# ============================================================== appendices
P("Appendix A - API reference", H1)
table([["Method & path", "Request", "Response"],
       ["GET /health", "-", "200 {status: ok}"],
       ["GET /ready", "-", "200 {status: ready, model_version}"],
       ["GET /model-info", "-", "model name/version, features + descriptions, hold-out and CV metrics, threshold"],
       ["POST /predict", "13-field JSON object; or list; or {instances: [...]} (max 100); ca and thal may be null",
        "200 {prediction, label, probability_disease, confidence, model_version, request_id} or {predictions: [...]}"],
       ["", "", "400 invalid JSON | 413 batch too large | 422 validation failed (details per instance)"],
       ["GET /metrics", "-", "Prometheus text format 0.0.4"]],
      widths=[3.2 * cm, 6.4 * cm, W - 9.6 * cm], caption="HTTP API.", font=8)
P("Input ranges enforced: " + "; ".join(f"{k} in [{lo}, {hi}]" for k, (lo, hi) in C.NUMERIC_RANGES.items()) + "; categorical codes: "
  + "; ".join(f"{k} in {sorted(v)}" for k, v in C.CATEGORICAL_VALUES.items()) + ".", SMALL)
P("Appendix B - command reference", H1)
code("""
make setup | data | all | test | lint | run | docker-build | docker-run | smoke | compose-up | k8s-deploy | k8s-status | mlflow-ui | diagrams | report
python scripts/load_test.py http://localhost:8000 --n 300 --rps 15      # generate traffic for Grafana
kubectl -n heart-mlops logs deploy/heart-api --tail=20                  # structured logs
kubectl -n heart-mlops port-forward svc/grafana 3000:3000               # Grafana on k8s
""")

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                        title="MLOps Assignment 01 - Heart Disease Risk Prediction", author=STUDENT)
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=page_deco)
print("wrote", OUT)
