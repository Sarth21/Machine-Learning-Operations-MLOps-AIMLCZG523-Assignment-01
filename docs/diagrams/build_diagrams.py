#!/usr/bin/env python
"""Render architecture diagrams (PNG) with Graphviz. Requires the `dot` binary on PATH."""
from pathlib import Path

import subprocess


def _q(v):
    return '"' + str(v).replace('"', '\\"').replace("\n", "\\n") + '"'


def _attrs(d):
    return ", ".join(f"{k}={_q(v)}" for k, v in d.items())


class Digraph:
    """Tiny DOT writer (keeps the project free of the `graphviz` Python package)."""

    def __init__(self, name, format="png"):
        self.name, self.nodes, self.edges, self.subs = name, [], [], []
        self.graph_attr, self.node_attr, self.edge_attr = {}, {}, {}

    def attr(self, kind=None, **kw):
        {None: self.graph_attr, "graph": self.graph_attr, "node": self.node_attr, "edge": self.edge_attr}[kind].update(kw)

    def node(self, name, label=None, **kw):
        self.nodes.append((name, {"label": label if label is not None else name, **kw}))

    def edge(self, a, b, **kw):
        self.edges.append((a, b, kw))

    def subgraph(self, sub):
        self.subs.append(sub)

    def body(self, indent="  "):
        out = [f"{indent}{k}={_q(v)};" for k, v in self.graph_attr.items()]
        if self.node_attr:
            out.append(f"{indent}node [{_attrs(self.node_attr)}];")
        if self.edge_attr:
            out.append(f"{indent}edge [{_attrs(self.edge_attr)}];")
        out += [f"{indent}{_q(n)} [{_attrs(a)}];" for n, a in self.nodes]
        out += [f"{indent}{_q(a)} -> {_q(b)} [{_attrs(k)}];" for a, b, k in self.edges]
        for s in self.subs:
            out += [f"{indent}subgraph {_q(s.name)} {{", s.body(indent + "  "), f"{indent}}}"]
        return "\n".join(out)

    def render(self, filename, cleanup=True):
        dot = f"digraph {_q(self.name)} {{\n{self.body()}\n}}\n"
        subprocess.run(["dot", "-Tpng", "-o", filename + ".png"], input=dot.encode(), check=True)

OUT = Path(__file__).resolve().parent
FONT = "Helvetica"


def base(name, rankdir="LR", **kw):
    g = Digraph(name, format="png")
    g.attr(rankdir=rankdir, fontname=FONT, dpi="170", pad="0.25", nodesep="0.35", ranksep="0.55", **kw)
    g.attr("node", fontname=FONT, fontsize="11", shape="box", style="rounded,filled", fillcolor="#FFFFFF", color="#4A5568")
    g.attr("edge", fontname=FONT, fontsize="9", color="#4A5568")
    return g


def cluster(g, name, label, color):
    c = Digraph(f"cluster_{name}")
    c.attr(label=label, style="rounded,filled", color="#CBD5E0", fillcolor=color, fontname=FONT, fontsize="12")
    return c


def system():
    g = base("system_architecture")
    c1 = cluster(g, "dev", "Developer / GitHub", "#EBF4FF")
    c1.node("dev", "Developer\n(git push / PR)")
    c1.node("repo", "GitHub repo")
    g.subgraph(c1)
    c2 = cluster(g, "ci", "GitHub Actions CI/CD", "#F0FFF4")
    for k, v in [("lint", "lint\nflake8"), ("test", "test\npytest + coverage"), ("train", "train\nEDA + MLflow"),
                 ("build", "docker build\n+ smoke test")]:
        c2.node(k, v)
    c2.node("ghcr", "GHCR\nimage registry", fillcolor="#FEFCBF")
    g.subgraph(c2)
    c3 = cluster(g, "k8s", "Kubernetes (Minikube / Docker Desktop / GKE / EKS / AKS)", "#FFFAF0")
    c3.node("lb", "LoadBalancer /\nIngress", fillcolor="#FEEBC8")
    c3.node("pods", "heart-api pods x2-5\n(Flask + gunicorn)\n/predict /metrics", fillcolor="#C6F6D5")
    c3.node("hpa", "HPA (CPU 70%)", fillcolor="#E9D8FD")
    c3.node("prom", "Prometheus", fillcolor="#FED7D7")
    c3.node("graf", "Grafana\ndashboard", fillcolor="#FED7D7")
    g.subgraph(c3)
    g.node("client", "Client / clinician app", shape="ellipse", fillcolor="#E2E8F0")
    g.node("mlflow", "MLflow tracking\n(params, metrics, plots, models)", fillcolor="#FEFCBF")
    g.edge("dev", "repo"); g.edge("repo", "lint", label="trigger")
    g.edge("lint", "test"); g.edge("test", "train"); g.edge("train", "build")
    g.edge("train", "mlflow", style="dashed", label="log runs")
    g.edge("build", "ghcr", label="push (main)"); g.edge("ghcr", "pods", label="pull", style="dashed")
    g.edge("client", "lb", label="HTTPS JSON"); g.edge("lb", "pods"); g.edge("hpa", "pods", style="dashed", label="scales")
    g.edge("prom", "pods", label="scrape /metrics"); g.edge("prom", "graf")
    g.render(str(OUT / "system_architecture"), cleanup=True)


def ml_pipeline():
    g = base("ml_pipeline", rankdir="TB")
    steps = [("raw", "UCI Heart Disease\n(download script)"), ("clean", "Clean + validate\n('?'->NaN, range checks,\nbinary target)"),
             ("eda", "EDA\nfigures + summary"), ("split", "Stratified\ntrain / test split"),
             ("fe", "Feature engineering\n+ ColumnTransformer\n(impute, scale, one-hot)"),
             ("tune", "GridSearchCV x3 models\n(LR, RF, GB) - 5-fold"), ("cv", "Repeated CV\n(5x3) of tuned models"),
             ("select", "Select best\n(CV ROC-AUC, recall)"), ("eval", "Hold-out evaluation\n+ plots"),
             ("final", "Refit on all data\nsave model.joblib\n+ metadata.json")]
    for k, v in steps:
        g.node(k, v, fillcolor="#EBF8FF")
    g.node("mlflow", "MLflow / tracker\nparams - metrics - artifacts - models", fillcolor="#FEFCBF")
    for (a, _), (b, _) in zip(steps, steps[1:]):
        g.edge(a, b)
    for k in ("tune", "cv", "eval", "final"):
        g.edge(k, "mlflow", style="dashed")
    g.render(str(OUT / "ml_pipeline"), cleanup=True)


def serving():
    g = base("serving_flow", rankdir="TB")
    g.node("req", "POST /predict\nJSON body", shape="ellipse", fillcolor="#E2E8F0")
    g.node("parse", "Parse JSON\n(400 if invalid)")
    g.node("val", "Schema validation\ntypes, ranges, categories\n(422 + details)")
    g.node("feat", "FeatureAdder -> impute\n-> scale / one-hot")
    g.node("model", "Classifier\npredict_proba", fillcolor="#C6F6D5")
    g.node("resp", "JSON response\nprediction, label,\nprobability, confidence,\nmodel_version, request_id", fillcolor="#E2E8F0")
    g.node("log", "Structured JSON log\n(request_id, latency,\nprediction; features off)", fillcolor="#FEFCBF")
    g.node("met", "Prometheus registry\ncounters + histograms\n-> GET /metrics", fillcolor="#FED7D7")
    for a, b in [("req", "parse"), ("parse", "val"), ("val", "feat"), ("feat", "model"), ("model", "resp")]:
        g.edge(a, b)
    g.edge("val", "log", style="dashed"); g.edge("model", "log", style="dashed")
    g.edge("model", "met", style="dashed"); g.edge("val", "met", style="dashed")
    g.render(str(OUT / "serving_flow"), cleanup=True)


def deployment():
    g = base("k8s_deployment", rankdir="TB")
    c = cluster(g, "ns", "namespace: heart-mlops", "#FFFAF0")
    c.node("svc", "Service heart-api\n(LoadBalancer :80)", fillcolor="#FEEBC8")
    c.node("ing", "Ingress\nheart-api.local", fillcolor="#FEEBC8")
    c.node("dep", "Deployment heart-api\nreplicas 2, rolling update\nprobes: /ready /health\nnon-root, read-only FS", fillcolor="#C6F6D5")
    c.node("p1", "Pod 1", fillcolor="#C6F6D5"); c.node("p2", "Pod 2", fillcolor="#C6F6D5")
    c.node("cm", "ConfigMap\nLOG_LEVEL, LOG_FEATURES", fillcolor="#EDF2F7")
    c.node("hl", "Headless Service\n(per-pod discovery)", fillcolor="#EDF2F7")
    c.node("hpa", "HPA 2-5 @ CPU 70%", fillcolor="#E9D8FD")
    c.node("prom", "Prometheus\nDeployment + ConfigMap", fillcolor="#FED7D7")
    c.node("graf", "Grafana\nprovisioned dashboard", fillcolor="#FED7D7")
    g.subgraph(c)
    g.node("user", "User", shape="ellipse", fillcolor="#E2E8F0")
    g.edge("user", "svc"); g.edge("user", "ing"); g.edge("ing", "svc"); g.edge("svc", "dep")
    g.edge("dep", "p1"); g.edge("dep", "p2"); g.edge("cm", "dep", style="dashed")
    g.edge("hpa", "dep", style="dashed"); g.edge("hl", "p1", style="dashed"); g.edge("hl", "p2", style="dashed")
    g.edge("prom", "hl", label="DNS SD scrape"); g.edge("prom", "graf", label="datasource")
    g.render(str(OUT / "k8s_deployment"), cleanup=True)


def cicd():
    g = base("cicd_workflow")
    for k, v, col in [("push", "push / PR\nto main", "#E2E8F0"), ("lint", "lint\nflake8", "#F0FFF4"),
                      ("test", "test\npytest + cov\nfail on error", "#F0FFF4"), ("train", "train\ndownload -> clean\n-> EDA -> train", "#F0FFF4"),
                      ("art", "upload artifacts\nmodel, metrics,\nmlruns, logs", "#FEFCBF"),
                      ("docker", "docker build\nrun container\nsmoke test", "#F0FFF4"), ("push2", "push image\nGHCR (main only)", "#FEFCBF"),
                      ("k8s", "k8s-validate\nkustomize + kubeconform", "#F0FFF4")]:
        g.node(k, v, fillcolor=col)
    for a, b in [("push", "lint"), ("lint", "test"), ("test", "train"), ("train", "art"), ("train", "docker"),
                 ("docker", "push2"), ("lint", "k8s")]:
        g.edge(a, b)
    g.render(str(OUT / "cicd_workflow"), cleanup=True)


if __name__ == "__main__":
    for fn in (system, ml_pipeline, serving, deployment, cicd):
        fn()
    print("diagrams written to", OUT)
