"""Experiment tracking: MLflow when installed, otherwise a small file-based fallback.

Force a backend with HEART_TRACKING_BACKEND=mlflow|local.
"""
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import joblib

from heartml import config as C

log = logging.getLogger(__name__)

# Newer MLflow disables the local ./mlruns file store unless explicitly allowed.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:  # pragma: no cover
    import mlflow
    import mlflow.sklearn

    _HAS_MLFLOW = True
except Exception:  # noqa: BLE001
    mlflow = None
    _HAS_MLFLOW = False


def _clip(value):
    return str(value)[:500]


class RunHandle:
    def __init__(self, backend, run_id, name, local_dir=None):
        self.backend, self.run_id, self.name, self._dir = backend, run_id, name, local_dir
        self._params, self._metrics, self._tags = {}, {}, {}

    def log_params(self, params):
        params = {k: _clip(v) for k, v in params.items()}
        if self.backend == "mlflow":
            mlflow.log_params(params)
        self._params.update(params)

    def log_metrics(self, metrics, step=None):
        metrics = {k: float(v) for k, v in metrics.items()}
        if self.backend == "mlflow":
            mlflow.log_metrics(metrics, step=step)
        self._metrics.update(metrics)

    def set_tags(self, tags):
        tags = {k: _clip(v) for k, v in tags.items()}
        if self.backend == "mlflow":
            mlflow.set_tags(tags)
        self._tags.update(tags)

    def log_artifact(self, path, artifact_path=None):
        path = Path(path)
        if not path.exists():
            return
        if self.backend == "mlflow":
            mlflow.log_artifact(str(path), artifact_path)
        else:
            dest = self._dir / "artifacts" / (artifact_path or "")
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)

    def log_model(self, model, name="model", input_example=None):
        if self.backend == "mlflow":
            # Newer MLflow saves sklearn models with skops, which rejects custom classes
            # (heartml.features.FeatureAdder) unless trusted; try several serialisation options.
            attempts = [
                {"name": name, "serialization_format": "cloudpickle"},
                {"name": name, "skops_trusted_types": ["heartml.features.FeatureAdder", "numpy.dtype"]},
                {"name": name},
                {"artifact_path": name},  # older MLflow
            ]
            last = None
            for kwargs in attempts:
                try:
                    mlflow.sklearn.log_model(sk_model=model, input_example=input_example, **kwargs)
                    return
                except Exception as exc:  # noqa: BLE001 - try the next serialisation option
                    last = exc
            log.warning("Could not log model '%s' to MLflow: %s", name, last)
        else:
            dest = self._dir / "artifacts" / name
            dest.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, dest / "model.joblib")

    def _flush_local(self):
        if self.backend != "local":
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        for fname, obj in (("params.json", self._params), ("metrics.json", self._metrics),
                           ("tags.json", self._tags)):
            (self._dir / fname).write_text(json.dumps(obj, indent=2, sort_keys=True))


class Tracker:
    def __init__(self, experiment=C.EXPERIMENT_NAME, tracking_uri=None, backend=None):
        forced = backend or os.getenv("HEART_TRACKING_BACKEND")
        self.backend = forced or ("mlflow" if _HAS_MLFLOW else "local")
        if self.backend == "mlflow" and not _HAS_MLFLOW:
            raise RuntimeError("HEART_TRACKING_BACKEND=mlflow but mlflow is not installed")
        self.experiment = experiment
        self._stack = []
        if self.backend == "mlflow":
            uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or C.MLRUNS_DIR.as_uri()
            try:
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(experiment)
                self.uri = uri
            except Exception as exc:  # noqa: BLE001
                if forced == "mlflow":
                    raise
                log.warning("MLflow tracking store unavailable (%s); using the local file tracker", exc)
                self.backend = "local"
        if self.backend == "local":
            self.root = Path(tracking_uri or (C.ROOT / "mlruns_local")) / experiment
            self.root.mkdir(parents=True, exist_ok=True)
            self.uri = str(self.root)

    @contextmanager
    def run(self, name, nested=False, tags=None):
        if self.backend == "mlflow":
            with mlflow.start_run(run_name=name, nested=nested) as active:
                handle = RunHandle("mlflow", active.info.run_id, name)
                if tags:
                    handle.set_tags(tags)
                yield handle
            return
        run_id = uuid.uuid4().hex[:16]
        parent = self._stack[-1] if (nested and self._stack) else None
        base = (parent._dir / "children") if parent else self.root
        handle = RunHandle("local", run_id, name, base / f"{int(time.time())}_{name}_{run_id[:6]}")
        handle.set_tags({"run_name": name, **(tags or {}),
                         **({"parent_run_id": parent.run_id} if parent else {})})
        self._stack.append(handle)
        try:
            yield handle
        finally:
            self._stack.pop()
            handle._flush_local()
