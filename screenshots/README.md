# Screenshots to capture (needs Docker / Kubernetes / GitHub / a browser)

These cannot be produced offline. Save PNGs here with these exact names, then run `make report`
- `reports/build_report.py` embeds each one automatically (missing ones show an orange placeholder box).

| File name | What to capture |
|---|---|
| mlflow_runs_list | MLflow UI experiment page (parent run + 3 nested runs) |
| mlflow_run_detail | One run with params, metrics, artifacts/plots |
| mlflow_compare | Compare view of the 3 model runs |
| github_actions_run | Successful workflow run, all jobs green |
| github_actions_artifacts | Artifacts section of that run |
| github_actions_failure_demo | (optional) failing run with clear error |
| docker_build | `docker build -t heart-disease-api:1.0.0 .` finishing |
| docker_run_predict | `docker run -p 8000:8000 ...` + curl /predict |
| docker_smoke_test | `python scripts/smoke_test.py http://localhost:8000` all PASS |
| k8s_get_pods | `kubectl -n heart-mlops get pods,svc,hpa` |
| k8s_rollout | `kubectl rollout status` / describe |
| k8s_predict_via_lb | curl /predict via LoadBalancer or Ingress |
| grafana_dashboard | Dashboard after `python scripts/load_test.py` |
| prometheus_targets | Prometheus Status > Targets (UP) |
| api_logs | `kubectl logs` / `docker logs` showing JSON logs |
