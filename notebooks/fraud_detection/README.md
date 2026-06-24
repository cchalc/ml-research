# Fraud Detection — Classic ML End-to-End Demo

A simple, compelling walkthrough of the full classic-ML lifecycle on Databricks for a
financial use case: **credit-card fraud detection** with XGBoost and MLflow.

```
data  →  training  →  experimentation  →  registry  →  serving  →  inference  →  monitoring
```

Everything is synthetic and self-contained — no external data, no external services —
so it runs anywhere with zero setup.

## The notebooks

| Notebook | Stage | What it does |
|---|---|---|
| `00_data_prep.py` | **Data** | Generates 250K synthetic card transactions with realistic features (amount-vs-normal-spend, distance from home, velocity, merchant risk, foreign/card-present flags) and a `is_fraud` label. Writes `cjc.ml.fraud_transactions`. |
| `01_train_xgboost.py` | **Training + Experimentation + Registry** | Runs **3 MLflow experiments** (baseline, tuned, tuned+regularized XGBoost), compares them on **PR-AUC**, then wraps the best booster in a `FraudScorer` pyfunc that returns a fraud probability and registers it to Unity Catalog as `cjc.ml.fraud_xgboost@champion`. |
| `02_batch_inference.py` | **Inference (batch)** | Loads the champion as a Spark UDF and scores the whole table in parallel → `cjc.ml.fraud_scored_batch`. |
| `03_model_serving.py` | **Serving (real-time)** | Deploys the champion to a serverless **Model Serving** endpoint (`fraud_detection_demo`) and queries it over REST. |
| `04_streaming_inference.py` | **Inference (streaming)** | Uses Spark's `rate` source as a **live** transaction generator (~2/sec), scores each one with the same model, and streams alerts into `cjc.ml.fraud_scored_stream`. |
| `05_monitoring.py` | **Monitoring** | Attaches a Unity Catalog **Lakehouse Monitor** to the scored stream — auto-profiles score/alert-rate metrics and tracks drift over time, with a generated dashboard. |

One model, three inference patterns (batch, real-time REST, streaming) — all loading the
same `models:/cjc.ml.fraud_xgboost@champion` URI.

### Verified live (run on Azure workspace `adb-984752964297111`)

- **Experiments:** 3 runs logged to `/Shared/fraud_detection`. The calibrated `baseline`
  won on PR-AUC (**AUC 0.986, PR-AUC 0.736**); the class-weighted `tuned_regularized`
  traded precision for **0.95 recall** — a useful threshold/cost tradeoff to discuss.
- **Batch:** 250K transactions scored → at a 0.5 cutoff, **56% recall at 81% precision**
  (fraud avg score 0.56 vs 0.01 for legit).
- **Streaming:** ~2 transactions/sec scored continuously into `fraud_scored_stream`,
  alerts firing in real time.
- **Monitoring:** a Lakehouse Monitor on `fraud_scored_stream` writes
  `..._profile_metrics` + `..._drift_metrics` tables and a dashboard, tracking score
  distribution and alert-rate drift across refreshes.

> The streaming notebook has two modes via `STREAM_MODE`:
> - `serverless` (default) — stages transactions to a Delta landing table and scores
>   them with `Trigger.availableNow`. Runs on **serverless/shared** compute, easiest to
>   demo (click Run All).
> - `classic` — Spark `rate` source with a continuous `processingTime` trigger for an
>   always-on live stream. Needs a classic **Single-User** ML cluster (serverless/shared
>   compute can't run continuous triggers). Bounded by `RUN_SECONDS`; set to `None` for
>   always-on, and use the continuous job in `resources/fraud_detection_jobs.yml`.

## Run it

**Interactively:** open the notebooks `00 → 04` in order on an ML runtime cluster.

**As a job (Asset Bundle):** the pipeline and the streaming job are defined in
`resources/fraud_detection_jobs.yml`.

```bash
databricks bundle deploy -t dev
databricks bundle run fraud_detection_pipeline -t dev   # data → train → batch
# the streaming job is created PAUSED; unpause it in the UI or:
databricks bundle run fraud_detection_streaming -t dev
```

## Config

All notebooks share these constants (edit at the top of each):

- Catalog / schema: `cjc.ml`
- Experiment: `/Shared/fraud_detection`
- UC model: `cjc.ml.fraud_xgboost` (alias `@champion`)
- Serving endpoint: `fraud_detection_demo`

## Cleanup

```python
# Stop streaming
for q in spark.streams.active: q.stop()
# Delete the serving endpoint (stops cost)
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
w.serving_endpoints.delete(name="fraud_detection_demo")
# Delete the monitor
w.quality_monitors.delete(table_name="cjc.ml.fraud_scored_stream")
```
