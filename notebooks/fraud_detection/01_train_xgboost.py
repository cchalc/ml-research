# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "1"
# ///
# MAGIC %md
# MAGIC # Fraud Detection — Training & Experimentation with MLflow + XGBoost
# MAGIC This is the **training → experimentation → registry** stage.
# MAGIC
# MAGIC We run **three MLflow experiments** against `cjc.ml.fraud_transactions`:
# MAGIC 1. `baseline` — shallow XGBoost, no class weighting
# MAGIC 2. `tuned` — deeper trees, more rounds, `scale_pos_weight` for the class imbalance
# MAGIC 3. `tuned_regularized` — adds L1/L2 regularization and subsampling
# MAGIC
# MAGIC All runs log to the same experiment so they're directly comparable in the MLflow
# MAGIC UI. We pick the best run by validation **PR-AUC** (the right metric for rare-event
# MAGIC fraud), wrap its booster in a `FraudScorer` pyfunc that returns a calibrated
# MAGIC **fraud probability**, and register it to Unity Catalog as `cjc.ml.fraud_xgboost`
# MAGIC with a `champion` alias.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a custom package from a `.whl` file
# MAGIC Likely the fastest legacy-compatibility pattern — CI/CD ships the `.whl` to a
# MAGIC Volume, then it's available to any notebook/repo with Unity Catalog access.

# COMMAND ----------

# MAGIC %uv pip install /Volumes/cjc/ml/whl/mlflow_lens-0.2.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow_lens.classifier import roc_auc, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

CATALOG = "cjc"
SCHEMA = "ml"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.fraud_transactions"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_xgboost"
EXPERIMENT = "/Shared/fraud_detection"

FEATURES = [
    "amount", "amount_to_avg_ratio", "distance_from_home_km", "num_tx_last_hour",
    "hour_of_day", "merchant_risk", "is_foreign", "card_present", "account_age_days",
]
TARGET = "is_fraud"

mlflow.set_experiment(EXPERIMENT)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data and make a time-based split
# MAGIC We train on the oldest 80% of transactions and validate on the most recent 20% —
# MAGIC the honest way to evaluate a model that will score *future* transactions.

# COMMAND ----------

pdf = (
    spark.read.table(SOURCE_TABLE)
    .select("event_time", *FEATURES, TARGET)
    .toPandas()
    .sort_values("event_time")
    .reset_index(drop=True)
)

split = int(len(pdf) * 0.8)
train_pdf, valid_pdf = pdf.iloc[:split], pdf.iloc[split:]

X_train, y_train = train_pdf[FEATURES], train_pdf[TARGET]
X_valid, y_valid = valid_pdf[FEATURES], valid_pdf[TARGET]

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
print(f"Train fraud rate: {y_train.mean():.2%}, scale_pos_weight: {scale_pos_weight:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log multiple experiments
# MAGIC Each config is one MLflow run. We log params, the full metric set, and the
# MAGIC native XGBoost model so every run is reproducible from the registry.

# COMMAND ----------

def evaluate(booster, dmatrix, y_true):
    proba = booster.predict(dmatrix)
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


EXPERIMENTS = {
    "baseline": {
        "params": {"objective": "binary:logistic", "eval_metric": "aucpr",
                   "max_depth": 4, "eta": 0.2, "tree_method": "hist"},
        "num_boost_round": 80,
    },
    "tuned": {
        "params": {"objective": "binary:logistic", "eval_metric": "aucpr",
                   "max_depth": 7, "eta": 0.1, "tree_method": "hist",
                   "scale_pos_weight": scale_pos_weight},
        "num_boost_round": 300,
    },
    "tuned_regularized": {
        "params": {"objective": "binary:logistic", "eval_metric": "aucpr",
                   "max_depth": 6, "eta": 0.08, "tree_method": "hist",
                   "scale_pos_weight": scale_pos_weight,
                   "subsample": 0.8, "colsample_bytree": 0.8,
                   "reg_alpha": 0.5, "reg_lambda": 2.0},
        "num_boost_round": 400,
    },
}

runs = {}
for name, cfg in EXPERIMENTS.items():
    with mlflow.start_run(run_name=name) as run:
        booster = xgb.train(
            cfg["params"], dtrain,
            num_boost_round=cfg["num_boost_round"],
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        metrics = evaluate(booster, dvalid, y_valid)

        # Log a ROC curve via the custom mlflow_lens package (.whl loaded above)
        y_proba = booster.predict(xgb.DMatrix(X_valid))
        roc_auc.from_scores(y_valid, y_proba, log=True)

        mlflow.log_params(cfg["params"])
        mlflow.log_param("num_boost_round", cfg["num_boost_round"])
        mlflow.log_param("best_iteration", booster.best_iteration)
        mlflow.log_metrics(metrics)
        example = X_valid.head(5)
        sig = infer_signature(example, booster.predict(xgb.DMatrix(example)))
        mlflow.xgboost.log_model(booster, artifact_path="model", signature=sig, input_example=example)

        runs[name] = {"run_id": run.info.run_id, "booster": booster, "metrics": metrics}
        print(f"{name:18s}  AUC={metrics['auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
              f"recall={metrics['recall']:.3f}  precision={metrics['precision']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare runs and pick the champion
# MAGIC PR-AUC is the deciding metric — for a ~2% fraud rate it reflects real ranking
# MAGIC quality far better than accuracy or ROC-AUC.

# COMMAND ----------

import pandas as pd
leaderboard = pd.DataFrame({n: r["metrics"] for n, r in runs.items()}).T
leaderboard = leaderboard.sort_values("pr_auc", ascending=False)
print(leaderboard)

best_name = leaderboard.index[0]
best = runs[best_name]
print(f"\nChampion: {best_name}  (run_id={best['run_id']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap the champion as a probability-scoring pyfunc and register to UC
# MAGIC Production fraud systems consume a **risk score**, not a hard 0/1 label. We wrap
# MAGIC the booster in a small `FraudScorer` so batch, streaming, and the serving
# MAGIC endpoint all return a `fraud_probability` with one consistent interface.

# COMMAND ----------

class FraudScorer(mlflow.pyfunc.PythonModel):
    """Returns the fraud probability for each transaction row."""

    def __init__(self, features):
        self.features = features

    def load_context(self, context):
        import xgboost as xgb
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts["xgb_model"])

    def predict(self, context, model_input):
        import xgboost as xgb
        X = model_input[self.features]
        return self.booster.predict(xgb.DMatrix(X))


# Persist the champion booster to a file to ship as a pyfunc artifact.
model_path = "/tmp/fraud_champion.json"
best["booster"].save_model(model_path)

input_example = X_valid.head(5)
signature = infer_signature(input_example, best["booster"].predict(xgb.DMatrix(input_example)))

with mlflow.start_run(run_name=f"register_{best_name}"):
    mlflow.log_param("champion_run_id", best["run_id"])
    mlflow.log_metrics({f"champion_{k}": v for k, v in best["metrics"].items()})
    logged = mlflow.pyfunc.log_model(
        artifact_path="fraud_scorer",
        python_model=FraudScorer(FEATURES),
        artifacts={"xgb_model": model_path},
        signature=signature,
        input_example=input_example,
        pip_requirements=["xgboost", "pandas", "scikit-learn"],
        registered_model_name=UC_MODEL_NAME,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote to the `champion` alias
# MAGIC Downstream notebooks load `models:/cjc.ml.fraud_xgboost@champion`, so promoting a
# MAGIC new version is a one-line change with no edits to the consumers.

# COMMAND ----------

client = MlflowClient()
version = max(int(v.version) for v in client.search_model_versions(f"name='{UC_MODEL_NAME}'"))
client.set_registered_model_alias(UC_MODEL_NAME, "champion", version)
print(f"Registered {UC_MODEL_NAME} v{version} and set @champion")
