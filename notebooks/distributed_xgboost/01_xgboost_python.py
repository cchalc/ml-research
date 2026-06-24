# Databricks notebook source

# MAGIC %md
# MAGIC # XGBoost - Python (Single-Node)
# MAGIC Collects the entire dataset to the driver via `toPandas()` and trains
# MAGIC a single-node XGBoost regressor. Expected to OOM on memory-constrained nodes.

# COMMAND ----------

import time
import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

mlflow.set_experiment("/Shared/distributed_xgboost_profiling")
mlflow.set_registry_uri("databricks-uc")
process = psutil.Process()

def mem_gb():
    return process.memory_info().rss / (1024 ** 3)

# COMMAND ----------

mem_start = mem_gb()
start_load = time.time()

df = spark.read.table("cjc.ml.movielens_features")
feature_cols = [c for c in df.columns if c not in ("userId", "movieId", "rating")]
target_col = "rating"

pdf = df.select(feature_cols + [target_col]).toPandas()
load_time = time.time() - start_load
mem_after_load = mem_gb()

print(f"Loaded {len(pdf):,} rows x {len(feature_cols)} features in {load_time:.1f}s")
print(f"Driver memory: {mem_start:.1f} GB -> {mem_after_load:.1f} GB")

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    pdf[feature_cols].values, pdf[target_col].values,
    test_size=0.2, random_state=42,
)
del pdf
mem_after_split = mem_gb()
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Memory: {mem_after_split:.1f} GB")

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_python"):
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    num_rounds = 100

    mlflow.log_params(params)
    mlflow.log_param("num_boost_round", num_rounds)
    mlflow.log_param("approach", "python_single_node")
    mlflow.log_param("num_rows", X_train.shape[0] + X_test.shape[0])
    mlflow.log_metric("data_load_time_s", load_time)
    mlflow.log_metric("driver_mem_start_gb", mem_start)
    mlflow.log_metric("driver_mem_after_load_gb", mem_after_load)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    mem_after_dmatrix = mem_gb()
    mlflow.log_metric("driver_mem_after_dmatrix_gb", mem_after_dmatrix)

    start_train = time.time()
    model = xgb.train(
        params, dtrain, num_boost_round=num_rounds,
        evals=[(dtest, "test")], verbose_eval=20,
    )
    train_time = time.time() - start_train
    mem_after_train = mem_gb()

    preds = model.predict(dtest)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))

    mlflow.log_metric("train_time_s", train_time)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("driver_mem_after_train_gb", mem_after_train)
    mlflow.log_metric("driver_mem_peak_gb", max(mem_after_load, mem_after_dmatrix, mem_after_train))

    signature = mlflow.models.infer_signature(X_test[:5], preds[:5])
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_test[:5],
        registered_model_name="cjc.ml.movielens_xgboost_python",
    )

    print(f"Train: {train_time:.1f}s | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"Peak driver memory: {max(mem_after_load, mem_after_dmatrix, mem_after_train):.1f} GB")
    print(f"Model registered to cjc.ml.movielens_xgboost_python")
