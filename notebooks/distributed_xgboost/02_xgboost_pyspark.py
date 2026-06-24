# Databricks notebook source

# MAGIC %md
# MAGIC # XGBoost - PySpark (Distributed)
# MAGIC Trains XGBoost via `xgboost.spark.SparkXGBRegressor` with data fully
# MAGIC distributed across Spark workers. Feature column names are passed directly
# MAGIC (no VectorAssembler). Profiles cluster-wide memory usage.

# COMMAND ----------

import time
import json
import mlflow
import mlflow.xgboost
import psutil
from pyspark.ml.evaluation import RegressionEvaluator
from xgboost.spark import SparkXGBRegressor

mlflow.set_experiment("/Shared/distributed_xgboost_profiling")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

def get_cluster_memory_metrics(sc):
    """Collect memory usage from driver and all executors via Spark status tracker."""
    driver_process = psutil.Process()
    driver_mem_gb = driver_process.memory_info().rss / (1024 ** 3)
    driver_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    driver_avail_gb = psutil.virtual_memory().available / (1024 ** 3)

    metrics = {
        "driver_rss_gb": round(driver_mem_gb, 2),
        "driver_total_gb": round(driver_total_gb, 2),
        "driver_available_gb": round(driver_avail_gb, 2),
    }

    try:
        status = sc.statusTracker()
        executor_infos = status.getExecutorInfos()
        worker_count = 0
        total_worker_mem_bytes = 0
        for info in executor_infos:
            if info.host() != "driver":
                worker_count += 1
                total_worker_mem_bytes += info.totalOnHeapStorageMemory()
        metrics["num_workers"] = worker_count
        metrics["worker_heap_total_gb"] = round(total_worker_mem_bytes / (1024 ** 3), 2)
    except Exception:
        pass

    try:
        executor_mem_status = sc._jsc.sc().getExecutorMemoryStatus()
        entries = executor_mem_status.toSeq()
        total_max = 0
        total_used = 0
        for i in range(entries.size()):
            entry = entries.apply(i)
            max_mem = entry._2()._1()
            remaining = entry._2()._2()
            total_max += max_mem
            total_used += (max_mem - remaining)
        metrics["cluster_max_mem_gb"] = round(total_max / (1024 ** 3), 2)
        metrics["cluster_used_mem_gb"] = round(total_used / (1024 ** 3), 2)
    except Exception:
        pass

    return metrics

# COMMAND ----------

start_load = time.time()
df = spark.read.table("cjc.ml.movielens_features")

feature_cols = [c for c in df.columns if c not in ("userId", "movieId", "rating")]
target_col = "rating"

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

train_count = train_df.count()
test_count = test_df.count()
load_time = time.time() - start_load

mem_pre_train = get_cluster_memory_metrics(spark.sparkContext)
print(f"Train: {train_count:,} | Test: {test_count:,} | Load time: {load_time:.1f}s")
print(f"Pre-train cluster memory: {json.dumps(mem_pre_train, indent=2)}")

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_pyspark"):
    params = dict(
        max_depth=6,
        learning_rate=0.1,
        num_round=100,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        num_workers=2,
    )

    mlflow.log_params(params)
    mlflow.log_param("approach", "pyspark_distributed")
    mlflow.log_param("num_rows_train", train_count)
    mlflow.log_param("num_rows_test", test_count)
    mlflow.log_metric("data_load_time_s", load_time)

    for k, v in mem_pre_train.items():
        mlflow.log_metric(f"pre_train_{k}", v)

    xgb_regressor = SparkXGBRegressor(
        features_col=feature_cols,
        label_col=target_col,
        **params,
    )

    start_train = time.time()
    model = xgb_regressor.fit(train_df)
    train_time = time.time() - start_train

    mem_post_train = get_cluster_memory_metrics(spark.sparkContext)
    print(f"Post-train cluster memory: {json.dumps(mem_post_train, indent=2)}")

    predictions = model.transform(test_df)

    rmse_eval = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
    mae_eval = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")
    rmse = rmse_eval.evaluate(predictions)
    mae = mae_eval.evaluate(predictions)

    mlflow.log_metric("train_time_s", train_time)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    for k, v in mem_post_train.items():
        mlflow.log_metric(f"post_train_{k}", v)

    booster = model.get_booster()
    test_sample = test_df.select(feature_cols).limit(5).toPandas()
    pred_sample = predictions.select("prediction").limit(5).toPandas().values.flatten()
    signature = mlflow.models.infer_signature(test_sample, pred_sample)

    mlflow.xgboost.log_model(
        booster,
        artifact_path="model",
        signature=signature,
        input_example=test_sample,
        registered_model_name="cjc.ml.movielens_xgboost_pyspark",
    )

    print(f"Train time: {train_time:.1f}s | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"Model registered to cjc.ml.movielens_xgboost_pyspark")

# COMMAND ----------

train_df.unpersist()
test_df.unpersist()
