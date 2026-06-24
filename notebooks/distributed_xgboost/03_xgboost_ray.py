# Databricks notebook source

# MAGIC %md
# MAGIC # XGBoost - Ray (Distributed)
# MAGIC Uses Ray Train's XGBoostTrainer for distributed data-parallel XGBoost
# MAGIC training on Spark workers. Data is written as Parquet via Spark and read
# MAGIC by Ray workers. Profiles Ray cluster memory.

# COMMAND ----------

import time
import json
import mlflow
import mlflow.xgboost
import xgboost as xgb
import psutil
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

mlflow.set_experiment("/Shared/distributed_xgboost_profiling")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

def get_ray_cluster_memory():
    """Collect memory usage across all Ray nodes."""
    nodes = ray.nodes()
    metrics = {"num_ray_nodes": len(nodes)}
    total_mem = 0
    total_used = 0
    total_obj_store = 0
    total_obj_used = 0
    for i, node in enumerate(nodes):
        res = node.get("Resources", {})
        mem = res.get("memory", 0)
        obj = res.get("object_store_memory", 0)
        total_mem += mem
        total_obj_store += obj

        avail_res = node.get("AvailableResources", {})
        avail_mem = avail_res.get("memory", 0)
        avail_obj = avail_res.get("object_store_memory", 0)
        total_used += (mem - avail_mem)
        total_obj_used += (obj - avail_obj)

        metrics[f"node_{i}_mem_total_gb"] = round(mem / (1024 ** 3), 2)
        metrics[f"node_{i}_mem_used_gb"] = round((mem - avail_mem) / (1024 ** 3), 2)

    metrics["cluster_mem_total_gb"] = round(total_mem / (1024 ** 3), 2)
    metrics["cluster_mem_used_gb"] = round(total_used / (1024 ** 3), 2)
    metrics["cluster_obj_store_total_gb"] = round(total_obj_store / (1024 ** 3), 2)
    metrics["cluster_obj_store_used_gb"] = round(total_obj_used / (1024 ** 3), 2)

    driver_rss = psutil.Process().memory_info().rss / (1024 ** 3)
    metrics["driver_rss_gb"] = round(driver_rss, 2)
    return metrics

# COMMAND ----------

print("Setting up Ray cluster on Spark workers...")
t0 = time.time()
setup_ray_cluster(
    min_worker_nodes=2,
    max_worker_nodes=2,
    num_cpus_worker_node=2,
    num_cpus_head_node=0,
    num_gpus_worker_node=0,
    num_gpus_head_node=0,
)
ray.init(ignore_reinit_error=True)
print(f"Ray cluster ready in {time.time()-t0:.1f}s")
print(f"Ray version: {ray.__version__}")
print(f"Ray cluster resources: {ray.cluster_resources()}")

# COMMAND ----------

start_load = time.time()
print("Reading feature table from Delta...")

df = spark.read.table("cjc.ml.movielens_features")
feature_cols = [c for c in df.columns if c not in ("userId", "movieId", "rating")]
target_col = "rating"

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Writing train/test parquet...")
train_path = "dbfs:/tmp/movielens_ray_train"
test_path = "dbfs:/tmp/movielens_ray_test"
dbutils.fs.rm(train_path, recurse=True)
dbutils.fs.rm(test_path, recurse=True)
train_df.select(feature_cols + [target_col]).coalesce(4).write.mode("overwrite").parquet(train_path)
test_df.select(feature_cols + [target_col]).coalesce(4).write.mode("overwrite").parquet(test_path)

train_count = train_df.count()
test_count = test_df.count()
print(f"Wrote {train_count:,} train / {test_count:,} test rows as parquet")

# COMMAND ----------

print("Loading parquet into Ray datasets (materializing)...")
t1 = time.time()
train_ds = ray.data.read_parquet("/dbfs/tmp/movielens_ray_train/").materialize()
print(f"Train dataset materialized in {time.time()-t1:.1f}s")
t2 = time.time()
test_ds = ray.data.read_parquet("/dbfs/tmp/movielens_ray_test/").materialize()
print(f"Test dataset materialized in {time.time()-t2:.1f}s")
load_time = time.time() - start_load
print(f"Total data load time: {load_time:.1f}s")

mem_pre_train = get_ray_cluster_memory()
print(f"Pre-train Ray cluster memory: {json.dumps(mem_pre_train, indent=2)}")

# COMMAND ----------

from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

print("Starting Ray XGBoost training...")
with mlflow.start_run(run_name="xgboost_ray"):
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": ["rmse", "mae"],
    }
    num_rounds = 100

    mlflow.log_params(params)
    mlflow.log_param("num_boost_round", num_rounds)
    mlflow.log_param("approach", "ray_distributed")
    mlflow.log_param("num_rows_train", train_count)
    mlflow.log_param("num_rows_test", test_count)
    mlflow.log_metric("data_load_time_s", load_time)
    for k, v in mem_pre_train.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"pre_train_{k}", v)

    start_train = time.time()
    trainer = XGBoostTrainer(
        label_column=target_col,
        params=params,
        datasets={"train": train_ds, "validation": test_ds},
        scaling_config=ScalingConfig(
            num_workers=2,
            resources_per_worker={"CPU": 1},
        ),
        run_config=RunConfig(
            verbose=2,
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            storage_path="/dbfs/tmp/ray_results",
        ),
        num_boost_round=num_rounds,
    )
    result = trainer.fit()
    train_time = time.time() - start_train
    print(f"Training complete in {train_time:.1f}s")

    mem_post_train = get_ray_cluster_memory()
    print(f"Post-train Ray cluster memory: {json.dumps(mem_post_train, indent=2)}")

    metrics = result.metrics or {}
    rmse = metrics.get("validation-rmse", metrics.get("train-rmse"))
    mae = metrics.get("validation-mae", metrics.get("train-mae"))
    if rmse is not None:
        mlflow.log_metric("rmse", rmse)
    if mae is not None:
        mlflow.log_metric("mae", mae)

    mlflow.log_metric("train_time_s", train_time)
    for k, v in mem_post_train.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"post_train_{k}", v)

    checkpoint = result.checkpoint
    if checkpoint:
        bst = XGBoostTrainer.get_model(checkpoint)
        import pandas as pd
        test_sample = pd.read_parquet("/dbfs/tmp/movielens_ray_test/").drop(columns=[target_col]).head(5)
        pred_sample = bst.predict(xgb.DMatrix(test_sample))
        signature = mlflow.models.infer_signature(test_sample, pred_sample)

        mlflow.xgboost.log_model(
            bst,
            artifact_path="model",
            signature=signature,
            input_example=test_sample,
            registered_model_name="cjc.ml.movielens_xgboost_ray",
        )
        print("Model registered to cjc.ml.movielens_xgboost_ray")

    print(f"Train time: {train_time:.1f}s")
    print(f"Result metrics: {metrics}")

# COMMAND ----------

shutdown_ray_cluster()
