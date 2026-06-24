# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Fraud Detection — Structured Streaming Inference (Live Data)
# MAGIC The third **inference** pattern, and the headline of the demo: scoring a stream
# MAGIC of transactions with the champion model and writing alerts to
# MAGIC `cjc.ml.fraud_scored_stream`.
# MAGIC
# MAGIC Two run modes, controlled by `STREAM_MODE`:
# MAGIC
# MAGIC | Mode | Source & trigger | Compute |
# MAGIC |---|---|---|
# MAGIC | `serverless` | Delta landing table + `Trigger.availableNow` (bounded) | **Serverless** / shared — easiest to demo |
# MAGIC | `classic` | `rate` source + continuous `processingTime` trigger | Classic **Single-User** ML cluster |
# MAGIC
# MAGIC Serverless and shared (Spark Connect) compute **do not support** continuous
# MAGIC `processingTime` triggers — they require `availableNow`/`Once`. So for a quick,
# MAGIC click-and-run demo, leave `STREAM_MODE = "serverless"`. For a truly always-on
# MAGIC live stream, use `classic` on a Single-User ML cluster.
# MAGIC
# MAGIC Swap the source for Kafka / Auto Loader / Zerobus and this exact pipeline scores
# MAGIC real production traffic.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

CATALOG = "cjc"
SCHEMA = "ml"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_xgboost"
MODEL_URI = f"models:/{UC_MODEL_NAME}@champion"
SCORED_TABLE = f"{CATALOG}.{SCHEMA}.fraud_scored_stream"
LANDING_TABLE = f"{CATALOG}.{SCHEMA}.fraud_transactions_landing"
CHECKPOINT = f"/tmp/fraud_detection/checkpoints/{SCORED_TABLE}"

FEATURES = [
    "amount", "amount_to_avg_ratio", "distance_from_home_km", "num_tx_last_hour",
    "hour_of_day", "merchant_risk", "is_foreign", "card_present", "account_age_days",
]
ALERT_THRESHOLD = 0.50

# --- Run mode -------------------------------------------------------------
STREAM_MODE = "serverless"   # "serverless" (availableNow) or "classic" (continuous)
ROWS_PER_SECOND = 2          # classic mode: ~one new transaction every 0.5s
RUN_SECONDS = 90             # classic mode: bounded window, then stop (None = forever)
N_LANDING = 100             # serverless mode: synthetic transactions staged per run
# --------------------------------------------------------------------------

mlflow.set_registry_uri("databricks-uc")
print(f"STREAM_MODE = {STREAM_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthesize transaction features
# MAGIC One helper turns a base row (an id + a timestamp) into a realistic synthetic
# MAGIC transaction with exactly the features the model was trained on. Both run modes
# MAGIC reuse it, so the feature logic lives in a single place.

# COMMAND ----------

def add_transaction_features(base):
    """base must have columns `tx_id` (long) and `tx_time` (timestamp)."""
    return (
        base
        .withColumn("transaction_id", F.col("tx_id"))
        .withColumn("event_time", F.col("tx_time"))
        .withColumn("customer_id", (F.col("tx_id") % F.lit(20000)).cast("int"))
        .withColumn("amount", F.round(F.exp(F.lit(3.5) + F.lit(1.1) * F.randn()), 2))
        .withColumn("amount_to_avg_ratio", F.round(F.exp(F.lit(0.6) * F.randn()), 3))
        .withColumn("distance_from_home_km", F.round(-F.lit(12.0) * F.log(F.rand() + F.lit(1e-9)), 1))
        .withColumn("num_tx_last_hour", F.floor(F.rand() * F.lit(6)).cast("int"))
        .withColumn("hour_of_day", F.floor(F.rand() * F.lit(24)).cast("int"))
        .withColumn("merchant_risk", F.round(F.rand(), 3))
        .withColumn("is_foreign", (F.rand() < F.lit(0.06)).cast("int"))
        .withColumn("card_present", (F.rand() < F.lit(0.72)).cast("int"))
        .withColumn("account_age_days", (F.floor(F.rand() * F.lit(3620)) + F.lit(30)).cast("int"))
        .select("transaction_id", "event_time", "customer_id", *FEATURES)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the transaction stream for the selected mode
# MAGIC - **serverless**: append a fresh batch of synthetic transactions to a Delta
# MAGIC   landing table, then read that table as a stream. A Delta source works with
# MAGIC   `Trigger.availableNow` on serverless.
# MAGIC - **classic**: read Spark's built-in `rate` source as a continuous generator.

# COMMAND ----------

if STREAM_MODE == "serverless":
    # Stage a fresh batch of transactions so each run has new rows to score.
    landing_base = (
        spark.range(N_LANDING)
        .withColumnRenamed("id", "tx_id")
        .withColumn("tx_time", F.current_timestamp())
    )
    add_transaction_features(landing_base).write.mode("append").saveAsTable(LANDING_TABLE)
    print(f"Staged {N_LANDING:,} transactions into {LANDING_TABLE}")
    transactions = spark.readStream.table(LANDING_TABLE)
else:
    raw = (
        spark.readStream.format("rate")
        .option("rowsPerSecond", ROWS_PER_SECOND)
        .load()
        .withColumnRenamed("value", "tx_id")
        .withColumnRenamed("timestamp", "tx_time")
    )
    transactions = add_transaction_features(raw)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the stream with the champion model
# MAGIC The same `spark_udf` used in batch — one model, every inference pattern. On
# MAGIC serverless we use `env_manager="virtualenv"` so the model's dependencies
# MAGIC (xgboost) are restored into the UDF environment; on a classic ML cluster they're
# MAGIC already installed.

# COMMAND ----------

env_manager = "virtualenv" if STREAM_MODE == "serverless" else "local"
score_udf = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double", env_manager=env_manager)

scored = (
    transactions
    .withColumn("fraud_probability", score_udf(*[F.col(c) for c in FEATURES]))
    .withColumn("fraud_alert", (F.col("fraud_probability") >= F.lit(ALERT_THRESHOLD)).cast("int"))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the scored stream to Delta
# MAGIC A checkpoint makes the stream exactly-once and restartable.

# COMMAND ----------

writer = (
    scored.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT)
)

if STREAM_MODE == "serverless":
    # Trigger.availableNow: process all staged transactions in micro-batches, then
    # stop on its own. Supported on serverless / shared compute.
    query = writer.trigger(availableNow=True).toTable(SCORED_TABLE)
    query.awaitTermination()
    print("availableNow run complete — all staged transactions scored.")
else:
    # Continuous micro-batch streaming. Needs a classic Single-User ML cluster.
    query = writer.trigger(processingTime="2 seconds").toTable(SCORED_TABLE)
    print(f"Streaming live transactions into {SCORED_TABLE} (query id: {query.id})")
    if RUN_SECONDS:
        query.awaitTermination(RUN_SECONDS)
        query.stop()
        print(f"Stopped after {RUN_SECONDS}s.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Watch the alerts arrive
# MAGIC Freshly scored transactions and the high-risk alerts flagged from the stream.
# MAGIC (In classic continuous mode, re-run these cells to see new rows arrive.)

# COMMAND ----------

display(spark.sql(f"""
    SELECT
      COUNT(*)                                   AS total_scored,
      SUM(fraud_alert)                           AS alerts,
      ROUND(AVG(fraud_probability), 4)           AS avg_score,
      MAX(scored_at)                             AS latest_score_time
    FROM {SCORED_TABLE}
"""))

# COMMAND ----------

display(spark.sql(f"""
    SELECT transaction_id, customer_id, amount, distance_from_home_km,
           is_foreign, card_present, fraud_probability, scored_at
    FROM {SCORED_TABLE}
    WHERE fraud_alert = 1
    ORDER BY scored_at DESC
    LIMIT 15
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop the stream when you're done
# MAGIC (Classic continuous mode only — `availableNow` stops itself.)
# MAGIC ```python
# MAGIC for q in spark.streams.active:
# MAGIC     q.stop()
# MAGIC ```
