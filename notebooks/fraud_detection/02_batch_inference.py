# Databricks notebook source

# MAGIC %md
# MAGIC # Fraud Detection — Batch Inference
# MAGIC The first of three **inference** patterns. We load the registered champion model
# MAGIC as a Spark UDF and score a full table of transactions in parallel, writing the
# MAGIC results to `cjc.ml.fraud_scored_batch`.
# MAGIC
# MAGIC This is how you'd nightly-score a day's worth of settled transactions, or
# MAGIC backfill scores for an analytics dashboard.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

CATALOG = "cjc"
SCHEMA = "ml"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.fraud_transactions"
SCORED_TABLE = f"{CATALOG}.{SCHEMA}.fraud_scored_batch"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_xgboost"
MODEL_URI = f"models:/{UC_MODEL_NAME}@champion"

FEATURES = [
    "amount", "amount_to_avg_ratio", "distance_from_home_km", "num_tx_last_hour",
    "hour_of_day", "merchant_risk", "is_foreign", "card_present", "account_age_days",
]
ALERT_THRESHOLD = 0.50  # transactions at/above this score get sent to a reviewer

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model as a Spark UDF
# MAGIC `spark_udf` distributes the model to every executor — no single-node bottleneck,
# MAGIC scales to billions of rows.

# COMMAND ----------

score_udf = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double")
print("Loaded champion model as a Spark UDF")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the table

# COMMAND ----------

scored = (
    spark.read.table(SOURCE_TABLE)
    .withColumn("fraud_probability", score_udf(*[F.col(c) for c in FEATURES]))
    .withColumn("fraud_alert", (F.col("fraud_probability") >= F.lit(ALERT_THRESHOLD)).cast("int"))
    .withColumn("scored_at", F.current_timestamp())
)

(
    scored.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SCORED_TABLE)
)
print(f"Wrote scored transactions to {SCORED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the results
# MAGIC How many transactions would be flagged, and how well do the scores separate
# MAGIC actual fraud from legitimate activity?

# COMMAND ----------

display(spark.sql(f"""
    SELECT
      fraud_alert,
      COUNT(*)                         AS n_transactions,
      ROUND(AVG(fraud_probability), 4) AS avg_score,
      SUM(is_fraud)                    AS actual_fraud_caught
    FROM {SCORED_TABLE}
    GROUP BY fraud_alert
    ORDER BY fraud_alert DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Highest-risk transactions for analyst review

# COMMAND ----------

display(spark.sql(f"""
    SELECT transaction_id, customer_id, amount, distance_from_home_km,
           merchant_risk, is_foreign, fraud_probability, is_fraud
    FROM {SCORED_TABLE}
    ORDER BY fraud_probability DESC
    LIMIT 20
"""))
