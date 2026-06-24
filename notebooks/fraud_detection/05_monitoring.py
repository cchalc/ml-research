# Databricks notebook source

# MAGIC %md
# MAGIC # Fraud Detection — Monitoring with Lakehouse Monitoring
# MAGIC The **monitoring** stage that closes the loop. We attach a Unity Catalog
# MAGIC **Lakehouse Monitor** to the live scored stream `cjc.ml.fraud_scored_stream`.
# MAGIC
# MAGIC Lakehouse Monitoring automatically profiles the table on every refresh and tracks
# MAGIC how those profiles **drift over time** — exactly what a bank needs to know its
# MAGIC fraud model is still behaving in production (score distribution shifting, alert
# MAGIC rate spiking, feature drift). It writes two Delta tables and a dashboard:
# MAGIC
# MAGIC - `..._profile_metrics` — per-refresh, per-column statistics (count, mean, %null,
# MAGIC   distinct, distribution) including our `fraud_probability` and `fraud_alert`.
# MAGIC - `..._drift_metrics` — how each metric changed vs. the previous refresh / baseline.
# MAGIC
# MAGIC No agents, no external tooling — governed monitoring built into Unity Catalog.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorSnapshot

w = WorkspaceClient()

CATALOG, SCHEMA = "cjc", "ml"
TABLE = f"{CATALOG}.{SCHEMA}.fraud_scored_stream"
OUTPUT_SCHEMA = f"{CATALOG}.{SCHEMA}"
ASSETS_DIR = "/Workspace/Users/scott.mckean@databricks.com/fraud_detection_demo/monitor_assets"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the monitor (idempotent)
# MAGIC We use a **Snapshot** monitor: it profiles the full table each refresh and tracks
# MAGIC drift across refreshes. (For windowed time-series drift, swap in
# MAGIC `MonitorTimeSeries(timestamp_col="scored_at", granularities=["5 minutes","1 hour"])`;
# MAGIC for model-quality metrics with ground-truth labels, use `MonitorInferenceLog`.)

# COMMAND ----------

try:
    info = w.quality_monitors.get(table_name=TABLE)
    print("Monitor already exists:", info.status)
except Exception:
    info = w.quality_monitors.create(
        table_name=TABLE,
        output_schema_name=OUTPUT_SCHEMA,
        assets_dir=ASSETS_DIR,
        snapshot=MonitorSnapshot(),
        slicing_exprs=["fraud_alert"],   # break metrics out by alerted vs not
    )
    print("Created monitor.")

print("profile metrics table:", info.profile_metrics_table_name)
print("drift metrics table:  ", info.drift_metrics_table_name)
print("dashboard id:         ", info.dashboard_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh the monitor
# MAGIC Recomputes the profile and drift metrics over the current table contents.
# MAGIC In production you'd schedule this (e.g. hourly) so drift is tracked continuously.

# COMMAND ----------

run = w.quality_monitors.run_refresh(table_name=TABLE)
print(f"Refresh {run.refresh_id} started ({run.state}). Waiting...")

import time
while True:
    r = w.quality_monitors.get_refresh(table_name=TABLE, refresh_id=run.refresh_id)
    if str(r.state) in ("MonitorRefreshInfoState.SUCCESS", "MonitorRefreshInfoState.FAILED",
                         "MonitorRefreshInfoState.CANCELED", "SUCCESS", "FAILED", "CANCELED"):
        print("Refresh state:", r.state)
        break
    time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring view 1 — operational health
# MAGIC Transaction volume and fraud-alert rate from the latest profile. This is the
# MAGIC headline a fraud-ops team watches: are we scoring traffic, and is the alert rate
# MAGIC where we expect it?

# COMMAND ----------

profile_tbl = f"{TABLE}_profile_metrics"
display(spark.sql(f"""
    SELECT window, column_name, count, num_nulls,
           ROUND(avg, 4)    AS mean_value,
           ROUND(percent_null, 4) AS percent_null
    FROM {profile_tbl}
    WHERE column_name IN ('fraud_probability', 'fraud_alert', ':table')
      AND slice_key IS NULL
    ORDER BY window DESC, column_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring view 2 — score distribution by alert slice
# MAGIC The monitor automatically computes metrics per `fraud_alert` slice, so you can
# MAGIC compare the score profile of alerted vs. non-alerted transactions.

# COMMAND ----------

display(spark.sql(f"""
    SELECT window, slice_key, slice_value, column_name,
           count, ROUND(avg, 4) AS mean_score
    FROM {profile_tbl}
    WHERE column_name = 'fraud_probability' AND slice_key = 'fraud_alert'
    ORDER BY window DESC, slice_value
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring view 3 — drift across refreshes
# MAGIC After two or more refreshes, this table shows how the score distribution and
# MAGIC alert rate are shifting — the early-warning signal for model/data drift.

# COMMAND ----------

drift_tbl = f"{TABLE}_drift_metrics"
display(spark.sql(f"""
    SELECT window, column_name, drift_type,
           ROUND(js_distance, 5) AS js_distance
    FROM {drift_tbl}
    WHERE column_name IN ('fraud_probability', 'fraud_alert')
    ORDER BY window DESC, column_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The monitoring dashboard
# MAGIC Lakehouse Monitoring also generated a ready-made dashboard (see the monitor's
# MAGIC **Quality** tab on the table in Catalog Explorer, or `monitor_assets/`). It
# MAGIC visualizes volume, null rates, score distributions, and drift over time with
# MAGIC zero additional setup.
