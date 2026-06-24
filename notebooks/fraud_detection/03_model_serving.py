# Databricks notebook source

# MAGIC %md
# MAGIC # Fraud Detection — Real-Time Model Serving
# MAGIC The second **inference** pattern: deploy the champion model to a **Model Serving**
# MAGIC endpoint for low-latency, synchronous scoring — the path an authorization system
# MAGIC calls while the cardholder waits at the terminal.
# MAGIC
# MAGIC We create (or update) a serverless endpoint backed by the UC model, wait for it
# MAGIC to become ready, then send it a couple of transactions and get scores back over
# MAGIC REST in milliseconds.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, ServedModelInputWorkloadSize,
)
from mlflow.tracking import MlflowClient
import mlflow

CATALOG = "cjc"
SCHEMA = "ml"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_xgboost"
ENDPOINT_NAME = "fraud_detection_demo"

mlflow.set_registry_uri("databricks-uc")
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the champion version

# COMMAND ----------

client = MlflowClient()
champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
print(f"Serving {UC_MODEL_NAME} v{champion.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update the serving endpoint
# MAGIC `scale_to_zero_enabled=True` means the endpoint costs nothing when idle — ideal
# MAGIC for a demo. The first request after idle incurs a cold start.

# COMMAND ----------

served = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=champion.version,
    workload_size=ServedModelInputWorkloadSize.SMALL,
    scale_to_zero_enabled=True,
)
config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served])

existing = [e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME]
if existing:
    print(f"Updating existing endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT_NAME, served_entities=[served])
else:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create_and_wait(name=ENDPOINT_NAME, config=config)

print("Endpoint is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the endpoint
# MAGIC Two transactions: a normal local purchase and a risky foreign,
# MAGIC far-from-home, card-not-present transaction at a high-risk merchant.

# COMMAND ----------

records = [
    {  # looks legitimate
        "amount": 42.50, "amount_to_avg_ratio": 1.05, "distance_from_home_km": 3.2,
        "num_tx_last_hour": 1, "hour_of_day": 13, "merchant_risk": 0.10,
        "is_foreign": 0, "card_present": 1, "account_age_days": 1800,
    },
    {  # looks like fraud
        "amount": 980.00, "amount_to_avg_ratio": 7.8, "distance_from_home_km": 480.0,
        "num_tx_last_hour": 6, "hour_of_day": 3, "merchant_risk": 0.85,
        "is_foreign": 1, "card_present": 0, "account_age_days": 95,
    },
]

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=records,
)
for rec, score in zip(records, response.predictions):
    print(f"amount=${rec['amount']:>7.2f}  foreign={rec['is_foreign']}  "
          f"card_present={rec['card_present']}  ->  fraud_probability={score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC The endpoint can now be called from any application over REST. To clean up and
# MAGIC stop incurring cost, delete it:
# MAGIC ```python
# MAGIC w.serving_endpoints.delete(name=ENDPOINT_NAME)
# MAGIC ```
