# Databricks notebook source

# MAGIC %md
# MAGIC # Fraud Detection — Data Preparation
# MAGIC Generates a synthetic credit-card transaction dataset with engineered features
# MAGIC and a `is_fraud` label, then writes it to `cjc.ml.fraud_transactions`.
# MAGIC
# MAGIC This is the **data** stage of the demo:
# MAGIC `data → training → experimentation → registry → serving → inference`.
# MAGIC
# MAGIC The data is synthetic but the feature design mirrors what a card-issuing bank
# MAGIC actually scores in production: transaction amount relative to the customer's
# MAGIC normal spend, geographic distance, recent transaction velocity, merchant risk,
# MAGIC card-present vs card-not-present, and foreign-transaction flags.

# COMMAND ----------

import numpy as np
import pandas as pd

CATALOG = "cjc"
SCHEMA = "ml"
TABLE = f"{CATALOG}.{SCHEMA}.fraud_transactions"

N_ROWS = 250_000
SEED = 42
rng = np.random.default_rng(SEED)

# Feature columns used by the model downstream. Keep this list in one place —
# the training, batch, serving, and streaming notebooks all import the same names.
FEATURES = [
    "amount",
    "amount_to_avg_ratio",
    "distance_from_home_km",
    "num_tx_last_hour",
    "hour_of_day",
    "merchant_risk",
    "is_foreign",
    "card_present",
    "account_age_days",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate raw features
# MAGIC Each feature is drawn from a distribution that roughly matches real card data.

# COMMAND ----------

amount = np.round(rng.lognormal(mean=3.5, sigma=1.1, size=N_ROWS), 2)          # ~$5–$1000+
amount_to_avg_ratio = np.round(rng.lognormal(mean=0.0, sigma=0.6, size=N_ROWS), 3)  # 1.0 == typical spend
distance_from_home_km = np.round(rng.exponential(scale=12.0, size=N_ROWS), 1)  # most tx near home
num_tx_last_hour = rng.poisson(lam=1.2, size=N_ROWS)                            # transaction velocity
hour_of_day = rng.integers(0, 24, size=N_ROWS)
merchant_risk = np.round(rng.beta(a=2.0, b=8.0, size=N_ROWS), 3)               # 0=safe, 1=risky merchant
is_foreign = rng.binomial(1, 0.06, size=N_ROWS)
card_present = rng.binomial(1, 0.72, size=N_ROWS)
account_age_days = rng.integers(30, 3650, size=N_ROWS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Derive the fraud label
# MAGIC We build a continuous **risk signal** from the features — high when a transaction
# MAGIC is far from home, high-velocity, at night, card-not-present, foreign, and at a
# MAGIC risky merchant — add realistic noise (so the label isn't perfectly predictable),
# MAGIC and flag the riskiest ~2.8% as fraud. This gives a strong, learnable signal:
# MAGIC a good model reaches ~0.95 AUC, but noise keeps it from being trivial.

# COMMAND ----------

night = ((hour_of_day < 6) | (hour_of_day >= 23)).astype(float)

signal = (
    0.9 * np.log1p(amount_to_avg_ratio)
    + 0.020 * distance_from_home_km
    + 0.35 * num_tx_last_hour
    + 1.1 * night
    + 2.4 * merchant_risk
    + 1.3 * is_foreign
    + 0.8 * (1 - card_present)
    - 0.0003 * account_age_days
)
score = signal + rng.normal(0, 0.45 * signal.std(), size=N_ROWS)  # realistic overlap
threshold = np.quantile(score, 1 - 0.028)                          # riskiest ~2.8%
is_fraud = (score >= threshold).astype(int)

print(f"Fraud rate: {is_fraud.mean():.2%}  ({is_fraud.sum():,} of {N_ROWS:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble the dataset
# MAGIC Add identifiers and an `event_time` spread over the last 30 days so the
# MAGIC training notebook can do a realistic time-based train/test split.

# COMMAND ----------

now = pd.Timestamp.utcnow().tz_localize(None)
event_time = now - pd.to_timedelta(rng.integers(0, 30 * 24 * 3600, size=N_ROWS), unit="s")

pdf = pd.DataFrame({
    "transaction_id": np.arange(N_ROWS),
    "event_time": event_time,
    "customer_id": rng.integers(1, 20_000, size=N_ROWS),
    "amount": amount,
    "amount_to_avg_ratio": amount_to_avg_ratio,
    "distance_from_home_km": distance_from_home_km,
    "num_tx_last_hour": num_tx_last_hour.astype(int),
    "hour_of_day": hour_of_day.astype(int),
    "merchant_risk": merchant_risk,
    "is_foreign": is_foreign.astype(int),
    "card_present": card_present.astype(int),
    "account_age_days": account_age_days.astype(int),
    "is_fraud": is_fraud,
})
print(pdf.shape)
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

(
    spark.createDataFrame(pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE)
)

print(f"Wrote {pdf.shape[0]:,} rows to {TABLE}")
display(spark.sql(f"SELECT is_fraud, COUNT(*) AS n FROM {TABLE} GROUP BY is_fraud ORDER BY is_fraud"))
