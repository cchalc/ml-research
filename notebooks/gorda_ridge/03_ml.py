# Databricks notebook source

# MAGIC %md
# MAGIC # Gorda Ridge — ML
# MAGIC
# MAGIC Source: USGS Escanaba Trough data release —
# MAGIC [sciencebase.gov/catalog/item/67004442d34e80be174aea95](https://www.sciencebase.gov/catalog/item/67004442d34e80be174aea95).
# MAGIC
# MAGIC Treats every scanned depth interval as one observation in 6-D log-element
# MAGIC space (Mn, Fe, Cu, Pb, S, Ca) and looks for sediment "facies" — recurring
# MAGIC chemistries that cut across cores.
# MAGIC
# MAGIC Pipeline:
# MAGIC 1. sweep `k` from 2..10 for KMeans, log silhouette / CH / DB / inertia
# MAGIC 2. fit at the chosen `k` (default 4) and log model + centroids to MLflow
# MAGIC 3. write `gold_xrf_clusters` (per-row labels + PC1/PC2)
# MAGIC 4. write `gold_xrf_centroids` (cluster → ppm)
# MAGIC 5. write `gold_site_summary` (per-site lat/lon + chemistry + dominant cluster)

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

CATALOG = "cjc"
SCHEMA = "ml"
PREFIX = "gordaridge_"
XRF_ELEMENTS = ("Mn", "Fe", "Cu", "Pb", "S", "Ca")
K = 4
RANDOM_STATE = 0

dbutils.widgets.text("experiment_path", f"/Shared/{PREFIX.rstrip('_')}",
                     "MLflow experiment path")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")


def t(name):
    return f"{CATALOG}.{SCHEMA}.{PREFIX}{name}"

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.set_registry_uri("databricks-uc")
print(f"experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md ## Build feature matrix (geochem mode, median-imputed)

# COMMAND ----------

silver = spark.table(t("silver_xrf")).toPandas()
feat_cols = [f"log_{el}" for el in XRF_ELEMENTS]

work = silver[silver["mode"] == "geochem"].copy()
for c in feat_cols:
    work[c] = work[c].fillna(work[c].median())
X = work[feat_cols].reset_index(drop=True)
idx = work[["core_id", "site", "mode", "core_depth_cm"]].reset_index(drop=True)
print(f"feature matrix: {X.shape}")

# COMMAND ----------

# MAGIC %md ## Sweep k

# COMMAND ----------

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

sweep_rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    labels = km.labels_
    sweep_rows.append({
        "k": k,
        "inertia": float(km.inertia_),
        "silhouette": float(silhouette_score(Xs, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Xs, labels)),
        "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
    })
sweep = pd.DataFrame(sweep_rows)
sweep

# COMMAND ----------

with mlflow.start_run(run_name="kmeans_sweep") as sweep_run:
    mlflow.log_param("mode", "geochem")
    mlflow.log_param("features", ",".join(feat_cols))
    for row in sweep_rows:
        for metric, value in row.items():
            if metric == "k":
                continue
            mlflow.log_metric(metric, value, step=row["k"])
    mlflow.log_table(sweep, artifact_file="sweep.json")
print(f"logged sweep run: {sweep_run.info.run_id}")

# COMMAND ----------

# MAGIC %md ## Fit final model and register

# COMMAND ----------

with mlflow.start_run(run_name=f"kmeans_k{K}") as run:
    model = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    labels = model.labels_

    pca = PCA(n_components=2).fit(Xs)
    pcs = pca.transform(Xs)

    metrics = {
        "k": float(K),
        "n_samples": float(len(X)),
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(Xs, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Xs, labels)),
        "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
        "pca_evr_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_evr_pc2": float(pca.explained_variance_ratio_[1]),
    }
    mlflow.log_params({"k": K, "mode": "geochem", "random_state": RANDOM_STATE,
                        "features": ",".join(feat_cols)})
    mlflow.log_metrics(metrics)

    centroid_log = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_), columns=feat_cols
    )
    centroids_ppm = np.expm1(centroid_log).round(0)
    centroids_ppm.columns = [c.replace("log_", "") + "_ppm" for c in centroid_log.columns]
    centroids_ppm.insert(0, "cluster", range(K))
    mlflow.log_table(centroids_ppm, artifact_file="centroids_ppm.json")

    signature = mlflow.models.infer_signature(X.head(5), labels[:5])
    mlflow.sklearn.log_model(
        model,
        artifact_path="kmeans",
        signature=signature,
        input_example=X.head(5),
        registered_model_name=f"{CATALOG}.{SCHEMA}.{PREFIX}xrf_kmeans",
    )
    print("metrics:", metrics)
    print(f"run_id: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md ## gold_xrf_clusters & gold_xrf_centroids

# COMMAND ----------

clusters = idx.copy()
clusters["cluster"] = labels.astype("int32")
clusters["pc1"] = pcs[:, 0]
clusters["pc2"] = pcs[:, 1]

(spark.createDataFrame(clusters)
   .write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable(t("gold_xrf_clusters")))
(spark.createDataFrame(centroids_ppm)
   .write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable(t("gold_xrf_centroids")))

display(spark.sql(f"""
  SELECT cluster, COUNT(*) AS n
  FROM {t('gold_xrf_clusters')}
  GROUP BY cluster
  ORDER BY cluster
"""))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {t('gold_xrf_centroids')} ORDER BY cluster"))

# COMMAND ----------

# MAGIC %md ## Narrative — what the four clusters look like
# MAGIC
# MAGIC From the run that materialised this schema (k=4 on the geochem mode):
# MAGIC
# MAGIC | cluster | Mn   | Fe     | Cu    | Pb  | S     | Ca    | reading                                  |
# MAGIC | ------- | ---- | ------ | ----- | --- | ----- | ----- | ---------------------------------------- |
# MAGIC | 0       | ~510 | ~24k   | ~38   | ~9  | ~1.1k | ~5.0k | "background" pelagic sediment            |
# MAGIC | 1       | ~0   | ~1     | ~0    | ~0  | ~2    | ~2    | imputation floor — rows where every element was below detection and got median-filled near zero. Treat as a missingness bucket, not a real facies. |
# MAGIC | 2       | ~49  | ~81k   | ~2.0k | 472 | ~7.0k | ~2.4k | hydrothermal end-member: Fe ≈ 3× background, big Cu/Pb/S spikes — consistent with massive-sulfide chimney debris near the Escanaba vents. |
# MAGIC | 3       | ~458 | ~26k   | ~39   | ~5  | ~0    | ~2.0k | a sulfur-poor, slightly Mn-enriched twin of cluster 0 — likely diagenetically-reduced background.                                            |
# MAGIC
# MAGIC **Implications for future ML work**
# MAGIC * Cluster 1 is an artefact of `fillna(median)` on rows where the entire
# MAGIC   ppm panel was BDL. A meaningful next pass should either drop those rows
# MAGIC   (`impute="drop"`) or model BDL explicitly (e.g. tobit / left-censored).
# MAGIC * The hydrothermal class (cluster 2) is rare relative to background, so
# MAGIC   downstream supervised tasks should expect heavy class imbalance.
# MAGIC * `gold_site_summary.dominant_cluster` lets you map facies geographically
# MAGIC   and check whether cluster-2 sites cluster spatially around the known
# MAGIC   vent field — a quick sanity check that the chemistry is picking up
# MAGIC   real geology, not core-handling noise.

# COMMAND ----------

# MAGIC %md ## gold_site_summary
# MAGIC Per-site rollup: lat/lon, average element ppm, average MSCL response,
# MAGIC dominant cluster.

# COMMAND ----------

silver_xrf = spark.table(t("silver_xrf")).where(F.col("mode") == "geochem")
silver_mscl = spark.table(t("silver_mscl"))
clusters_sdf = spark.table(t("gold_xrf_clusters"))
locs = spark.table(t("bronze_locations"))

chem = silver_xrf.groupBy("site").agg(
    *[F.avg(f"{el}_ppm").alias(f"avg_{el}_ppm") for el in XRF_ELEMENTS]
)
phys = silver_mscl.groupBy("site").agg(
    F.avg("gamma_density").alias("avg_gamma_density"),
    F.avg("magnetic_susceptibility").alias("avg_magnetic_susceptibility"),
)
dom = clusters_sdf.groupBy("site").agg(
    F.expr("approx_percentile(cluster, 0.5)").alias("dominant_cluster"),
    F.count("*").alias("n_samples"),
)

site_summary = (
    locs.alias("l")
    .join(chem.alias("c"), F.col("l.core_id") == F.col("c.site"), "left")
    .join(phys.alias("p"), F.col("l.core_id") == F.col("p.site"), "left")
    .join(dom.alias("k"), F.col("l.core_id") == F.col("k.site"), "left")
    .select(
        F.col("l.core_id").alias("site"),
        "latitude", "longitude", "water_depth_m",
        *[f"avg_{el}_ppm" for el in XRF_ELEMENTS],
        "avg_gamma_density", "avg_magnetic_susceptibility",
        "dominant_cluster", "n_samples",
    )
)
(site_summary.write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable(t("gold_site_summary")))

display(spark.sql(f"""
  SELECT site, water_depth_m, dominant_cluster, n_samples,
         avg_Fe_ppm, avg_Mn_ppm, avg_Cu_ppm, avg_S_ppm,
         avg_gamma_density, avg_magnetic_susceptibility
  FROM {t('gold_site_summary')}
  ORDER BY dominant_cluster, site
"""))
