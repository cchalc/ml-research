# Databricks notebook source

# MAGIC %md
# MAGIC # Gorda Ridge — Process
# MAGIC
# MAGIC Source: USGS Escanaba Trough data release —
# MAGIC [sciencebase.gov/catalog/item/67004442d34e80be174aea95](https://www.sciencebase.gov/catalog/item/67004442d34e80be174aea95).
# MAGIC
# MAGIC Turns the bronze landings into a single analytics-ready table per
# MAGIC sensor stream (all under `cjc.ml` with a `gordaridge_` prefix):
# MAGIC
# MAGIC | Silver table              | Built from                                                                |
# MAGIC | ------------------------- | ------------------------------------------------------------------------- |
# MAGIC | `gordaridge_silver_xrf`   | `gordaridge_bronze_xrf_geochem` ∪ `gordaridge_bronze_xrf_soil` + locs     |
# MAGIC | `gordaridge_silver_mscl`  | `gordaridge_bronze_mscl` + locs                                           |
# MAGIC
# MAGIC Adds:
# MAGIC * a `site` column (`GC01-Y` → `GC01`) so we can roll up across the
# MAGIC   X/Y/Z section splits;
# MAGIC * `log_{El}` columns (log1p) since elemental ppm spans several orders
# MAGIC   of magnitude;
# MAGIC * lat / lon / water depth from the locations table.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "cjc"
SCHEMA = "ml"
PREFIX = "gordaridge_"

XRF_ELEMENTS = ("Mn", "Fe", "Cu", "Pb", "S", "Ca")

# COMMAND ----------

def site_from_core(col="core_id"):
    return F.regexp_replace(F.col(col), r"-[XYZ]$", "")


def t(name):
    return f"{CATALOG}.{SCHEMA}.{PREFIX}{name}"


def write(df, name):
    full = t(name)
    (df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full))
    n = spark.table(full).count()
    print(f"wrote {full}: {n:,} rows")


locations = spark.table(t("bronze_locations"))

# COMMAND ----------

# MAGIC %md ## silver_xrf

# COMMAND ----------

xrf = (
    spark.table(t("bronze_xrf_geochem"))
    .unionByName(spark.table(t("bronze_xrf_soil")))
)
for el in XRF_ELEMENTS:
    xrf = xrf.withColumn(
        f"log_{el}", F.log1p(F.greatest(F.col(f"{el}_ppm"), F.lit(0.0)))
    )
silver_xrf = (
    xrf.withColumn("site", site_from_core())
    .join(locations, F.col("site") == locations["core_id"], "left")
    .drop(locations["core_id"])
)
write(silver_xrf, "silver_xrf")

# COMMAND ----------

# MAGIC %md ## silver_mscl

# COMMAND ----------

silver_mscl = (
    spark.table(t("bronze_mscl"))
    .withColumn("site", site_from_core())
    .join(locations, F.col("site") == locations["core_id"], "left")
    .drop(locations["core_id"])
)
write(silver_mscl, "silver_mscl")

# COMMAND ----------

# MAGIC %md ## Sanity checks

# COMMAND ----------

display(spark.sql(f"""
  SELECT mode, COUNT(*) AS n_rows, COUNT(DISTINCT core_id) AS n_cores
  FROM {t('silver_xrf')}
  GROUP BY mode
  ORDER BY mode
"""))

# COMMAND ----------

display(spark.sql(f"""
  SELECT site, mode, COUNT(*) AS n_samples,
         AVG(latitude) AS lat, AVG(longitude) AS lon,
         AVG(water_depth_m) AS water_depth
  FROM {t('silver_xrf')}
  GROUP BY site, mode
  ORDER BY site, mode
"""))
