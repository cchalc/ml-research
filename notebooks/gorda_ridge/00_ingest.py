# Databricks notebook source

# MAGIC %md
# MAGIC # Gorda Ridge — Ingest
# MAGIC
# MAGIC Pulls the Escanaba Trough sediment-core dataset from USGS ScienceBase
# MAGIC ([doi:10.5066/P13B46QX](https://doi.org/10.5066/P13B46QX),
# MAGIC [data release page](https://www.sciencebase.gov/catalog/item/67004442d34e80be174aea95))
# MAGIC into the `cjc.ml` schema with a `gordaridge_` table prefix.
# MAGIC
# MAGIC We keep four child items (the rest of the release is media zips or
# MAGIC narrow analytical products):
# MAGIC
# MAGIC | Bronze table                       | Source item             |
# MAGIC | ---------------------------------- | ----------------------- |
# MAGIC | `gordaridge_bronze_locations`      | TN403_CoreLocations.csv |
# MAGIC | `gordaridge_bronze_xrf_geochem`    | XRF (geochem mode)      |
# MAGIC | `gordaridge_bronze_xrf_soil`       | XRF (soil mode)         |
# MAGIC | `gordaridge_bronze_mscl`           | Multi-Sensor Core Logger|
# MAGIC
# MAGIC Files land in the `gordaridge_raw` UC Volume; bronze tables are written
# MAGIC next to them in the same schema.

# COMMAND ----------

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

CATALOG = "cjc"
SCHEMA = "ml"
PREFIX = "gordaridge_"
VOLUME = "gordaridge_raw"
VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

ITEMS = {
    "locations":   "679282a3d34e88f5864c49e9",
    "xrf_geochem": "671aab6bd34efed5620fb81a",
    "xrf_soil":    "671aab7dd34efed5620fb829",
    "mscl":        "670045e9d34e80be174aeaa4",
}

USER_AGENT = "ml-research-gorda-ridge/0.1"

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"target schema: {CATALOG}.{SCHEMA}")
print(f"raw volume:    {VOL_PATH}")

# COMMAND ----------

# MAGIC %md ## Download CSVs from ScienceBase

# COMMAND ----------

def _open(url, retries=3, timeout=300):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  retry {attempt + 1}/{retries}: {exc}")


def list_files(item_id):
    with _open(f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json", timeout=60) as r:
        return json.load(r).get("files", [])


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open(url) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return True

for slug, item_id in ITEMS.items():
    out_dir = Path(VOL_PATH) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = n_skip = 0
    for entry in list_files(item_id):
        name = entry.get("name") or ""
        uri = entry.get("downloadUri") or entry.get("url")
        if not uri or not name.lower().endswith(".csv"):
            continue
        if download(uri, out_dir / name):
            n_new += 1
        else:
            n_skip += 1
    print(f"{slug:<12} downloaded={n_new:<3} cached={n_skip}")

# COMMAND ----------

# MAGIC %md ## Bronze tables
# MAGIC Minimal transforms — attach `core_id` from filename, replace USGS
# MAGIC sentinel `-9999` with NULL, sanitize column names.

# COMMAND ----------

XRF_ELEMENTS = ("Mn", "Fe", "Cu", "Pb", "S", "Ca")
SENTINEL = -9999.0
_INVALID_COL = re.compile(r"[ ,;{}()\n\t=\-]")


def sanitize(df):
    new = [_INVALID_COL.sub("_", c) for c in df.columns]
    return df.toDF(*new) if new != df.columns else df


def core_id_from_path(suffix):
    fname = F.element_at(F.split("source_file", "/"), -1)
    pat = r"^(.+?)" + (re.escape(f"_{suffix}") if suffix else "") + r"\.csv$"
    return F.regexp_extract(fname, pat, 1)


def replace_sentinel(df, cols, sentinel=SENTINEL):
    out = df
    for c in cols:
        if c in out.columns:
            out = out.withColumn(c, F.when(F.col(c) == sentinel, None).otherwise(F.col(c)))
    return out


def write(df, name):
    full = f"{CATALOG}.{SCHEMA}.{PREFIX}{name}"
    (df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full))
    n = spark.table(full).count()
    print(f"wrote {full}: {n:,} rows")

# COMMAND ----------

# MAGIC %md ### Locations

# COMMAND ----------

locations = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(f"{VOL_PATH}/locations/TN403_CoreLocations.csv")
    .where(F.col("Core_ID").isNotNull())
    .select(
        F.trim("Core_ID").alias("core_id"),
        F.col("Latitude").cast(DoubleType()).alias("latitude"),
        F.col("Longitude").cast(DoubleType()).alias("longitude"),
        F.col("Water_Depth").cast(DoubleType()).alias("water_depth_m"),
    )
)
write(locations, "bronze_locations")

# COMMAND ----------

# MAGIC %md ### XRF — geochem & soil
# MAGIC Each per-core CSV shares the schema
# MAGIC `core_depth_cm, XRF_TotalCounts, XRF_Live_Time_sec, {El}_ppm, {El}-Error_ppm`.

# COMMAND ----------

def read_xrf(vol_dir, mode):
    df = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{vol_dir}/*.csv")
        .withColumn("source_file", F.col("_metadata.file_path"))
        .withColumn("mode", F.lit(mode))
        .withColumn("core_id", core_id_from_path(mode))
    )
    numeric = (
        ["XRF_TotalCounts", "XRF_Live_Time_sec"]
        + [f"{el}_ppm" for el in XRF_ELEMENTS]
        + [f"{el}-Error_ppm" for el in XRF_ELEMENTS]
    )
    for c in numeric:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
    df = replace_sentinel(df, numeric)
    return sanitize(df.select("core_id", "mode", "core_depth_cm", *numeric, "source_file"))

write(read_xrf(f"{VOL_PATH}/xrf_geochem", "geochem"), "bronze_xrf_geochem")
write(read_xrf(f"{VOL_PATH}/xrf_soil", "soil"), "bronze_xrf_soil")

# COMMAND ----------

# MAGIC %md ### MSCL physical properties

# COMMAND ----------

mscl = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(f"{VOL_PATH}/mscl/*.csv")
    .withColumn("source_file", F.col("_metadata.file_path"))
    .withColumn("core_id", core_id_from_path(None))
)
for c in ("gamma_density", "magnetic_susceptibility", "core_depth"):
    if c in mscl.columns:
        mscl = mscl.withColumn(c, F.col(c).cast(DoubleType()))
mscl = replace_sentinel(mscl, ["gamma_density", "magnetic_susceptibility"])
write(
    sanitize(mscl.select(
        "core_id", "core_depth", "gamma_density", "magnetic_susceptibility", "source_file",
    )),
    "bronze_mscl",
)

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA} LIKE '{PREFIX}bronze_*'"))
