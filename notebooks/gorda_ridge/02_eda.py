# Databricks notebook source

# MAGIC %md
# MAGIC # Gorda Ridge — EDA
# MAGIC
# MAGIC Source: USGS Escanaba Trough data release —
# MAGIC [sciencebase.gov/catalog/item/67004442d34e80be174aea95](https://www.sciencebase.gov/catalog/item/67004442d34e80be174aea95).
# MAGIC
# MAGIC Descriptive look at `cjc.ml.gordaridge_silver_xrf`:
# MAGIC * sample counts per core / mode
# MAGIC * below-detection rates per element
# MAGIC * log-element distributions
# MAGIC * inter-element correlations (geochem mode)
# MAGIC * depth profiles
# MAGIC * map of core sites coloured by water depth

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

CATALOG = "cjc"
SCHEMA = "ml"
PREFIX = "gordaridge_"
XRF_ELEMENTS = ("Mn", "Fe", "Cu", "Pb", "S", "Ca")

sns.set_theme(context="notebook", style="whitegrid")

# COMMAND ----------

xrf = spark.table(f"{CATALOG}.{SCHEMA}.{PREFIX}silver_xrf").toPandas()
locs = spark.table(f"{CATALOG}.{SCHEMA}.{PREFIX}bronze_locations").toPandas()

print(f"silver_xrf rows : {len(xrf):,}")
print(f"cores           : {xrf['core_id'].nunique()}")
print(f"sites           : {xrf['site'].nunique()}")
print(f"locations rows  : {len(locs)}")

# COMMAND ----------

# MAGIC %md ## Per-core sample counts

# COMMAND ----------

counts = xrf.groupby(["core_id", "mode"]).size().unstack(fill_value=0).sort_index()
counts["total"] = counts.sum(axis=1)
counts.head(20)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 12))
counts.drop(columns="total").plot(kind="barh", stacked=True, ax=ax)
ax.set_xlabel("samples")
ax.set_title("XRF samples per core, by mode")
plt.tight_layout()

# COMMAND ----------

# MAGIC %md ## Below-detection rate per element

# COMMAND ----------

val_cols = [f"{el}_ppm" for el in XRF_ELEMENTS]
bdl = xrf.groupby("mode")[val_cols].apply(lambda g: g.isna().mean()).round(3)
bdl

# COMMAND ----------

# MAGIC %md ## Log-element distributions

# COMMAND ----------

log_cols = [f"log_{el}" for el in XRF_ELEMENTS]
melted = xrf.melt(id_vars=["mode"], value_vars=log_cols,
                  var_name="element", value_name="log_ppm")
melted["element"] = melted["element"].str.replace("log_", "", regex=False)
melted = melted.dropna(subset=["log_ppm"])

g = sns.catplot(
    data=melted, x="element", y="log_ppm", hue="mode", kind="violin",
    split=True, inner="quartile", height=4, aspect=2,
)
g.fig.suptitle("log(1+ppm) by element and mode", y=1.03)

# COMMAND ----------

# MAGIC %md ## Inter-element correlations (geochem mode)

# COMMAND ----------

geo = xrf[xrf["mode"] == "geochem"]
corr = geo[log_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
            xticklabels=XRF_ELEMENTS, yticklabels=XRF_ELEMENTS, ax=ax)
ax.set_title("log-element correlation (geochem)")

# COMMAND ----------

# MAGIC %md ## Depth profiles
# MAGIC One panel per element; one trace per core.

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
for ax, el in zip(axes.flat, XRF_ELEMENTS):
    for cid, sub in geo.groupby("core_id"):
        ax.plot(sub[f"log_{el}"], sub["core_depth_cm"], lw=0.6, alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"log {el}_ppm")
    ax.set_ylabel("depth (cm)")
    ax.set_title(el)
fig.suptitle("Depth profiles (geochem mode, log-ppm)", y=1.02)
plt.tight_layout()

# COMMAND ----------

# MAGIC %md ## Map of core sites
# MAGIC Coloured by water depth, sized by total XRF samples per site.

# COMMAND ----------

samples_per_site = xrf.groupby("site").size()
loc_plot = locs.copy()
loc_plot["samples"] = loc_plot["core_id"].map(samples_per_site).fillna(0)

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(
    loc_plot["longitude"], loc_plot["latitude"],
    c=loc_plot["water_depth_m"], s=10 + loc_plot["samples"] * 0.6,
    cmap="viridis_r", edgecolor="k", linewidth=0.3,
)
plt.colorbar(sc, ax=ax, label="water depth (m)")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("Escanaba Trough core sites (TN403 cruise)")
for _, r in loc_plot.iterrows():
    ax.annotate(r["core_id"], (r["longitude"], r["latitude"]),
                fontsize=6, alpha=0.6, xytext=(2, 2), textcoords="offset points")
plt.tight_layout()
