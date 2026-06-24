# Databricks notebook source
# MAGIC %md
# MAGIC # Spark vs Ray — Heterogeneous NLP (base)
# MAGIC
# MAGIC Three stages, CPU → GPU → CPU, over 10k clinical docs. Same Python in
# MAGIC both jobs — three Pandas UDFs on the Spark side, three `map_batches`
# MAGIC on the Ray side.
# MAGIC
# MAGIC | Stage         | Bound | Work                                              |
# MAGIC | ------------- | ----- | ------------------------------------------------- |
# MAGIC | preprocess    | CPU   | sentence splitting + cleanup                      |
# MAGIC | gpu_infer     | GPU   | biomedical NER + sentiment classification         |
# MAGIC | aggregate     | CPU   | per-doc summary (entity counts, mean score, etc.) |
# MAGIC
# MAGIC This is the straightforward shape. The optimized variant in
# MAGIC `heterogeneous_nlp_demo_optimized` swaps stage 2 for `mapInPandas`
# MAGIC on the Spark side and defers per-doc unpacking off the GPU actor on
# MAGIC the Ray side.
# MAGIC
# MAGIC Outputs land in `cjc.ml.spark_vs_ray_docs`, `_spark_results`, and
# MAGIC `_ray_results`.

# COMMAND ----------

# MAGIC %pip install -q transformers==4.44.2 datasets==2.21.0 ray[data]==2.35.0 pynvml==11.5.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

CATALOG = "cjc"
SCHEMA = "ml"
PREFIX = "spark_vs_ray_"
VARIANT = "base"  # distinguishes outputs from the optimized notebook

N_DOCS = 10_000
NER_MODEL = "d4data/biomedical-ner-all"
CLF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

DOCS_TABLE = f"{CATALOG}.{SCHEMA}.{PREFIX}docs"
SPARK_RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.{PREFIX}spark_results_{VARIANT}"
RAY_RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.{PREFIX}ray_results_{VARIANT}"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"target schema: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md ## 2. Data prep — `spark_vs_ray_docs`

# COMMAND ----------

from pyspark.sql import functions as F


def _table_exists(name: str) -> bool:
    try:
        return spark.table(name).limit(1).count() >= 0
    except Exception:
        return False


if _table_exists(DOCS_TABLE) and spark.table(DOCS_TABLE).count() == N_DOCS:
    print(f"reusing {DOCS_TABLE} ({N_DOCS} rows)")
else:
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_artificial", split=f"train[:{N_DOCS}]")
    rows = [
        {"doc_id": i, "content": " ".join(r["context"]["contexts"])}
        for i, r in enumerate(ds)
    ]
    (
        spark.createDataFrame(rows)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(DOCS_TABLE)
    )
    print(f"wrote {DOCS_TABLE}: {spark.table(DOCS_TABLE).count():,} rows")

display(spark.table(DOCS_TABLE).limit(3))

# COMMAND ----------

# MAGIC %md ## 3. Three shared stage functions

# COMMAND ----------

import re
import time
from typing import Dict, List

import numpy as np
import torch

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def stage1_preprocess(texts: List[str]) -> List[List[str]]:
    """CPU: split each doc into sentences, drop empties, hard-truncate."""
    out = []
    for t in texts:
        sents = [s.strip()[:1000] for s in _SENT_SPLIT.split(t or "") if s.strip()]
        out.append(sents or [""])
    return out


class Stage2GPUInfer:
    """GPU: biomedical NER + sentiment classification.

    Forward pass and per-doc unpacking happen inside the same call.
    """

    def __init__(self, device: int = 0):
        from transformers import pipeline as hf_pipeline

        self.ner = hf_pipeline(
            "ner", model=NER_MODEL, device=device, aggregation_strategy="simple"
        )
        self.clf = hf_pipeline("text-classification", model=CLF_MODEL, device=device)

    def __call__(self, per_doc_sents: List[List[str]]) -> List[Dict]:
        flat: List[str] = []
        offsets: List[int] = [0]
        for sents in per_doc_sents:
            flat.extend(sents)
            offsets.append(len(flat))

        ner_out = self.ner(flat, batch_size=32) if flat else []
        clf_out = self.clf(flat, batch_size=64) if flat else []

        results = []
        for i in range(len(per_doc_sents)):
            s, e = offsets[i], offsets[i + 1]
            ents = []
            for sent_ents in ner_out[s:e]:
                ents.extend(
                    {"text": str(x.get("word", "")), "type": str(x.get("entity_group", ""))}
                    for x in (sent_ents or [])
                )
            doc_clf = clf_out[s:e] or []
            results.append({
                "entities": ents,
                "labels": [str(c.get("label", "")) for c in doc_clf],
                "scores": [float(c.get("score", 0.0)) for c in doc_clf],
            })
        return results


def stage3_aggregate(per_doc_results: List[Dict]) -> List[Dict]:
    """CPU: roll sentence-level outputs up to per-doc summary stats."""
    summaries = []
    for r in per_doc_results:
        ents = list(r["entities"]) if r.get("entities") is not None else []
        labels = list(r["labels"]) if r.get("labels") is not None else []
        scores = list(r["scores"]) if r.get("scores") is not None else []
        n_pos = sum(1 for L in labels if L == "POSITIVE")
        types = [e["type"] for e in ents if e.get("type")]
        top_type = max(set(types), key=types.count) if types else ""
        summaries.append({
            "n_entities": len(ents),
            "n_sentences": len(labels),
            "n_positive": n_pos,
            "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "top_entity_type": top_type,
        })
    return summaries

# COMMAND ----------

# MAGIC %md ### Smoke test on 3 docs

# COMMAND ----------

assert torch.cuda.is_available(), "This demo requires a GPU node."
_sample_texts = [r["content"] for r in spark.table(DOCS_TABLE).limit(3).toPandas().to_dict("records")]

_t0 = time.perf_counter()
_sents = stage1_preprocess(_sample_texts)
_t1 = time.perf_counter()
_infer = Stage2GPUInfer(device=0)
_inf = _infer(_sents)
_t2 = time.perf_counter()
_summary = stage3_aggregate(_inf)
_t3 = time.perf_counter()

print(f"stage1 (CPU pre): {(_t1 - _t0) * 1000:.1f} ms — sentence counts: {[len(s) for s in _sents]}")
print(f"stage2 (GPU):     {(_t2 - _t1) * 1000:.1f} ms — first doc entities: {_inf[0]['entities'][:3]}")
print(f"stage3 (CPU agg): {(_t3 - _t2) * 1000:.1f} ms — summaries: {_summary}")

del _infer
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## 4. GPU utilization sampler

# COMMAND ----------

import threading
from contextlib import contextmanager

import pynvml


@contextmanager
def gpu_sampler(interval_s: float = 0.25):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    samples: list = []
    stop = threading.Event()
    t0 = time.perf_counter()

    def _loop():
        while not stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                samples.append((time.perf_counter() - t0, u.gpu, u.memory))
            except Exception:
                pass
            stop.wait(interval_s)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop.set()
        t.join(timeout=2)
        pynvml.nvmlShutdown()

# COMMAND ----------

# MAGIC %md ## 5. Spark — three Pandas UDFs
# MAGIC
# MAGIC Four things the GPU UDF needs that Spark won't give us by default:
# MAGIC
# MAGIC 1. **Lazy module-global singleton** so the model loads once per worker
# MAGIC    process instead of once per Arrow batch.
# MAGIC 2. **`maxRecordsPerBatch=1000`** to cap memory per batch. The 5,000
# MAGIC    cap is fixed and there's no minimum knob.
# MAGIC 3. **`asNondeterministic()`** to keep Spark from re-running the UDF
# MAGIC    inside its query plan.
# MAGIC 4. **`repartition(1)`** before stage 2 — manual GPU affinity, since
# MAGIC    Spark's resource model can't pin GPU-bound tasks without also
# MAGIC    capping CPU usage on the node.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType, DoubleType, IntegerType, StringType, StructField, StructType,
)
import pandas as pd

# Force small batches — same shape as a checkpoint size trick.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")

_SENTS_T = ArrayType(StringType())
_ENT_T = StructType([
    StructField("text", StringType()),
    StructField("type", StringType()),
])
_INFER_T = StructType([
    StructField("entities", ArrayType(_ENT_T)),
    StructField("labels", ArrayType(StringType())),
    StructField("scores", ArrayType(DoubleType())),
])
_SUMMARY_T = StructType([
    StructField("n_entities", IntegerType()),
    StructField("n_sentences", IntegerType()),
    StructField("n_positive", IntegerType()),
    StructField("mean_score", DoubleType()),
    StructField("top_entity_type", StringType()),
])

# Module-global lazy load. Without this every Arrow batch reloads ~500MB.
_GPU_INFER_SINGLETON: dict = {}


def _get_gpu_infer() -> Stage2GPUInfer:
    if "p" not in _GPU_INFER_SINGLETON:
        _GPU_INFER_SINGLETON["p"] = Stage2GPUInfer(device=0)
    return _GPU_INFER_SINGLETON["p"]


@pandas_udf(_SENTS_T)
def udf_stage1_preprocess(content: pd.Series) -> pd.Series:
    return pd.Series(stage1_preprocess(content.tolist()))


@pandas_udf(_INFER_T)
def udf_stage2_infer(sents: pd.Series) -> pd.DataFrame:
    pipe = _get_gpu_infer()
    out = pipe([list(s) for s in sents.tolist()])
    return pd.DataFrame({
        "entities": [r["entities"] for r in out],
        "labels": [r["labels"] for r in out],
        "scores": [r["scores"] for r in out],
    })


# Mark the GPU UDF non-deterministic — rerun-prevention hack.
udf_stage2_infer = udf_stage2_infer.asNondeterministic()


@pandas_udf(_SUMMARY_T)
def udf_stage3_aggregate(nlp: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(stage3_aggregate(nlp.to_dict("records")))

# COMMAND ----------

with gpu_sampler() as spark_gpu:
    spark_t0 = time.perf_counter()
    spark_df = (
        spark.table(DOCS_TABLE)
        .withColumn("sentences", udf_stage1_preprocess(F.col("content")))
        .repartition(1)  # GPU-affinity hack: serialize stage 2 onto 1 task
        .withColumn("nlp", udf_stage2_infer(F.col("sentences")))
        .withColumn("summary", udf_stage3_aggregate(F.col("nlp")))
        .select(
            "doc_id",
            F.col("nlp.entities").alias("entities"),
            F.col("nlp.labels").alias("labels"),
            F.col("nlp.scores").alias("scores"),
            F.col("summary.n_entities").alias("n_entities"),
            F.col("summary.n_sentences").alias("n_sentences"),
            F.col("summary.n_positive").alias("n_positive"),
            F.col("summary.mean_score").alias("mean_score"),
            F.col("summary.top_entity_type").alias("top_entity_type"),
        )
    )
    (
        spark_df.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(SPARK_RESULTS_TABLE)
    )
    spark_total_s = time.perf_counter() - spark_t0

spark_rows = spark.table(SPARK_RESULTS_TABLE).count()
print(f"Spark: {spark_rows:,} rows in {spark_total_s:.1f}s ({spark_rows / spark_total_s:.1f} rows/s)")

# Free GPU memory before the Ray section.
_GPU_INFER_SINGLETON.clear()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## 6. Ray — three `map_batches` stages
# MAGIC
# MAGIC One `.map_batches` per stage. Stage 2 declares `num_gpus=1,
# MAGIC concurrency=1` — Ray pins one actor to the GPU, runs `__init__`
# MAGIC exactly once, and reuses it for every batch. Stages 1 and 3 take
# MAGIC `num_cpus=2, concurrency=2` so they parallelize on CPU.
# MAGIC
# MAGIC None of the four Spark workarounds above need a Ray equivalent —
# MAGIC the resource declarations handle it.

# COMMAND ----------

import ray

# Single-node demo: Ray runs locally on the driver, which is all this
# comparison needs. Scaling onto the Spark workers is where the sharp edges
# live, so spell the correct flow out rather than hand-wave it:
#
#   from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
#
#   ray.shutdown()                 # drop any local/stale driver context first
#   try:
#       shutdown_ray_cluster()     # tear down a prior Ray-on-Spark cluster
#   except Exception:
#       pass
#
#   conn, _ = setup_ray_cluster(
#       min_worker_nodes=N, max_worker_nodes=N,   # fixed size — from_spark()
#                                                 # does NOT support autoscaling
#       num_cpus_worker_node=8,
#       num_gpus_worker_node=1,                   # Ray pins GPU work to GPU nodes
#       collect_log_to_path="/Volumes/<cat>/<schema>/<vol>/ray_logs",
#   )                              # leave head-node compute at 0 in hybrid mode —
#                                  # the driver's Spark will starve/kill Ray otherwise
#   ray.init(address=conn, ignore_reinit_error=True)   # CONNECT to that cluster.
#                                  # A bare ray.init() here starts a *separate*
#                                  # context, so actors land elsewhere and you get
#                                  # "Can't find actor ... it's from a different
#                                  # cluster" / EADDRINUSE worker deaths.
#   ... run the pipeline ...
#   shutdown_ray_cluster()         # required, or collected logs never copy out
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=4, num_gpus=1)
print(f"ray cluster resources: {ray.cluster_resources()}")

# COMMAND ----------

import json
import ray.data


# Intermediate columns are JSON strings — Arrow object-extension types don't
# round-trip cleanly across stage boundaries on this runtime.


def ray_stage1(batch: Dict[str, np.ndarray]) -> Dict[str, list]:
    sents = stage1_preprocess([str(t) for t in batch["content"]])
    return {
        "doc_id": list(batch["doc_id"]),
        "sentences_json": [json.dumps(s) for s in sents],
    }


class RayStage2GPU:
    """GPU stage as a Ray actor — model loads once in __init__."""

    def __init__(self):
        self.pipe = Stage2GPUInfer(device=0)

    def __call__(self, batch):
        sents = [json.loads(s) for s in batch["sentences_json"]]
        out = self.pipe(sents)
        return {
            "doc_id": list(batch["doc_id"]),
            "infer_json": [json.dumps(r) for r in out],
        }


def ray_stage3(batch: Dict[str, np.ndarray]) -> Dict[str, list]:
    rows = [json.loads(s) for s in batch["infer_json"]]
    summaries = stage3_aggregate(rows)
    return {
        "doc_id": list(batch["doc_id"]),
        "infer_json": list(batch["infer_json"]),
        "n_entities": [s["n_entities"] for s in summaries],
        "n_sentences": [s["n_sentences"] for s in summaries],
        "n_positive": [s["n_positive"] for s in summaries],
        "mean_score": [s["mean_score"] for s in summaries],
        "top_entity_type": [s["top_entity_type"] for s in summaries],
    }

# COMMAND ----------

with gpu_sampler() as ray_gpu:
    ray_t0 = time.perf_counter()

    pdf = spark.table(DOCS_TABLE).select("doc_id", "content").toPandas()
    ds = (
        ray.data.from_pandas(pdf)
        .map_batches(ray_stage1, batch_size=128, num_cpus=2, concurrency=2)
        .map_batches(RayStage2GPU, batch_size=64, num_gpus=1, concurrency=1)
        .map_batches(ray_stage3, batch_size=128, num_cpus=2, concurrency=2)
    )
    out_pdf = ds.to_pandas()
    ray_compute_s = time.perf_counter() - ray_t0

    _infer = [json.loads(s) for s in out_pdf["infer_json"]]
    out_pdf["entities"] = [r["entities"] for r in _infer]
    out_pdf["labels"] = [r["labels"] for r in _infer]
    out_pdf["scores"] = [[float(x) for x in r["scores"]] for r in _infer]
    out_pdf = out_pdf.drop(columns=["infer_json"])

    (
        spark.createDataFrame(out_pdf)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(RAY_RESULTS_TABLE)
    )
    ray_total_s = time.perf_counter() - ray_t0

ray_rows = spark.table(RAY_RESULTS_TABLE).count()
print(f"Ray:   {ray_rows:,} rows in {ray_total_s:.1f}s "
      f"({ray_rows / ray_total_s:.1f} rows/s; compute-only {ray_compute_s:.1f}s)")

# COMMAND ----------

ray.shutdown()

# COMMAND ----------

# MAGIC %md ## 7. Comparison — wall time + throughput

# COMMAND ----------

summary = pd.DataFrame([
    {
        "framework": "Spark (3x Pandas UDF)",
        "rows": spark_rows,
        "wall_time_s": round(spark_total_s, 1),
        "rows_per_s": round(spark_rows / spark_total_s, 1),
    },
    {
        "framework": "Ray (3x map_batches)",
        "rows": ray_rows,
        "wall_time_s": round(ray_total_s, 1),
        "rows_per_s": round(ray_rows / ray_total_s, 1),
    },
])
display(spark.createDataFrame(summary))

# COMMAND ----------

# MAGIC %md ## 8. Comparison — GPU utilization

# COMMAND ----------

import matplotlib.pyplot as plt


def _plot(ax, samples, title):
    if not samples:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        ax.set_title(title)
        return
    arr = np.array(samples)
    ax.plot(arr[:, 0], arr[:, 1], label="GPU %", linewidth=1.0)
    ax.plot(arr[:, 0], arr[:, 2], label="GPU mem %", linewidth=1.0, alpha=0.6)
    ax.set_ylim(0, 105)
    ax.set_xlabel("elapsed (s)")
    ax.set_ylabel("utilization %")
    ax.set_title(f"{title} (mean GPU={arr[:, 1].mean():.1f}%)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)


fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
_plot(axes[0], spark_gpu, "Spark — 3x Pandas UDF")
_plot(axes[1], ray_gpu, "Ray — 3x map_batches")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## What each Spark workaround replaces
# MAGIC | Need                                      | Spark workaround                          | Ray equivalent              |
# MAGIC | ----------------------------------------- | ----------------------------------------- | --------------------------- |
# MAGIC | Load the model once per process           | Module-global dict + lazy init            | Actor `__init__`            |
# MAGIC | Don't re-execute the GPU UDF              | `asNondeterministic()`                    | n/a — actors run once       |
# MAGIC | Pin GPU work to one task                   | `repartition(1)` before the UDF           | `num_gpus=1` on map_batches |
# MAGIC | Let CPU stages parallelize independently  | Re-shuffle around the GPU UDF             | `num_cpus=N` per stage      |
# MAGIC | Control batch size                         | `arrow.maxRecordsPerBatch` (5k cap, no min) | `batch_size=N` per stage   |
