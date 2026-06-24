# Databricks notebook source
# MAGIC %md
# MAGIC # Spark vs Ray — Heterogeneous NLP (optimized)
# MAGIC
# MAGIC Same three CPU → GPU → CPU stages as the base notebook, tuned on
# MAGIC both sides for a fair fight:
# MAGIC
# MAGIC - **Spark**: `pandas_udf` → `mapInPandas` → `pandas_udf`. Switching
# MAGIC   stage 2 to `mapInPandas` collapses the Arrow batch boundaries that
# MAGIC   were forcing model reloads — one Python invocation per partition,
# MAGIC   model stays warm across batches. Closest Spark gets to Ray's actor
# MAGIC   model.
# MAGIC - **Ray**: three `map_batches` calls. The GPU actor emits raw model
# MAGIC   outputs and pushes per-doc unpacking onto stage 3 (CPU), so the
# MAGIC   GPU returns to forward passes instead of doing list-comprehension
# MAGIC   work between batches.
# MAGIC
# MAGIC Pair with `heterogeneous_nlp_demo_base` to see what each optimization
# MAGIC actually buys.
# MAGIC
# MAGIC | Stage              | Bound | Work                                              |
# MAGIC | ------------------ | ----- | ------------------------------------------------- |
# MAGIC | preprocess         | CPU   | sentence splitting + cleanup                      |
# MAGIC | gpu_infer          | GPU   | biomedical NER + sentiment classification         |
# MAGIC | aggregate          | CPU   | per-doc summary                                   |
# MAGIC
# MAGIC The Python core (`stage1_preprocess`, `Stage2GPUInfer`, `unpack_raw`,
# MAGIC `stage3_aggregate`) is identical to the base. Only the wrappers and
# MAGIC where postprocessing runs differ.
# MAGIC
# MAGIC Outputs:
# MAGIC - `cjc.ml.spark_vs_ray_docs`           — input (10k pubmed_qa rows)
# MAGIC - `cjc.ml.spark_vs_ray_spark_results`  — Spark job output
# MAGIC - `cjc.ml.spark_vs_ray_ray_results`    — Ray job output

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
VARIANT = "optimized"  # distinguishes outputs from the base notebook

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
# MAGIC
# MAGIC 10k synthetic-question rows from `pubmed_qa/pqa_artificial`, projected
# MAGIC to `(doc_id, content)` and written to Delta. Skipped on rerun.

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
# MAGIC
# MAGIC Plain Python. The Spark and Ray sections below import these — they are
# MAGIC the only place inference logic lives.

# COMMAND ----------

import re
import time
from typing import Dict, List

import numpy as np
import torch

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---- Stage 1 — CPU preprocess ----------------------------------------------
def stage1_preprocess(texts: List[str]) -> List[List[str]]:
    """CPU: split each doc into sentences, drop empties, hard-truncate."""
    out = []
    for t in texts:
        sents = [s.strip()[:1000] for s in _SENT_SPLIT.split(t or "") if s.strip()]
        out.append(sents or [""])
    return out


# ---- Stage 2 — GPU inference (NER + CLF) -----------------------------------
class Stage2GPUInfer:
    """GPU: biomedical NER + sentiment classification.

    Returns RAW model outputs only — no per-doc dict construction. That work
    happens in `unpack_raw` and is intentionally pushed onto a CPU stage so
    the GPU actor (Ray) / executor (Spark) returns to GPU work immediately.
    """

    def __init__(self, device: int = 0):
        from transformers import pipeline as hf_pipeline

        self.ner = hf_pipeline(
            "ner", model=NER_MODEL, device=device, aggregation_strategy="simple"
        )
        self.clf = hf_pipeline("text-classification", model=CLF_MODEL, device=device)

    def __call__(self, per_doc_sents: List[List[str]]) -> Dict:
        flat: List[str] = []
        offsets: List[int] = [0]
        for sents in per_doc_sents:
            flat.extend(sents)
            offsets.append(len(flat))

        # Bigger inner batch sizes amortize CUDA-launch + Python overhead.
        ner_raw = self.ner(flat, batch_size=128) if flat else []
        clf_raw = self.clf(flat, batch_size=128) if flat else []
        return {"ner_raw": ner_raw, "clf_raw": clf_raw, "offsets": offsets}


def unpack_raw(stage2_out: Dict) -> List[Dict]:
    """CPU: turn flat NER/CLF outputs into per-doc dicts. Cheap, but we keep
    it off the GPU actor so the GPU isn't waiting on Python list-comp work.
    """
    ner_raw = stage2_out["ner_raw"]
    clf_raw = stage2_out["clf_raw"]
    offsets = stage2_out["offsets"]
    out = []
    for i in range(len(offsets) - 1):
        s, e = offsets[i], offsets[i + 1]
        ents = []
        for sent_ents in ner_raw[s:e]:
            ents.extend(
                {"text": str(x.get("word", "")), "type": str(x.get("entity_group", ""))}
                for x in (sent_ents or [])
            )
        doc_clf = clf_raw[s:e] or []
        out.append({
            "entities": ents,
            "labels": [str(c.get("label", "")) for c in doc_clf],
            "scores": [float(c.get("score", 0.0)) for c in doc_clf],
        })
    return out


# ---- Stage 3 — CPU aggregate -----------------------------------------------
def stage3_aggregate(per_doc_results: List[Dict]) -> List[Dict]:
    """CPU: roll sentence-level outputs up to per-doc summary stats."""
    summaries = []
    for r in per_doc_results:
        # Use `is None` checks — `or []` errors on numpy arrays (Arrow→pandas
        # converts struct fields to arrays, not lists).
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

# MAGIC %md ### Smoke test the three stages on 3 docs

# COMMAND ----------

assert torch.cuda.is_available(), "This demo requires a GPU node."
_sample_texts = [r["content"] for r in spark.table(DOCS_TABLE).limit(3).toPandas().to_dict("records")]

_t0 = time.perf_counter()
_sents = stage1_preprocess(_sample_texts)
_t1 = time.perf_counter()
_infer = Stage2GPUInfer(device=0)
_raw = _infer(_sents)
_t2 = time.perf_counter()
_unpacked = unpack_raw(_raw)
_summary = stage3_aggregate(_unpacked)
_t3 = time.perf_counter()

print(f"stage1 (CPU pre): {(_t1 - _t0) * 1000:.1f} ms — sentence counts: {[len(s) for s in _sents]}")
print(f"stage2 (GPU):     {(_t2 - _t1) * 1000:.1f} ms — first doc entities: {_unpacked[0]['entities'][:3]}")
print(f"stage3 (CPU agg): {(_t3 - _t2) * 1000:.1f} ms — summaries: {_summary}")

del _infer
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## 4. GPU utilization sampler
# MAGIC
# MAGIC Background-thread context manager polling `pynvml` every 250 ms. Returns
# MAGIC `(elapsed_s, gpu_pct, mem_pct)` samples for plotting.

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

# MAGIC %md ## 5. Spark — optimized
# MAGIC
# MAGIC Two changes from the base notebook:
# MAGIC
# MAGIC - **Stage 2 is `mapInPandas`** instead of a `pandas_udf`. Each
# MAGIC   `pandas_udf` invocation is one Arrow batch and the model only stays
# MAGIC   warm via a module-global singleton. `mapInPandas` ships the whole
# MAGIC   partition as an iterator into one Python call, so the model loads
# MAGIC   once and the GPU stays warm across batches. As close to Ray's actor
# MAGIC   model as Spark gets.
# MAGIC - **Stage 2 yields raw NER/CLF outputs**, with per-doc unpacking
# MAGIC   pushed into stage 3 on CPU. The executor returns to GPU work right
# MAGIC   after `forward()` instead of doing list-comp postprocessing.
# MAGIC
# MAGIC The two remaining Spark workarounds aren't going anywhere — they're
# MAGIC inherent to its resource model:
# MAGIC
# MAGIC 1. **`maxRecordsPerBatch=1000`** to cap in-flight memory. The 5,000
# MAGIC    cap is a hard ceiling and there's no minimum knob.
# MAGIC 2. **`repartition(1)`** before stage 2 — manual GPU affinity, since
# MAGIC    Spark's resource model can't pin GPU-bound tasks without also
# MAGIC    capping CPU usage on the node.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType, DoubleType, IntegerType, LongType, StringType, StructField, StructType,
)
import pandas as pd

# Force small batches — checkpoint size trick (5k cap).
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")

# ---- Schemas ---------------------------------------------------------------
_SENTS_T = ArrayType(StringType())
_ENT_T = StructType([
    StructField("text", StringType()),
    StructField("type", StringType()),
])
_SUMMARY_T = StructType([
    StructField("n_entities", IntegerType()),
    StructField("n_sentences", IntegerType()),
    StructField("n_positive", IntegerType()),
    StructField("mean_score", DoubleType()),
    StructField("top_entity_type", StringType()),
])
# Output schema for the mapInPandas stage 2.
_STAGE2_OUT = StructType([
    StructField("doc_id", LongType()),
    StructField("entities", ArrayType(_ENT_T)),
    StructField("labels", ArrayType(StringType())),
    StructField("scores", ArrayType(DoubleType())),
])


# ---- Stage 1: pandas_udf (CPU, fine as-is) ---------------------------------
@pandas_udf(_SENTS_T)
def udf_stage1_preprocess(content: pd.Series) -> pd.Series:
    return pd.Series(stage1_preprocess(content.tolist()))


# ---- Stage 2: mapInPandas — model loads ONCE per partition -----------------
def stage2_mapinpandas(iterator):
    """Single Python invocation per Spark partition. The model is loaded
    on the first batch and stays warm for every subsequent batch the
    iterator yields — no Arrow round-trip and no module-global singleton
    gymnastics in between.
    """
    pipe = Stage2GPUInfer(device=0)  # one load, not one-per-batch
    for pdf in iterator:
        sents = [list(s) for s in pdf["sentences"]]
        raw = pipe(sents)
        unpacked = unpack_raw(raw)
        yield pd.DataFrame({
            "doc_id": pdf["doc_id"].values,
            "entities": [u["entities"] for u in unpacked],
            "labels": [u["labels"] for u in unpacked],
            "scores": [u["scores"] for u in unpacked],
        })


# ---- Stage 3: pandas_udf (CPU, summary stats) ------------------------------
@pandas_udf(_SUMMARY_T)
def udf_stage3_aggregate(nlp: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(stage3_aggregate(nlp.to_dict("records")))

# COMMAND ----------

with gpu_sampler() as spark_gpu:
    spark_t0 = time.perf_counter()
    stage1_df = (
        spark.table(DOCS_TABLE)
        .withColumn("sentences", udf_stage1_preprocess(F.col("content")))
        .select("doc_id", "sentences")
        .repartition(1)  # GPU-affinity hack: serialize stage 2 onto 1 task
    )
    stage2_df = stage1_df.mapInPandas(stage2_mapinpandas, schema=_STAGE2_OUT)
    spark_df = (
        stage2_df
        .withColumn(
            "summary",
            udf_stage3_aggregate(F.struct("entities", "labels", "scores")),
        )
        .select(
            "doc_id", "entities", "labels", "scores",
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
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## 6. Ray — optimized
# MAGIC
# MAGIC Three changes from the base:
# MAGIC
# MAGIC - **Stage 2 emits raw NER/CLF outputs** and stage 3 does the
# MAGIC   per-doc unpacking on CPU. The GPU actor's `__call__` becomes a
# MAGIC   thin shell around two `forward()`s — postprocessing list-comp time
# MAGIC   no longer counts as GPU-idle time.
# MAGIC - **Inner HF batch_size=128** (was 32/64). Each forward pass amortizes
# MAGIC   more CUDA-launch overhead; the T4's 16GB swallows it fine.
# MAGIC - **Outer batch_size=256** for the GPU stage. Fewer trips into the
# MAGIC   actor, longer GPU bursts between Python returns.
# MAGIC
# MAGIC `num_gpus=1, concurrency=1` for stage 2 pins one actor to the GPU and
# MAGIC runs `__init__` once. CPU stages stay at `num_cpus=2, concurrency=2`.

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


class _NpEncoder(json.JSONEncoder):
    """HuggingFace pipelines return numpy scalars; json doesn't know them."""
    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder)


# Intermediate columns are JSON strings — Arrow object-extension types don't
# round-trip cleanly across stage boundaries on this runtime.


def ray_stage1(batch: Dict[str, np.ndarray]) -> Dict[str, list]:
    sents = stage1_preprocess([str(t) for t in batch["content"]])
    return {
        "doc_id": list(batch["doc_id"]),
        "sentences_json": [_dumps(s) for s in sents],
    }


class RayStage2GPU:
    """GPU actor — emits RAW model outputs only. No per-doc unpacking here:
    that's CPU work and we want this actor back on the GPU immediately.
    """

    def __init__(self):
        self.pipe = Stage2GPUInfer(device=0)

    def __call__(self, batch):
        sents = [json.loads(s) for s in batch["sentences_json"]]
        # One JSON string per *batch* containing flat ner_raw / clf_raw /
        # offsets — minimal Python work, GPU goes back to work right away.
        raw = self.pipe(sents)
        # Slice the raw output back into one JSON string per doc so stage 3
        # can parallelize per-row. The slice itself is just dict slicing,
        # which is fast.
        offsets = raw["offsets"]
        per_doc_raw = []
        for i in range(len(offsets) - 1):
            s, e = offsets[i], offsets[i + 1]
            per_doc_raw.append(_dumps({
                "ner_raw": raw["ner_raw"][s:e],
                "clf_raw": raw["clf_raw"][s:e],
                "offsets": [0, e - s],
            }))
        return {"doc_id": list(batch["doc_id"]), "raw_json": per_doc_raw}


def ray_stage3(batch: Dict[str, np.ndarray]) -> Dict[str, list]:
    """CPU: unpack raw model outputs and aggregate to per-doc summaries.
    Both unpack_raw and stage3_aggregate run here so the GPU actor doesn't.
    Output columns stay scalar-or-string so Ray Data's to_pandas() doesn't
    hit the PythonObject extension-type bug on this runtime.
    """
    raws = [json.loads(s) for s in batch["raw_json"]]
    unpacked = [unpack_raw(r)[0] for r in raws]
    summaries = stage3_aggregate(unpacked)
    return {
        "doc_id": list(batch["doc_id"]),
        "entities_json": [_dumps(u["entities"]) for u in unpacked],
        "labels_json": [_dumps(u["labels"]) for u in unpacked],
        "scores_json": [_dumps(u["scores"]) for u in unpacked],
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
        .map_batches(RayStage2GPU, batch_size=256, num_gpus=1, concurrency=1)
        .map_batches(ray_stage3, batch_size=128, num_cpus=2, concurrency=2)
    )
    out_pdf = ds.to_pandas()
    ray_compute_s = time.perf_counter() - ray_t0

    out_pdf["entities"] = [json.loads(s) for s in out_pdf["entities_json"]]
    out_pdf["labels"] = [json.loads(s) for s in out_pdf["labels_json"]]
    out_pdf["scores"] = [[float(x) for x in json.loads(s)] for s in out_pdf["scores_json"]]
    out_pdf = out_pdf.drop(columns=["entities_json", "labels_json", "scores_json"])

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
        "framework": "Spark (UDF + mapInPandas + UDF)",
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
# MAGIC
# MAGIC Each plot is GPU% over the wall-clock of one job. Look for **valleys** —
# MAGIC GPU idle time. Spark valleys come from:
# MAGIC - shuffle / repartition between stages
# MAGIC - Arrow batch (de)serialization at UDF boundaries
# MAGIC - the lazy model load on the first batch
# MAGIC
# MAGIC Ray's middle stage is one long-lived actor; the outer two CPU stages
# MAGIC stream into and out of it without GPU involvement.

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

# MAGIC %md ## Side-by-side
# MAGIC | Need                                       | Spark approach                                              | Ray approach                       |
# MAGIC | ------------------------------------------ | ----------------------------------------------------------- | ---------------------------------- |
# MAGIC | Load the model once per process            | `mapInPandas` (one invocation per partition)                | Actor `__init__`                   |
# MAGIC | Don't re-execute the GPU work              | n/a — `mapInPandas` is naturally one-shot                   | n/a — actors run once              |
# MAGIC | Pin GPU work to one task                   | `repartition(1)` before stage 2                             | `num_gpus=1` on map_batches        |
# MAGIC | Let CPU stages parallelize independently   | Outer pandas_udfs run on default partitioning               | `num_cpus=N` per stage             |
# MAGIC | Control batch size                          | `arrow.maxRecordsPerBatch` (5k cap, no min)                 | `batch_size=N` per stage           |
# MAGIC | Keep GPU off Python postprocessing         | Stage 2 unpacks inline — `mapInPandas` keeps it warm anyway | Stage 2 emits raw, stage 3 unpacks |

# COMMAND ----------

# MAGIC %md ## 10. Heterogeneous Ray clusters
# MAGIC
# MAGIC This demo runs on a single node carved into role-typed actor pools.
# MAGIC At production scale you want **physical** separation: CPU stages on
# MAGIC cheap CPU-only workers, GPU stages on GPU workers, all one job. Ray's
# MAGIC resource model handles this; Spark's doesn't, because pinning a GPU
# MAGIC on an executor pins the executor's CPUs with it.
# MAGIC
# MAGIC ### Provisioning
# MAGIC
# MAGIC Two node pools in the bundle, Ray on top:
# MAGIC
# MAGIC ```yaml
# MAGIC # databricks.yml — two job clusters
# MAGIC job_clusters:
# MAGIC   - job_cluster_key: gpu_pool
# MAGIC     new_cluster:
# MAGIC       node_type_id: Standard_NC4as_T4_v3
# MAGIC       num_workers: 4
# MAGIC   - job_cluster_key: cpu_pool
# MAGIC     new_cluster:
# MAGIC       node_type_id: Standard_D8s_v5
# MAGIC       num_workers: 8
# MAGIC ```
# MAGIC
# MAGIC ```python
# MAGIC from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
# MAGIC
# MAGIC ray.shutdown()                      # clear any stale driver context
# MAGIC try:
# MAGIC     shutdown_ray_cluster()          # and any prior Ray-on-Spark cluster
# MAGIC except Exception:
# MAGIC     pass
# MAGIC
# MAGIC conn, _ = setup_ray_cluster(
# MAGIC     min_worker_nodes=12, max_worker_nodes=12,  # fixed: from_spark() can't autoscale
# MAGIC     num_cpus_worker_node=8,
# MAGIC     num_gpus_worker_node=1,         # Ray only schedules GPU work on GPU nodes
# MAGIC     collect_log_to_path="/Volumes/<cat>/<schema>/<vol>/ray_logs",
# MAGIC )                                   # no head-node compute in hybrid mode
# MAGIC ray.init(address=conn, ignore_reinit_error=True)  # connect — never a bare ray.init()
# MAGIC # ... run the pipeline, then:
# MAGIC shutdown_ray_cluster()              # or the collected logs never copy out
# MAGIC ```
# MAGIC
# MAGIC The pipeline code is unchanged. `.map_batches(RayStage2GPU, num_gpus=1)`
# MAGIC schedules across the 4-node GPU pool; `ray_stage1` and `ray_stage3`
# MAGIC land on the 8-node CPU pool. Ray Data streams batches between them.
# MAGIC
# MAGIC ### Feeding Spark data into Ray at scale
# MAGIC
# MAGIC This demo does `spark.table(...).toPandas()` → `ray.data.from_pandas(...)`.
# MAGIC At 10k rows that's fine, but `toPandas()` collects the whole table onto
# MAGIC the driver — it won't survive a real workload. Two scale-out swaps, in
# MAGIC increasing order of friction:
# MAGIC
# MAGIC 1. **Isolated (recommended): Spark writes Delta/Parquet, Ray reads it.**
# MAGIC    Spark `.write` the prepared subset, then `ray.data.read_parquet(path)`
# MAGIC    in the Ray stage. No live coupling between the two engines — this is
# MAGIC    the simplest pattern and fits most workloads.
# MAGIC 2. **Hybrid: `ray.data.from_spark(df)`** keeps it in one job but is the
# MAGIC    sharpest edge. It requires, at the cluster level:
# MAGIC    - `spark.databricks.pyspark.dataFrameChunk.enabled true`
# MAGIC    - `spark.task.resource.gpu.amount 0` (don't let Spark hold the GPUs Ray needs)
# MAGIC    - shrink Spark's footprint so Ray has room — e.g. `spark.executor.memory 1g`,
# MAGIC      `spark.driver.memory 1g`; default reservations can starve Ray hard
# MAGIC
# MAGIC    It also does **not** support autoscaling Ray-on-Spark clusters, and
# MAGIC    Databricks-patch / Ray-version skew tends to surface right here
# MAGIC    (`BlockMetadata(... schema=)`, `get_read_tasks(... per_task_row_limit=)`).
# MAGIC    Pin a known-good Ray (e.g. 2.35.0) on the affected runtime.
# MAGIC
# MAGIC Whichever you pick, keep worker functions pure Python (pandas in →
# MAGIC inference → pandas out). `JVM wasn't initialised` / pyspark UDF warnings
# MAGIC inside a Ray worker mean it's reaching back into Spark — pull that out.
# MAGIC And make it boring before fast: 1 GPU actor per physical GPU with
# MAGIC `num_gpus=1` (as stage 2 already does) is the stable baseline; fractional
# MAGIC GPU packing (`num_gpus=1/k`, several models per device) is an
# MAGIC optimization to reach for only once the single-actor version runs clean.
# MAGIC
# MAGIC ### Custom resource tags
# MAGIC
# MAGIC When CPU and GPU counts aren't enough — e.g. a stage needs a fast NIC
# MAGIC or extra RAM — start Ray on those nodes with a custom resource label
# MAGIC and request it per stage:
# MAGIC
# MAGIC ```bash
# MAGIC ray start --resources='{"highmem": 1}'  # on the chosen workers
# MAGIC ```
# MAGIC ```python
# MAGIC ds.map_batches(MyAggregator, num_cpus=4, resources={"highmem": 0.01})
# MAGIC ```
# MAGIC
# MAGIC ### Tradeoffs
# MAGIC
# MAGIC - Two pools to size and autoscale. Spark's autoscaler is per-pool and
# MAGIC   doesn't see Ray's per-stage demand.
# MAGIC - Cross-pool traffic. Ray Data ships intermediate batches between the
# MAGIC   pools — keep them in the same AZ or pay the bandwidth bill.
# MAGIC - Failure handling: a CPU-pool node dropping mid-job only invalidates
# MAGIC   that stage's in-flight batches. Worth testing for your workload.

# COMMAND ----------


