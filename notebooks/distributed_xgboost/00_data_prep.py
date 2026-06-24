# Databricks notebook source

# MAGIC %md
# MAGIC # MovieLens 25M Data Preparation
# MAGIC Downloads the MovieLens 25M dataset (25M ratings, 62K movies, 162K users),
# MAGIC engineers features for rating prediction, and writes to `cjc.ml.movielens_features`.

# COMMAND ----------

import os
import urllib.request
import zipfile
import pandas as pd

tmpdir = "/tmp/movielens"
os.makedirs(tmpdir, exist_ok=True)

url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
zip_path = f"{tmpdir}/ml-25m.zip"
csv_dir = f"{tmpdir}/ml-25m"

if not os.path.exists(f"{csv_dir}/ratings.csv"):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded {os.path.getsize(zip_path) / 1e6:.0f} MB")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    print("Extraction complete.")
else:
    print("Data already exists, skipping download.")

print("Files:", os.listdir(csv_dir))

# COMMAND ----------

print("Reading ratings CSV with pandas on driver...")
ratings_pdf = pd.read_csv(f"{csv_dir}/ratings.csv")
print(f"Ratings shape: {ratings_pdf.shape}")

movies_pdf = pd.read_csv(f"{csv_dir}/movies.csv")
print(f"Movies shape: {movies_pdf.shape}")

# COMMAND ----------

ratings_df = spark.createDataFrame(ratings_pdf)
movies_df = spark.createDataFrame(movies_pdf)

ratings_df.cache()
print(f"Ratings: {ratings_df.count():,} rows")
movies_df.cache()
print(f"Movies: {movies_df.count():,} rows")

del ratings_pdf, movies_pdf

# COMMAND ----------

from pyspark.sql import functions as F

user_stats = ratings_df.groupBy("userId").agg(
    F.avg("rating").alias("user_avg_rating"),
    F.count("rating").alias("user_rating_count"),
    F.stddev("rating").alias("user_rating_std"),
)

movie_stats = ratings_df.groupBy("movieId").agg(
    F.avg("rating").alias("movie_avg_rating"),
    F.count("rating").alias("movie_rating_count"),
    F.stddev("rating").alias("movie_rating_std"),
)

# COMMAND ----------

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

movies_genre = movies_df
for genre in ALL_GENRES:
    col_name = f"genre_{genre.lower().replace('-', '_')}"
    movies_genre = movies_genre.withColumn(
        col_name,
        F.when(F.col("genres").contains(genre), 1).otherwise(0).cast("int"),
    )

# COMMAND ----------

ratings_ts = (
    ratings_df
    .withColumn("ts", F.from_unixtime(F.col("timestamp").cast("long")))
    .withColumn("hour_of_day", F.hour("ts"))
    .withColumn("day_of_week", F.dayofweek("ts"))
    .withColumn("month", F.month("ts"))
    .withColumn("year", F.year("ts"))
    .drop("ts")
)

features_df = (
    ratings_ts
    .join(user_stats, "userId")
    .join(movie_stats, "movieId")
    .join(movies_genre.drop("title", "genres"), "movieId")
    .drop("timestamp")
    .fillna(0)
)

print(f"Feature table: {len(features_df.columns)} columns")
features_df.printSchema()

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS cjc.ml")
features_df.write.mode("overwrite").saveAsTable("cjc.ml.movielens_features")

ratings_df.unpersist()
movies_df.unpersist()

row_count = spark.read.table("cjc.ml.movielens_features").count()
print(f"Written {row_count:,} rows to cjc.ml.movielens_features")
