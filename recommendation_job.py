from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

# ✅ 0. Initialize Spark session
spark = SparkSession.builder.appName("ProductRecommendation").getOrCreate()

# ✅ 1. Load User Events CSV from GCS
df = spark.read.csv("gs://product-recommendation-data/user_events.csv", header=True, inferSchema=True)

# ✅ 2. Filter only purchase events
df = df.filter(col("event_type") == "purchase")

# ✅ 3. Index user_id and product_id
user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex").fit(df)
product_indexer = StringIndexer(inputCol="product_id", outputCol="productIndex").fit(df)

df = user_indexer.transform(df)
df = product_indexer.transform(df)

# ✅ 4. Assign dummy rating = 1 (since it's a purchase)
df = df.withColumn("rating", col("event_type").isNotNull().cast("int"))

# ✅ 5. Train ALS recommender model
als = ALS(
    userCol="userIndex",
    itemCol="productIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)
model = als.fit(df)

# ✅ 6. Generate top 5 recommendations per user
recommendations = model.recommendForAllUsers(5)

# ✅ 7. Explode nested recommendation list
exploded = recommendations.withColumn("rec", explode("recommendations")) \
    .select("userIndex", col("rec.productIndex").alias("productIndex"), col("rec.rating").alias("rating"))

# ✅ 8. Map userIndex and productIndex back to original user_id and product_id
user_map = df.select("user_id", "userIndex").distinct()
product_map = df.select("product_id", "productIndex").distinct()

final_df = exploded \
    .join(user_map, on="userIndex", how="left") \
    .join(product_map, on="productIndex", how="left") \
    .select("user_id", "product_id", "rating")

# ✅ 9. Save final recommendations to BigQuery
final_df.write \
    .format("bigquery") \
    .option("table", "reactfirebaseaugus.ecommerce_data_aug.final_user_recommendations") \
    .option("temporaryGcsBucket", "product-recommendation-data") \
    .mode("overwrite") \
    .save()
