from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("CreditRiskIngestion") \
    .getOrCreate()


raw_df = spark.read.csv("../data/raw/credit_card.csv", header=True, inferSchema=True)
clean_df = raw_df.dropna()
clean_df.write.mode("overwrite").parquet("../data/processed/credit.parquet")
print("Raw row count:", raw_df.count())
