from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

def main(limit, partitions, name):
    spark = SparkSession.builder.config("spark.driver.memory", "16G") .master("local[16]").appName('spark-stats').getOrCreate() 
    df = spark.read.parquet("training/laion-400m/")

    df = df.orderBy(rand()).limit(limit)
    writer = df.repartition(partitions).write
    writer.mode("overwrite").parquet(name)

main(100_000, 200, "training/laion-100k-meta/")
# main(20, 1, "training/laion-20-meta/")
