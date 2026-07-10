# Distributed XGBoost using PySpark
# PySpark is the enterprise standard for data pipelines.
# XGBoost integrates directly with Spark MLlib pipelines via SparkXGBRegressor/Classifier.

from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand

def main():
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("XGBoost-PySpark-Demo") \
        .master("local[*]") \
        .getOrCreate()
    
    print("Generating dummy data...")
    # Create a dummy DataFrame
    df = spark.range(0, 10000).select("id")
    df = df.withColumn("f1", rand(seed=42)) \
           .withColumn("f2", rand(seed=43)) \
           .withColumn("f3", rand(seed=44))
    
    # Target is 1 if f1 + f2 > 1, else 0
    df = df.withColumn("label", ((df.f1 + df.f2) > 1.0).cast("integer"))
    
    # Split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # In Spark MLlib, features must be assembled into a single vector column
    assembler = VectorAssembler(inputCols=["f1", "f2", "f3"], outputCol="features")

    print("Configuring SparkXGBClassifier...")
    # Define XGBoost model
    xgb_estimator = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        num_workers=2, # Number of Spark executors to use
        max_depth=4,
        eta=0.1
    )

    # Build Pipeline
    pipeline = Pipeline(stages=[assembler, xgb_estimator])

    print("Training Pipeline...")
    model = pipeline.fit(train_df)
    
    print("Making Predictions...")
    predictions = model.transform(test_df)
    
    # Show results
    predictions.select("id", "label", "probability", "prediction").show(5)

    spark.stop()

if __name__ == "__main__":
    main()
