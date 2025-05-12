import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, regexp_replace, isnan, count, corr, round
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sparkxgb import XGBoostClassifier
import mlflow
import mlflow.spark
import sys # Import the sys module


# Step 1: Initialize Spark Session
spark = SparkSession.builder.appName("InsuranceClaimMLPipeline").getOrCreate()   ## This step needs setup

# Step 2: Load Dataset
print("Reading CSV file from data/insurance.csv...")
csv_path = "data/insurance.csv"  # Define the path explicitly
print(f"Current working directory: {os.getcwd()}")  # Print the current working directory
print(f"Checking for file existence: {os.path.exists(csv_path)}") # Print the result of the check
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    print(f"List of files in current directory: {os.listdir()}") # List files
    exit(1)  # Exit if the file doesn't exist

df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Step 3: Clean and Preprocess Data
print("Cleaning and preprocessing data...")
string_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == 'string']
for col_name in string_cols:
    df = df.withColumn(col_name, trim(col(col_name)))
    df = df.withColumn(col_name, when(col(col_name) == "NA", None).otherwise(col(col_name)))

df = df.withColumn("cost", regexp_replace("cost", "\\$", "").cast(DoubleType()))
df = df.withColumn("label", when(col("claim_status") == "Approved", 1).otherwise(0))
df = df.dropna(subset=["age", "gender", "diagnosis", "procedure", "claim_status"])

# Step 4: Exploratory Data Analysis (EDA)
print("Exploratory Data Analysis (EDA)...")
print("Basic Statistics for Numerical Features:")
df.select("age", "cost").describe().show()

print("Missing Values by Column:")
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

print("Correlation Between Numerical Features:")
correlations = [(x, y, df.stat.corr(x, y)) for x in ["age", "cost"] for y in ["age", "cost"] if x != y]
for x, y, corr_val in correlations:
    print(f"Correlation between {x} and {y}: {round(corr_val, 4)}")

# Step 5: Impute Missing Values
print("Imputing missing values...")
imputer = Imputer(inputCols=["age", "cost"], outputCols=["age", "cost"])
df = imputer.fit(df).transform(df)

# Step 6: Feature Engineering
print("Engineering features...")
categorical_cols = ["gender", "diagnosis", "procedure"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe") for col in categorical_cols]
feature_cols = ["age", "cost"] + [col + "_ohe" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# Step 7: Define and Compare Candidate Models
print("Defining and comparing candidate models...")
models = {
    "RandomForest": RandomForestClassifier(labelCol="label", featuresCol="features", seed=42),
    "GBT": GBTClassifier(labelCol="label", featuresCol="features", seed=42),
    "LogisticRegression": LogisticRegression(labelCol="label", featuresCol="features"),
    "DecisionTree": DecisionTreeClassifier(labelCol="label", featuresCol="features"),
    "XGBoost": XGBoostClassifier(labelCol="label", featuresCol="features", missing=0.0, numWorkers=4)
}

train, test = df.randomSplit([0.8, 0.2], seed=42)
roc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

best_model_name, best_auc = None, 0.0
for name, estimator in models.items():
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, estimator])
    model = pipeline.fit(train)
    predictions = model.transform(test) # Make predictions on the test set.
    auc = roc_evaluator.evaluate(predictions)
    print(f"Model: {name}, AUC: {auc:.4f}")
    if auc > best_auc:
        best_model_name, best_auc = name, auc

print(f"\nBest Performing Model: {best_model_name} with AUC = {best_auc:.4f}")

# Step 9: Define Hyperparameter Grids for Tuning
print("Defining hyperparameter grids for tuning...")
if best_model_name == "RandomForest":
    clf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)
    grid = ParamGridBuilder() \
        .addGrid(clf.numTrees, [50, 100]) \
        .addGrid(clf.maxDepth, [5, 10]) \
        .build()
elif best_model_name == "GBT":
    clf = GBTClassifier(labelCol="label", featuresCol="features", seed=42)
    grid = ParamGridBuilder() \
        .addGrid(clf.maxDepth, [3, 5]) \
        .addGrid(clf.maxIter, [50, 100]) \
        .build()
elif best_model_name == "LogisticRegression":
    clf = LogisticRegression(labelCol="label", featuresCol="features")
    grid = ParamGridBuilder() \
        .addGrid(clf.regParam, [0.01, 0.1]) \
        .addGrid(clf.elasticNetParam, [0.0, 0.5]) \
        .build()
elif best_model_name == "XGBoost":
    clf = XGBoostClassifier(labelCol="label", featuresCol="features", missing=0.0, numWorkers=4)
    grid = ParamGridBuilder() \
        .addGrid(clf.maxDepth, [3, 5]) \
        .addGrid(clf.eta, [0.1, 0.3]) \
        .addGrid(clf.numRound, [100]) \
        .build()
else:
    clf = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    grid = ParamGridBuilder() \
        .addGrid(clf.maxDepth, [5, 10]) \
        .build()

# Step 10: Tune Best Model with Cross Validation
print("Tuning the best model with cross-validation...")
final_pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, clf])
cv = CrossValidator(estimator=final_pipeline,
                    estimatorParamMaps=grid,
                    evaluator=roc_evaluator,
                    numFolds=5,
                    parallelism=2,
                    seed=42)

with mlflow.start_run():
    tuned_model = cv.fit(train)
    predictions = tuned_model.transform(test)
    auc = roc_evaluator.evaluate(predictions)
    acc = accuracy_evaluator.evaluate(predictions)

    mlflow.log_param("model", best_model_name)
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("accuracy", acc)
    mlflow.spark.log_model(tuned_model.bestModel, "insurance_model")

    print("\nFinal Model Performance:")
    print(f"Model: {best_model_name}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

# Step 11: Save Model and Show Output
print("Saving the tuned model...")
tuned_model.bestModel.write().overwrite().save("models/best_claim_model")

print("Sample of predictions:")
predictions.select("label", "prediction", "probability").show(10, truncate=False)
