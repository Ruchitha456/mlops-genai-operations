# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: MLflow Experiment Tracking & Model Registry
# MAGIC
# MAGIC In this module, we structure ML experiments in a production-style MLflow setup.
# MAGIC
# MAGIC We compare:
# MAGIC
# MAGIC * Statistical baseline (Z-score anomaly detection)
# MAGIC * Machine learning model (Isolation Forest)
# MAGIC
# MAGIC We track:
# MAGIC
# MAGIC * metrics
# MAGIC * parameters
# MAGIC * model artifacts
# MAGIC * reproducible experiment runs
# MAGIC
# MAGIC We also prepare the model outputs for downstream registry and deployment.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports 

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from mlflow.models.signature import infer_signature
from scipy.stats import zscore
from datetime import datetime, timezone

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

df = spark.table("features_operations")
pdf = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Z-Score Anomaly Detection

# COMMAND ----------

pdf["z_score"] = pdf.groupby("service_name")["resolution_time_mins"] \
                    .transform(lambda x: zscore(x))

threshold = 3
pdf["z_anomaly"] = (abs(pdf["z_score"]) > threshold).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering 

# COMMAND ----------

features = [
    "resolution_time_mins",
    "severity",
    "repeat_incident",
    "lag_1d",
    "lag_7d",
    "lag_30d",
    "rolling_mean_7d",
    "rolling_std_7d",
    "rolling_mean_14d",
    "rolling_std_14d",
    "rolling_mean_30d",
    "rolling_std_30d",
    "severity_x_resolution",
    "z_score"
]

X = pdf[features].astype("float64").fillna(0)
y_true = pdf["sla_breached"]

# COMMAND ----------

# MAGIC %md
# MAGIC ###  MLflow Experiment Setup 

# COMMAND ----------

mlflow.set_experiment("/Users/druchitha49234@gmail.com/mlops-genai-anomaly")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parent Run

# COMMAND ----------

with mlflow.start_run(run_name="anomaly_detection_parent") as parent_run:

    mlflow.set_tag("domain", "operations")
    mlflow.set_tag("author", "druchitha")
    mlflow.set_tag("version", "1.0")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Z-Score Model Run

# COMMAND ----------

with mlflow.start_run(run_name="z_score_model", nested=True):

        mlflow.log_param("method", "z_score")

        y_pred_z = pdf["z_anomaly"]

        mlflow.log_metric("precision", precision_score(y_true, y_pred_z))
        mlflow.log_metric("recall", recall_score(y_true, y_pred_z))
        mlflow.log_metric("f1", f1_score(y_true, y_pred_z))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Isolation Forest  Model Run

# COMMAND ----------

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

pdf["anomaly_score"] = model.decision_function(X)
pdf["if_anomaly"] = model.predict(X)
pdf["if_anomaly"] = pdf["if_anomaly"].map({1: 0, -1: 1})

with mlflow.start_run(run_name="isolation_forest", nested=True):

        mlflow.log_param("model", "IsolationForest")
        mlflow.log_param("contamination", 0.05)

        y_pred_if = pdf["if_anomaly"]

        mlflow.log_metric("precision", precision_score(y_true, y_pred_if))
        mlflow.log_metric("recall", recall_score(y_true, y_pred_if))
        mlflow.log_metric("f1", f1_score(y_true, y_pred_if))

        # Model signature (M3 requirement)
        signature = infer_signature(X, model.predict(X))
        input_example = X.head(5)

        mlflow.sklearn.log_model(
            model,
            name="isolation_forest_model",
            input_example=input_example,
            signature=signature
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Scores To Delta Table

# COMMAND ----------

with mlflow.start_run(run_name="save_scores", nested=True):

        pdf["is_anomaly"] = pdf["if_anomaly"].astype(bool)
        pdf["anomaly_type"] = "isolation_forest"
        pdf["model_name"] = "isolation_forest_v1"
        pdf["scored_at"] = datetime.now(timezone.utc)

        result_df = spark.createDataFrame(pdf)

        result_df.write.format("delta") \
            .mode("overwrite") \
            .saveAsTable("anomaly_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Selection Summary
# MAGIC
# MAGIC Two models were evaluated for anomaly detection:
# MAGIC
# MAGIC - Z-Score anomaly detection (statistical baseline)
# MAGIC - Isolation Forest (machine learning model)
# MAGIC
# MAGIC Isolation Forest demonstrated better overall performance due to improved recall and a more balanced precision–recall trade-off.
# MAGIC
# MAGIC
# MAGIC ## Final Decision
# MAGIC
# MAGIC - **Champion Model:** Isolation Forest (best overall performance)
# MAGIC - **Challenger Model:** Z-Score method (baseline statistical approach)
# MAGIC
# MAGIC Isolation Forest is more suitable for operational anomaly detection where missing incidents is costly.
# MAGIC
# MAGIC
# MAGIC ## Model Registry Note
# MAGIC
# MAGIC Due to environment constraints, model registration to Unity Catalog Model Registry was not performed in this notebook.
# MAGIC
# MAGIC However, full MLflow experiment tracking was completed successfully, including:
# MAGIC
# MAGIC - Nested runs for baseline and ML model comparison  
# MAGIC - Logging of precision, recall, and F1-score metrics  
# MAGIC - Model artifact logging for Isolation Forest  
# MAGIC - Reproducible experiment structure with parent–child runs  
# MAGIC
# MAGIC The Isolation Forest model is therefore selected as the final candidate for downstream production and evaluation in subsequent modules.