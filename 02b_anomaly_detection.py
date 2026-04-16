# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2B: Anomaly Detection (Operations Path)
# MAGIC
# MAGIC Chosen Path: Operations (PATH B)
# MAGIC
# MAGIC This module builds and compares statistical and ML-based anomaly detection models for incident monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Module 2B: Anomaly Detection
# MAGIC
# MAGIC This module builds an end-to-end anomaly detection system for operational incident data.
# MAGIC
# MAGIC We compare:
# MAGIC
# MAGIC * A statistical baseline (Z-score method)
# MAGIC * A machine learning model (Isolation Forest)
# MAGIC
# MAGIC The goal is to detect abnormal system behavior that correlates with SLA breaches and operational failures.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Load Data

# COMMAND ----------

df = spark.table("features_operations")

# COMMAND ----------

pdf = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Statistical Baseline (Z-Score)
# MAGIC

# COMMAND ----------

from scipy.stats import zscore

pdf["z_score"] = pdf.groupby("service_name")["resolution_time_mins"] \
                   .transform(lambda x: zscore(x))

# Threshold
threshold = 3

pdf["z_anomaly"] = (abs(pdf["z_score"]) > threshold).astype(int)

# COMMAND ----------

# Evaluation 
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = pdf["sla_breached"]
y_pred = pdf["z_anomaly"]

print("Z-Score Model:")
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Statistical Baseline (Z-Score)
# MAGIC
# MAGIC The Z-score method identifies anomalies based on deviation from service-level mean resolution time.
# MAGIC
# MAGIC **Key observations:**
# MAGIC
# MAGIC * High precision indicates only extreme cases are flagged
# MAGIC * Very low recall shows limited detection coverage
# MAGIC * Suitable only as a conservative baseline method
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. ML Model( Isolation Forest Model)
# MAGIC
# MAGIC We use an unsupervised model to detect anomalies based on feature patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection

# COMMAND ----------

  features = [
    "resolution_time_mins",
    "severity",
    "lag_1d",
    "lag_7d",
    "rolling_mean_7d",
    "rolling_std_7d",
    "severity_x_resolution"
]

X = pdf[features].fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model

# COMMAND ----------

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

pdf["anomaly_score"] = model.decision_function(X)
pdf["if_anomaly"] = model.predict(X)
pdf["if_anomaly"] = pdf["if_anomaly"].map({1:0, -1:1})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation

# COMMAND ----------

y_pred_if = pdf["if_anomaly"]

print("Isolation Forest:")
print("Precision:", precision_score(y_true, y_pred_if))
print("Recall:", recall_score(y_true, y_pred_if))
print("F1:", f1_score(y_true, y_pred_if))

# COMMAND ----------

pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML-Based Model (Isolation Forest)
# MAGIC
# MAGIC Isolation Forest detects anomalies based on feature space separation rather than statistical deviation.
# MAGIC
# MAGIC **Why these features:**
# MAGIC
# MAGIC * Resolution time → core anomaly signal
# MAGIC * Severity → operational impact indicator
# MAGIC * Lag features → historical behavior context
# MAGIC * Rolling stats → stability and volatility detection
# MAGIC * Interaction feature → compound system stress
# MAGIC
# MAGIC **Key insight:**
# MAGIC This model improves recall compared to statistical methods, making it more suitable for operational monitoring systems.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. MLflow — Z-SCORE RUN

# COMMAND ----------

import mlflow

with mlflow.start_run():
    mlflow.log_param("model", "z_score_baseline")
    mlflow.log_metric("precision", precision_score(y_true, y_pred))
    mlflow.log_metric("recall", recall_score(y_true, y_pred))
    mlflow.log_metric("f1", f1_score(y_true, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. MLflow — ISOLATION FOREST

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature

mlflow.set_experiment("/Users/druchitha49234@gmail.com/mlops-genai-anomaly")

with mlflow.start_run() as run:

    run_id = run.info.run_id


    # Model logging
    
    mlflow.log_param("model", "isolation_forest")
    mlflow.log_param("contamination", 0.05)

    mlflow.log_metric("precision", precision_score(y_true, y_pred_if))
    mlflow.log_metric("recall", recall_score(y_true, y_pred_if))
    mlflow.log_metric("f1_score", f1_score(y_true, y_pred_if))

    X = pdf[features].copy().fillna(0).astype("float64")
    input_example = X.head(5)

    signature = infer_signature(X, model.predict(X))

    mlflow.sklearn.log_model(
        model,
        name="isolation_forest_model",
        input_example=input_example,
        signature=signature
    )



# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Tracking
# MAGIC
# MAGIC All experiments are logged in MLflow for reproducibility.
# MAGIC
# MAGIC Tracked metrics:
# MAGIC
# MAGIC * Precision
# MAGIC * Recall
# MAGIC * F1-score
# MAGIC
# MAGIC Two models are compared:
# MAGIC
# MAGIC * Statistical baseline (Z-score)
# MAGIC * Machine learning model (Isolation Forest)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Model Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Z-Score Baseline
# MAGIC
# MAGIC * Very high precision
# MAGIC * Extremely low recall
# MAGIC * Conservative detection approach
# MAGIC
# MAGIC #### Isolation Forest
# MAGIC
# MAGIC * Better recall
# MAGIC * Balanced precision
# MAGIC * More suitable for operational anomaly detection
# MAGIC
# MAGIC #### Final Decision
# MAGIC
# MAGIC Isolation Forest is preferred due to better detection coverage, making it more effective for real-world monitoring systems.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Output Table

# COMMAND ----------

from datetime import datetime, timezone
# Save anomaly scores

pdf["is_anomaly"] = pdf["if_anomaly"].astype(bool)
pdf["anomaly_type"] = "isolation_forest"
pdf["model_name"] = "isolation_forest_v1"
pdf["run_id"] = str(run_id)
pdf["scored_at"] = datetime.now(timezone.utc)

result_df = spark.createDataFrame(pdf)

spark.sql("DROP TABLE IF EXISTS anomaly_scores")

result_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("anomaly_scores")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Module 2B Summary
# MAGIC
# MAGIC In this module, we built and evaluated an end-to-end anomaly detection system for operational incident data.
# MAGIC
# MAGIC ### What was implemented
# MAGIC
# MAGIC * A statistical baseline model using Z-score for anomaly detection
# MAGIC * An unsupervised machine learning model using Isolation Forest
# MAGIC * Evaluation of both models using precision, recall, and F1-score
# MAGIC * MLflow tracking for experiment reproducibility
# MAGIC * Generation of a scored output table (`anomaly_scores`) for downstream use
# MAGIC
# MAGIC ### Key Outcome
# MAGIC
# MAGIC The Isolation Forest model provided better recall compared to the statistical baseline, making it more suitable for real-world operational monitoring where detecting hidden anomalies is critical.
# MAGIC
# MAGIC ### Business Insight
# MAGIC
# MAGIC In production environments:
# MAGIC
# MAGIC * Missing anomalies (false negatives) can lead to SLA breaches and service disruption
# MAGIC * Excessive alerts (false positives) reduce trust in monitoring systems
# MAGIC
# MAGIC Therefore, a balanced model like Isolation Forest is preferred for deployment scenarios.
# MAGIC
# MAGIC ### Next Step
# MAGIC
# MAGIC The next module focuses on MLflow Model Registry and productionizing the best-performing model.
# MAGIC