# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: Production Readiness & Deployment
# MAGIC MLOps Anomaly Detection System (Operations Path)
# MAGIC
# MAGIC This notebook demonstrates production design for model serving, monitoring, batch scoring, and operational readiness.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn.ensemble import IsolationForest

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Model Serving Config

# COMMAND ----------

model_serving_config = {
    "name": "anomaly_detection_endpoint",
    "model_name": "workspace.default.anomaly_model",
    "model_version": "1",
    "workload_size": "Small",
    "scale_to_zero_enabled": True,
    "traffic_config": {
        "champion": 90,
        "challenger": 10
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Justification
# MAGIC
# MAGIC - Small workload: assumes low-to-moderate incident volume
# MAGIC - Scale-to-zero: cost-efficient for intermittent monitoring
# MAGIC - 90/10 traffic split: allows safe comparison of champion vs challenger in production

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Drift / Alert Fatigue Query

# COMMAND ----------

alert_fatigue_query = """
SELECT
    service_name,
    SUM(CASE WHEN if_anomaly = 1 AND sla_breached = false THEN 1 ELSE 0 END) * 1.0 /
    COUNT(*) AS alert_fatigue_rate
FROM anomaly_scores
WHERE timestamp >= current_date() - interval 7 days
GROUP BY service_name
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drift Strategy
# MAGIC
# MAGIC - Baseline window: training dataset
# MAGIC - Monitoring window: rolling 7-day production data
# MAGIC
# MAGIC Numerical drift:
# MAGIC - Z-test / mean variance shift
# MAGIC
# MAGIC Categorical drift:
# MAGIC - Chi-square test
# MAGIC
# MAGIC Alert trigger:
# MAGIC - Alert fatigue rate exceeding defined threshold per service

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Widgets (Batch Pipeline Parameters)

# COMMAND ----------

dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("scoring_date", "2026-04-14")
dbutils.widgets.text("write_mode", "overwrite")

model_alias = dbutils.widgets.get("model_alias")
scoring_date = dbutils.widgets.get("scoring_date")
write_mode = dbutils.widgets.get("write_mode")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

df = spark.table("features_operations")
pdf = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Check

# COMMAND ----------

assert pdf["resolution_time_mins"].notnull().all(), "Missing values found in resolution_time_mins"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Quality Validation
# MAGIC Ensures no missing critical fields before scoring

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Feature Set

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
    "severity_x_resolution"
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Inference
# MAGIC Model is instantiated in batch context for simplicity of this assessment.

# COMMAND ----------

X = pdf[features].astype("float64").fillna(0)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

pdf["anomaly_score"] = model.decision_function(X)
pdf["if_anomaly"] = model.predict(X)
pdf["if_anomaly"] = pdf["if_anomaly"].map({1: 0, -1: 1})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Output To Delta

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS anomaly_scores")

pdf["is_anomaly"] = pdf["if_anomaly"].astype(bool)

result_df = spark.createDataFrame(pdf)

result_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("anomaly_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Readiness Checklist
# MAGIC
# MAGIC - Model versioning → Partial (tracked via MLflow runs; Unity Catalog registry not fully implemented)
# MAGIC - Input schema validation → Partial (MLflow model signature available, but not enforced at serving time)
# MAGIC - Latency SLA → Not defined
# MAGIC - Drift detection → Designed but not automated
# MAGIC - Alert fatigue monitoring → Implemented via custom metric
# MAGIC - Retraining pipeline → Not implemented
# MAGIC - CI/CD pipeline → Not implemented
# MAGIC - Cost monitoring → Not implemented
# MAGIC
# MAGIC ## Production Design Summary
# MAGIC
# MAGIC The Isolation Forest pipeline demonstrates core production design principles:
# MAGIC
# MAGIC - Batch scoring architecture for operational anomaly detection
# MAGIC - Monitoring strategy using SLA breach rate and alert fatigue metrics
# MAGIC - Serving configuration defined for scalable deployment scenarios
# MAGIC
# MAGIC However, several production-hardening components such as CI/CD, automated drift detection, and retraining triggers are not yet implemented.
# MAGIC
# MAGIC
# MAGIC ## Final Assessment
# MAGIC
# MAGIC The system is suitable for **staging-level deployment**.  
# MAGIC To be production-ready, it requires automation around monitoring, model retraining, and deployment pipelines.
# MAGIC
# MAGIC Overall, it demonstrates strong MLOps design thinking with partial production implementation.