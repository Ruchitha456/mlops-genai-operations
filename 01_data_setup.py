# Databricks notebook source
# MAGIC %md
# MAGIC # Path Selection
# MAGIC ## Chosen Path: Operations (PATH B)
# MAGIC Reason:
# MAGIC I selected the Operations path to design an end-to-end anomaly detection and SLA breach prediction system, which better demonstrates real-world production ML patterns and enables deeper integration with RAG and AI agents for incident analysis and resolution.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Module 1: Data Setup & Feature Engineering
# MAGIC
# MAGIC This module focuses on generating a realistic incident dataset and preparing it for downstream machine learning tasks.
# MAGIC
# MAGIC ### Objectives
# MAGIC
# MAGIC * Simulate operational incident logs with realistic patterns
# MAGIC * Store raw data in Delta Lake
# MAGIC * Prepare data for feature engineering in later steps
# MAGIC
# MAGIC The dataset includes:
# MAGIC
# MAGIC * Incident logs across services and regions
# MAGIC * SLA thresholds and breach indicators
# MAGIC * Injected anomalies (~5%) for anomaly detection use cases
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1: Data Generation

# COMMAND ----------

import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

data = pd.DataFrame({
    "incident_id": [f"INC_{i}" for i in range(n)],
    "timestamp": pd.date_range(start="2023-01-01", periods=n, freq="h"),
    "service_name": np.random.choice(["payments", "auth", "search"], n),
    "severity": np.random.choice([1,2,3,4], n, p=[0.1, 0.2, 0.4, 0.3]),
    "resolution_time_mins": np.random.normal(60, 20, n),
    "sla_threshold_mins": np.random.choice([30, 60, 120], n),
    "region": np.random.choice(["APAC", "EMEA", "NA"], n),
    "team": np.random.choice(["team_a", "team_b", "team_c"], n),
    "root_cause": np.random.choice(
        ["dependency_failure", "timeout", "code_bug", "infra_issue"], n
    )
})

# COMMAND ----------

# Business hour effect
data["hour"] = data["timestamp"].dt.hour
data.loc[data["hour"].between(9, 18), "resolution_time_mins"] *= 1.2  

# SLA Breach + Patterns
# SLA breach flag
data["sla_breached"] = data["resolution_time_mins"] > data["sla_threshold_mins"]

# COMMAND ----------

# Inject Real Anomalies (~5%)
anomaly_indices = np.random.choice(n, size=int(0.05*n), replace=False)

data.loc[anomaly_indices, "resolution_time_mins"] *= 3
data.loc[anomaly_indices, "sla_breached"] = True

data["is_anomaly"] = 0
data.loc[anomaly_indices, "is_anomaly"] = 1

# COMMAND ----------

# Repeat Incidents
data["repeat_incident"] = np.random.choice([0,1], size=n, p=[0.8, 0.2])

# COMMAND ----------

# Validate
print("Total:", len(data))
print("Anomalies:", data["is_anomaly"].sum())

display(data.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation Strategy
# MAGIC
# MAGIC The dataset simulates incident logs across multiple services with realistic operational patterns.
# MAGIC
# MAGIC ### Key Characteristics
# MAGIC
# MAGIC * Business-hour impact increases resolution times during peak hours
# MAGIC * Approximately 5% of records are injected as anomalies with extreme resolution times
# MAGIC * SLA breaches are calculated based on resolution time vs threshold
# MAGIC * Repeat incidents are included to mimic recurring service failures
# MAGIC
# MAGIC ### Data Quality Checks
# MAGIC
# MAGIC * Verified total record count and anomaly distribution
# MAGIC * Ensured SLA breach logic is correctly applied
# MAGIC * Checked distributions across services, severity, and regions
# MAGIC
# MAGIC ### Observations
# MAGIC
# MAGIC * Anomalies create clear outliers in resolution time
# MAGIC * Business-hour incidents tend to have slightly higher resolution times
# MAGIC * Dataset contains sufficient variation for anomaly detection tasks
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Save Raw Data

# COMMAND ----------

# Convert to Spark DataFrame

spark_df=spark.createDataFrame(data)

# Save as Delta Table
spark_df.write.format("delta").mode("overwrite").saveAsTable("raw_operations")

# Verify Table
display(spark.sql("SELECT * FROM raw_operations LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Storage & Lineage
# MAGIC
# MAGIC The dataset is stored as a Delta table named `raw_operations`.
# MAGIC
# MAGIC ### Design Considerations
# MAGIC
# MAGIC * Delta Lake ensures ACID transactions and scalability
# MAGIC * Table serves as the base layer for feature engineering
# MAGIC
# MAGIC ### Data Flow
# MAGIC
# MAGIC raw_operations → features_operations (next step)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: Feature Engineering
# MAGIC
# MAGIC This section transforms raw incident logs into a machine learning–ready feature set.
# MAGIC
# MAGIC We construct:
# MAGIC
# MAGIC * Temporal features
# MAGIC * Lag-based features
# MAGIC * Rolling statistics
# MAGIC * Domain-level aggregates
# MAGIC * Interaction features
# MAGIC
# MAGIC These features are designed to capture both short-term and long-term operational patterns in incident behavior.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Load Data

# COMMAND ----------

# Load Table as Spark DF
df=spark.read.table("raw_operations")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Temporal Features

# COMMAND ----------

from pyspark.sql.functions import dayofweek, weekofyear, month, last_day, col

df = df.withColumn("day_of_week", dayofweek("timestamp")) \
       .withColumn("week_of_year", weekofyear("timestamp")) \
       .withColumn("month", month("timestamp")) \
       .withColumn("is_month_end", col("timestamp") == last_day("timestamp"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Temporal Feature Engineering
# MAGIC
# MAGIC These features capture calendar-based patterns such as:
# MAGIC
# MAGIC * Weekly seasonality (day_of_week)
# MAGIC * Monthly cycles (month, week_of_year)
# MAGIC * End-of-month system load effects (is_month_end)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Window Setup

# COMMAND ----------

from pyspark.sql.window import Window

window_spec = Window.partitionBy("service_name").orderBy("timestamp")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Lag Features

# COMMAND ----------

from pyspark.sql.functions import lag

df = df.withColumn("lag_1d", lag("resolution_time_mins", 24).over(window_spec)) \
       .withColumn("lag_7d", lag("resolution_time_mins", 24*7).over(window_spec)) \
       .withColumn("lag_30d", lag("resolution_time_mins", 24*30).over(window_spec))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lag Features
# MAGIC
# MAGIC Lag features capture historical system behavior:
# MAGIC
# MAGIC * 1-day lag: short-term response behavior
# MAGIC * 7-day lag: weekly operational trends
# MAGIC * 30-day lag: long-term service performance
# MAGIC
# MAGIC These features help detect anomalies based on deviation from historical patterns.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Rolling Statistics

# COMMAND ----------

from pyspark.sql.functions import avg, stddev

rolling_7d = window_spec.rowsBetween(-24*7, 0)
rolling_14d = window_spec.rowsBetween(-24*14, 0)
rolling_30d = window_spec.rowsBetween(-24*30, 0)

df = df.withColumn("rolling_mean_7d", avg("resolution_time_mins").over(rolling_7d)) \
       .withColumn("rolling_std_7d", stddev("resolution_time_mins").over(rolling_7d)) \
       .withColumn("rolling_mean_14d", avg("resolution_time_mins").over(rolling_14d)) \
       .withColumn("rolling_std_14d", stddev("resolution_time_mins").over(rolling_14d)) \
       .withColumn("rolling_mean_30d", avg("resolution_time_mins").over(rolling_30d)) \
       .withColumn("rolling_std_30d", stddev("resolution_time_mins").over(rolling_30d))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rolling Statistics
# MAGIC
# MAGIC Rolling features capture system stability over time:
# MAGIC
# MAGIC * Short-term volatility (7-day window)
# MAGIC * Medium-term trends (14-day window)
# MAGIC * Long-term operational drift (30-day window)
# MAGIC
# MAGIC Standard deviation features help detect instability and irregular behavior spikes.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Domain Aggregates

# COMMAND ----------

from pyspark.sql.functions import sum as spark_sum

agg_df = df.groupBy("service_name").agg(
    spark_sum("resolution_time_mins").alias("total_resolution_time")
)

df = df.join(agg_df, on="service_name", how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Domain-Level Aggregates
# MAGIC
# MAGIC Service-level aggregates capture macro behavior patterns such as:
# MAGIC
# MAGIC * Total system load per service
# MAGIC * Persistent high-cost services in terms of resolution time
# MAGIC
# MAGIC These features help identify services that consistently contribute to operational strain.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Interaction Feature

# COMMAND ----------

df = df.withColumn(
    "severity_x_resolution",
    col("severity") * col("resolution_time_mins")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interaction Feature
# MAGIC
# MAGIC The interaction between severity and resolution time captures compound system stress:
# MAGIC
# MAGIC * High severity + high resolution time → critical operational risk
# MAGIC * Helps model nonlinear relationships in anomaly detection
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Handle Missing Values

# COMMAND ----------

df = df.fillna({
    "lag_1d": 0,
    "lag_7d": 0,
    "lag_30d": 0,
    "rolling_mean_7d": 0,
    "rolling_std_7d": 0
})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Value Strategy
# MAGIC
# MAGIC Lag and rolling features introduce missing values at the start of time series windows.
# MAGIC
# MAGIC We impute these with 0 under the assumption that:
# MAGIC
# MAGIC * No prior system history exists at those points
# MAGIC * Early-stage system behavior should not bias anomaly detection
# MAGIC
# MAGIC This ensures model stability without introducing artificial patterns.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9. Save Feature Table

# COMMAND ----------

df.write.format("delta") \
  .mode("overwrite") \
  .partitionBy("service_name") \
  .saveAsTable("features_operations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Summary
# MAGIC
# MAGIC The final feature set combines:
# MAGIC
# MAGIC * Temporal patterns
# MAGIC * Historical lag behavior
# MAGIC * Rolling system stability metrics
# MAGIC * Service-level aggregates
# MAGIC * Interaction-based stress indicators
# MAGIC
# MAGIC This feature store is optimized for anomaly detection and SLA breach prediction tasks, enabling both short-term and long-term system behavior analysis.
# MAGIC