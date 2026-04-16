# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5: AI BI Dashboard + Genie Space (Operations Path)
# MAGIC
# MAGIC This module transforms ML outputs into a self-service analytics layer using Databricks Genie and AI BI Dashboard.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 Unity Catalog Table Preparation

# COMMAND ----------

spark.sql("SHOW TABLES").show()

# COMMAND ----------

# MAGIC %md
# MAGIC All required tables are registered in Unity Catalog:
# MAGIC
# MAGIC - raw_operations → raw incident data
# MAGIC - features_operations → engineered ML features
# MAGIC - anomaly_scores → model predictions and anomaly outputs
# MAGIC
# MAGIC All tables include schema-aligned fields used for downstream Genie and dashboard analytics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 Genie Space Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Genie Custom Instructions
# MAGIC
# MAGIC The dataset represents an IT operations monitoring system that tracks service incidents, SLA breaches, and anomaly detection outputs. Each row in anomaly_scores represents a scored incident with a predicted anomaly label and severity context.
# MAGIC
# MAGIC The key relationship is:
# MAGIC - raw_operations → source incident logs
# MAGIC - features_operations → engineered time-series and statistical features
# MAGIC - anomaly_scores → ML model outputs used for operational alerting
# MAGIC
# MAGIC Business users should interpret "is_anomaly" as a system-generated flag indicating unusual or high-risk incidents. SLA breach rate reflects operational performance and service reliability.
# MAGIC
# MAGIC Time-based analysis should always consider timestamp granularity (hourly/daily aggregation). High anomaly_score values indicate higher severity risk.
# MAGIC
# MAGIC This Genie space is designed for operations teams to monitor system health, identify failing services, and investigate incident patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Genie Test Queries (Natural Language → SQL)
# MAGIC
# MAGIC ### Q1  Which service has most anomalies?
# MAGIC SELECT service_name, COUNT(*) AS anomaly_count
# MAGIC FROM anomaly_scores
# MAGIC WHERE is_anomaly = true
# MAGIC GROUP BY service_name
# MAGIC ORDER BY anomaly_count DESC;
# MAGIC - Insight: Payments service shows the highest number of anomalies, indicating potential instability or higher operational risk compared to other services.
# MAGIC
# MAGIC ### Q2  Show anomaly trend over time
# MAGIC SELECT DATE_TRUNC('day', timestamp) AS date, COUNT(*) AS anomaly_count
# MAGIC FROM anomaly_scores
# MAGIC WHERE is_anomaly = true
# MAGIC GROUP BY DATE_TRUNC('day', timestamp)
# MAGIC ORDER BY date;
# MAGIC - Insight: Anomaly trends help identify periods of high system instability, which may correlate with deployments, peak traffic, or system failures.
# MAGIC
# MAGIC ### Q3 Average resolution time by team
# MAGIC SELECT team, AVG(resolution_time_mins) AS avg_resolution_time
# MAGIC FROM anomaly_scores
# MAGIC GROUP BY team
# MAGIC ORDER BY avg_resolution_time DESC;
# MAGIC - Insight: Resolution time across teams is relatively consistent, suggesting balanced workload distribution and similar operational efficiency.
# MAGIC
# MAGIC ### Q4 Top anomaly incidents
# MAGIC SELECT incident_id, service_name, timestamp, severity, resolution_time_mins, anomaly_score, team, root_cause, sla_breached
# MAGIC FROM anomaly_scores
# MAGIC WHERE is_anomaly = true
# MAGIC ORDER BY anomaly_score DESC
# MAGIC LIMIT 20;
# MAGIC - Insight: Top anomaly incidents highlight the most critical failures, allowing teams to prioritize high-impact issues for faster resolution.
# MAGIC ### Q5 SLA breach rate
# MAGIC
# MAGIC SELECT 
# MAGIC   COUNT(*) AS total_incidents,
# MAGIC   SUM(CASE WHEN sla_breached = true THEN 1 ELSE 0 END) AS breached_incidents,
# MAGIC   ROUND(SUM(CASE WHEN sla_breached = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS breach_rate_percent
# MAGIC FROM anomaly_scores;
# MAGIC - Insight: SLA breach rate reflects overall system reliability. A higher percentage indicates performance issues that may impact customer experience.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 AI BI Dashboard Design
# MAGIC The dashboard is structured into two analytical layers:
# MAGIC
# MAGIC ### Tab 1: Operational Overview
# MAGIC - Incident volume trends
# MAGIC - SLA breach distribution
# MAGIC - Severity breakdown
# MAGIC - Team-wise resolution performance
# MAGIC
# MAGIC ### Tab 2: ML Insights
# MAGIC - Anomaly score distribution
# MAGIC - Temporal anomaly patterns
# MAGIC - High-risk incident ranking
# MAGIC
# MAGIC The dashboard includes interactive filters for time range and service-level analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Genie Space & Dashboard Access

# COMMAND ----------

# MAGIC %md
# MAGIC #### Genie Space URL:
# MAGIC - https://dbc-d5b89325-2db6.cloud.databricks.com/genie/rooms/01f137e9a5ab1269989e82f19b90de8e?o=7474646604545547
# MAGIC
# MAGIC #### Dashboard URL:
# MAGIC - https://dbc-d5b89325-2db6.cloud.databricks.com/dashboardsv3/01f137ef709d1ab6a4474575148adfed/published?o=7474646604545547&f_d73888ca%7Eaverage-resolution-time-by-team=%7B%22columns%22%3A%5B%22x%22%5D%2C%22rows%22%3A%5B%5B%22team_c%22%5D%5D%7D&f_167964a1%7Eanomaly-score-distribution=%7B%22columns%22%3A%5B%22x%22%5D%2C%22rows%22%3A%5B%5B%220%22%5D%5D%7D

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC The Genie Space and AI BI Dashboard provide a self-service analytics layer over operational data and ML outputs, enabling non-technical users to explore SLA breaches, anomaly trends, and service performance through natural language queries and interactive dashboards.
# MAGIC
# MAGIC Key value delivered:
# MAGIC - Real-time anomaly tracking via Genie SQL queries
# MAGIC - Visual insights into SLA breaches and service performance
# MAGIC - Business-friendly dashboards for non-technical stakeholders
# MAGIC
# MAGIC Together, these tools operationalize the `anomaly_scores` dataset into actionable insights, supporting faster incident investigation and improved operational decision-making.
# MAGIC
# MAGIC This completes the transition from raw data → ML outputs → business intelligence layer.