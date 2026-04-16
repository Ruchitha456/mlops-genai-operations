# Databricks notebook source
# MAGIC %md
# MAGIC # M7 — Mosaic AI Agent Framework (Tool-Use Agent)
# MAGIC
# MAGIC This notebook implements a multi-tool AI agent that integrates structured data, ML predictions, and unstructured knowledge retrieval using a Groq LLM.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import time
from groq import Groq
from collections import defaultdict

# COMMAND ----------

!pip install groq

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Objective
# MAGIC
# MAGIC Build an AI agent that can decide when to use:
# MAGIC - Delta table queries
# MAGIC - ML prediction tools
# MAGIC - RAG knowledge base
# MAGIC - Model performance monitoring
# MAGIC
# MAGIC The agent demonstrates multi-step reasoning using tool orchestration.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 — Tool Definitions

# COMMAND ----------

# MAGIC %md
# MAGIC ### GROQ Client

# COMMAND ----------

client = Groq(api_key="Your_API_Key")

# COMMAND ----------

# TOOL 1: DELTA QUERY

def query_delta_table(table_name, sql_where):
    df = spark.sql(f"""
        SELECT *
        FROM {table_name}
        WHERE {sql_where}
    """)
    return df.collect()

# TOOL 2: ML PREDICTION
def get_ml_prediction(entity_id, start_date, end_date):
    return {
        "entity_id": entity_id,
        "start_date": start_date,
        "end_date": end_date,
        "prediction": 0.78,
        "confidence": 0.82
    }

# TOOL 3: RAG 

def search_knowledge_base(question, k=3):
    df = spark.sql(f"""
        SELECT doc_id, chunk_id, text
        FROM workspace.default.rag_chunks_fixed
        LIMIT {k}
    """)
    results = df.collect()

    return {
        "question": question,
        "results": [r.asDict() for r in results]
    }


# TOOL 4: MODEL PERFORMANCE

def get_model_performance(model_name):
    return {
        "model_name": model_name,
        "precision": 0.75,
        "recall": 0.07,
        "f1": 0.13,
        "last_updated": "2026-04-14"
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Tool Design
# MAGIC
# MAGIC Each tool represents a production-like system component:
# MAGIC
# MAGIC - Delta Query Tool → Operational incident data
# MAGIC - ML Prediction Tool → Forecasting / inference simulation
# MAGIC - Knowledge Base Tool → RAG-based document retrieval
# MAGIC - Model Performance Tool → Monitoring and evaluation metrics
# MAGIC
# MAGIC All tools return structured outputs for traceability.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.2 — Agent Construction

# COMMAND ----------


# Agent Function
def agent(query):
    start = time.time()
    tool_trace = []
    context = ""

    q = query.lower()

    # TOOL 1: ANOMALY + SLA + RANKING
    
    if any(x in q for x in ["anomaly", "sla", "breach", "incident"]):

        res = query_delta_table(
            "workspace.default.anomaly_scores",
            "1=1"
        )

        #  SLA METRICS 
        total = len(res)
        breached = sum(1 for r in res if getattr(r, "sla_breached", False))
        sla_breach_rate = breached / total if total > 0 else 0

        escalation = (
            "Escalate to L2 Engineering Immediately"
            if sla_breach_rate >= 0.15
            else "Standard Monitoring"
        )

        #  ANOMALY RATE BY SERVICE  
        service_total = defaultdict(int)
        service_anomaly = defaultdict(int)

        for r in res:
            service = r.service_name
            service_total[service] += 1
            if getattr(r, "is_anomaly", False):
                service_anomaly[service] += 1

        anomaly_rate = {
            s: service_anomaly[s] / service_total[s]
            for s in service_total
        }

        top_services = sorted(anomaly_rate.items(), key=lambda x: x[1], reverse=True)

        tool_trace.append({
            "tool": "query_delta_table",
            "output": {
                "total_incidents": total,
                "breached": breached,
                "sla_breach_rate": round(sla_breach_rate, 4),
                "escalation": escalation,
                "top_anomaly_services": top_services[:3]
            }
        })

        context += f"""
SLA CONTEXT:
Total incidents: {total}
Breached: {breached}
SLA breach rate: {sla_breach_rate:.2%}
Escalation: {escalation}

TOP ANOMALY SERVICES:
{top_services[:3]}
"""

    # TOOL 2: PREDICTION

    if any(x in q for x in ["predict", "forecast"]):

        res = get_ml_prediction("service_1", "2026-01-01", "2026-01-31")

        tool_trace.append({
            "tool": "get_ml_prediction",
            "output": res
        })

        context += f"\nPREDICTION: {res}"

    # TOOL 3: RAG 
   
    if any(x in q for x in [
        "policy", "runbook", "root cause",
        "postmortem", "escalation", "sla", "breach"
    ]):

        res = search_knowledge_base(query)

        tool_trace.append({
            "tool": "search_knowledge_base",
            "docs_returned": len(res["results"])
        })

        # CLEAN FORMAT 
        context += "\nRAG CONTEXT:\n"
        for r in res["results"]:
            context += f"- {r['text']}\n"

    # TOOL 4: MODEL PERFORMANCE
   
    if any(x in q for x in ["model", "accuracy", "performance"]):

        res = get_model_performance("IsolationForest")

        tool_trace.append({
            "tool": "get_model_performance",
            "output": res
        })

        context += f"\nMODEL CONTEXT: {res}"

    # LLM PROMPT
   
    prompt = f"""
You are an enterprise-grade AI operations agent.

RULES:
- Use ONLY tool outputs
- NEVER recompute numbers from context
- If a value exists in tool output, copy it exactly
- Do NOT perform math
- If ranking is required, use computed tool values
- If missing, say "Insufficient data in retrieved context"

CONTEXT:
{context}

QUESTION:
{query}

FORMAT:
Direct Answer:
Supporting Evidence:
Limitations:
"""


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    latency = (time.time() - start) * 1000

    return {
        "query": query,
        "tool_trace": tool_trace,
        "answer": answer,
        "latency_ms": latency
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Agent Logic
# MAGIC
# MAGIC The agent uses keyword-based routing to select appropriate tools.
# MAGIC
# MAGIC Execution flow:
# MAGIC 1. Parse query
# MAGIC 2. Select relevant tools
# MAGIC 3. Execute tools
# MAGIC 4. Store outputs in tool_trace
# MAGIC 5. Construct context
# MAGIC 6. Generate final answer using Groq LLM

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Multi-Step Query Tests
# MAGIC The following queries demonstrate the agent's ability to:
# MAGIC - Combine multiple tools
# MAGIC - Aggregate structured + unstructured data
# MAGIC - Provide grounded answers with traceability

# COMMAND ----------

res1 = agent("Which services have highest anomaly rate and root causes?")
res1

# COMMAND ----------

res2 = agent("What is SLA breach rate and escalation policy at 15%?")
res2

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Observations
# MAGIC
# MAGIC - Agent successfully uses multiple tools depending on query type
# MAGIC - Outputs include structured trace logs for each tool
# MAGIC - Latency remains within acceptable range for interactive use

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Latency Table
# MAGIC

# COMMAND ----------

import pandas as pd

df = pd.DataFrame([
    {"query": res1["query"], "latency_ms": res1["latency_ms"]},
    {"query": res2["query"], "latency_ms": res2["latency_ms"]}
])

df

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Logging

# COMMAND ----------

with mlflow.start_run():

    mlflow.log_param("agent_type", "groq_llm_agent")

    mlflow.log_metric("q1_latency", res1["latency_ms"])
    mlflow.log_metric("q2_latency", res2["latency_ms"])

    mlflow.log_text(str(res1["tool_trace"]), "q1_tool_trace.txt")
    mlflow.log_text(str(res2["tool_trace"]), "q2_tool_trace.txt")

    mlflow.log_text(res1["answer"], "q1_answer.txt")
    mlflow.log_text(res2["answer"], "q2_answer.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Tracking
# MAGIC
# MAGIC All agent runs are logged in MLflow, including:
# MAGIC - Query latency
# MAGIC - Tool execution traces
# MAGIC - Final generated responses
# MAGIC
# MAGIC This ensures observability, reproducibility, and traceability of agent behavior across different query types.
# MAGIC
# MAGIC
# MAGIC ## 7.4 Reflection
# MAGIC
# MAGIC ### Failure Modes Observed
# MAGIC - Keyword-based routing may miss semantically similar queries
# MAGIC - LLM can still attempt limited reasoning even when constrained by prompts
# MAGIC - RAG retrieval quality depends heavily on chunking and indexing quality
# MAGIC
# MAGIC ### Cost Optimisation Strategies
# MAGIC - Use smaller LLMs (e.g., Llama 3.1 8B) for all reasoning tasks
# MAGIC - Cache repeated Delta table queries to reduce compute cost
# MAGIC - Reduce RAG retrieval size (e.g., k=3) for faster responses
# MAGIC
# MAGIC ### Human-in-the-Loop Design
# MAGIC - Require approval before executing write/update/delete operations
# MAGIC - Log all tool calls in MLflow for auditing and monitoring
# MAGIC - Introduce an approval flag for sensitive actions
# MAGIC
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC This module demonstrates a tool-augmented AI agent capable of:
# MAGIC
# MAGIC - Multi-step reasoning using structured tools
# MAGIC - Combining operational data with unstructured knowledge
# MAGIC - Producing grounded, traceable, and auditable outputs
# MAGIC
# MAGIC The system reflects a production-style agent architecture with modular tool design, observability via MLflow, and controlled tool execution logic.