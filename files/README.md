# Operational Data Intelligence (base project)

Turns raw operational data (orders, processing time, cost, revenue by department/day)
into KPIs, anomaly alerts and plain-language insights.

## Pipeline
Ingest -> Clean -> KPIs -> Anomaly detection (rolling z-score) -> Insights -> Report/Charts

## Run
    pip install -r requirements.txt
    python odi.py                 # synthetic data is auto-generated in data/operations.csv
    python odi.py --input my.csv  # your data
    streamlit run app.py          # optional dashboard

## Input columns
date, department, orders, completed, avg_processing_min, cost, revenue

## Outputs (./output)
dashboard.png, report.md, department_summary.csv, anomalies.csv

## Ideas to extend
- Swap CSV for a SQL/warehouse source
- Add Isolation Forest / Prophet forecasting
- Schedule with cron/Airflow and send alerts (email/Slack)
