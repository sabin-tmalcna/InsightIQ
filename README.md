# InsightIQ Pro – AI-Powered Business Intelligence Platform

**A production-ready, internship-worthy BI tool** that turns CSV/Excel files into actionable insights using AI, statistical analysis, and interactive dashboards.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🚀 Features

- **🔐 User Authentication** – Register, login, hashed passwords, user dashboards.
- **📊 Data Analysis** – Auto column detection, statistical summary, correlation matrix, anomaly detection (IQR/Z‑score).
- **🤖 AI-Generated Insights** – Executive summary, KPI suggestions, business recommendations (NVIDIA Kimi model + fallback rules).
- **📈 Interactive Charts** – Histograms, boxplots, correlation heatmaps, time series (Plotly).
- **💬 Dataset Q&A Chat** – Ask natural language questions about your data (anomalies, trends, missing values).
- **📑 PDF Reports** – One‑click export of analysis results (ReportLab).
- **⚙️ Advanced Analytics** – Data profiling, K‑Means clustering (visualised), linear regression forecasting.
- **Analysis History** – Save, view, and delete past analyses.

## Tech Stack

- **Backend**: Flask, SQLAlchemy, Flask‑Login
- **Data Processing**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit‑learn (clustering, forecasting)
- **Visualisation**: Plotly
- **AI Integration**: NVIDIA API (Kimi model) + rule‑based fallback
- **Frontend**: HTML5, CSS3, JavaScript
- **PDF Generation**: ReportLab

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/InsightIQ-Pro.git
cd InsightIQ-Pro
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt