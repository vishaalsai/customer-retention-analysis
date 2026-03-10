# 🛒 Customer Segmentation & Retention Analysis

**A full-stack data science project combining unsupervised learning, churn prediction, and customer lifetime value estimation — presented as a live interactive dashboard.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://customer-retention-analysis-n9qkjosvrfllakzbsamnqf.streamlit.app)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600?logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🔗 Live Demo

> 🚀 **[Launch Live Dashboard](https://customer-retention-analysis-n9qkjosvrfllakzbsamnqf.streamlit.app)**

Interactive 5-page Streamlit app — no setup required, runs in your browser.

---

## 📌 Business Problem

E-commerce companies like Amazon, Shopify, and Flipkart lose significant revenue every year to customer churn — yet most analytics stops at reporting who has already left. This project goes further: it predicts which customers are likely to churn, estimates the economic value at stake for each one, and recommends specific, budget-prioritised retention actions per segment.

By combining RFM segmentation, XGBoost classification, and customer lifetime value modelling on 779,425 real retail transactions, this project demonstrates the kind of end-to-end analytical thinking that drives measurable commercial outcomes — not just dashboards.

---

## 🔍 Key Findings

- **5,878 customers** analysed across **779,425 transactions** (UCI Online Retail II, 2009–2011)
- **2 behavioural segments** identified: **Champions** (39.3%) and **Dormant / At-Risk** (60.7%)
- Champions represent **39.3% of customers** but **87.4% of projected 3-year revenue**
- **£3,359,853** in revenue identified as at-risk due to churn
- XGBoost churn model: **98.3% accuracy · 98.5% recall · ROC-AUC 0.999**
- **Frequency is the #1 churn driver** (53.3% feature importance) — customers who stop purchasing frequently are the earliest-warning churn signal
- **3,860 customers** identified where a £10 retention intervention yields **£3,089,821 expected return (80× ROI)**
- **Platinum CLV tier:** 588 customers (10%) holding **£16.5M** in projected 3-year value

---

## 🗂️ Project Structure

```
customer-retention-analysis/
├── data/
│   ├── raw/                  # Original dataset (not tracked)
│   └── processed/            # Cleaned data and model outputs
├── notebooks/
│   ├── exploration.ipynb     # Phase 1: EDA
│   ├── segmentation.ipynb    # Phase 2: RFM + clustering
│   ├── churn_model.ipynb     # Phase 3: Churn prediction
│   └── clv.ipynb             # Phase 4: CLV estimation
├── src/
│   ├── data_processing.py    # Data loading and cleaning
│   ├── segmentation.py       # RFM + K-Means clustering
│   ├── churn_model.py        # XGBoost churn model
│   └── clv.py                # CLV estimation
├── app/
│   └── streamlit_app.py      # 5-page Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

1. **Data Ingestion:** Loaded UCI Online Retail II dataset (1,067,371 raw transactions, 2 sheets combined)
2. **Data Cleaning:** Removed nulls, duplicates, cancellations, and invalid prices — 287,946 rows removed (27%), 779,425 retained
3. **Feature Engineering:** Computed RFM (Recency, Frequency, Monetary Value) per customer using 2011-12-10 as reference date
4. **Segmentation:** Applied log transformation + StandardScaler, K-Means clustering with k=2 (silhouette score = 0.439)
5. **Churn Prediction:** XGBoost classifier, 80/20 stratified split, MLflow experiment tracking, evaluated on precision + recall
6. **CLV Estimation:** 3-year projection (Monetary × 1.5), risk-adjusted by churn probability, retention ROI calculated at £10/customer intervention cost
7. **Dashboard:** 5-page Streamlit app with interactive Plotly charts, customer lookup tool, and business action framework

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Data | Python, Pandas, NumPy |
| ML / Modelling | Scikit-learn, XGBoost, LightGBM |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
| Experiment Tracking | MLflow |
| Dataset | UCI Online Retail II (via Kaggle) |
| Version Control | Git, GitHub |

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Executive Summary | KPIs, segment overview, model metrics |
| 👥 Customer Segments | RFM distributions, 3D scatter, quadrant matrix |
| 🔮 Churn Prediction | Model performance, risk tiers, customer lookup |
| 💰 Customer Lifetime Value | CLV tiers, revenue at risk, retention ROI |
| 🎯 Retention Strategy | Budget allocation, action framework, methodology |

---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/vishaalsai/customer-retention-analysis.git
cd customer-retention-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place online_retail_II.xlsx in data/raw/

# Run the pipeline
python src/data_processing.py
python src/segmentation.py
python src/churn_model.py
python src/clv.py

# Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## ⚠️ Limitations & Future Work

### Known Limitations

- **Churn label** is derived from RFM-based segmentation, not actual cancellation events — in production this would use verified churn signals (account closure, 90-day inactivity) to eliminate label leakage between features and target
- **CLV model** uses a simplified linear projection (Monetary × 1.5) — a BG/NBD probabilistic model would be more accurate for production use
- **Dataset scope:** 2009–2011 UK retail transactions — behavioural patterns may differ in modern, mobile-first e-commerce contexts

### Future Improvements

- Replace churn label with an event-based definition (subscription cancellation logs, inactivity thresholds)
- Implement BG/NBD + Gamma-Gamma model for CLV using the `lifetimes` library
- Add an A/B test framework to measure actual retention intervention ROI post-deployment
- Build a real-time customer scoring API using FastAPI or Flask
- Add cohort analysis to track segment migration (Champion → At-Risk) over time

---

## 👤 Author

Built by **Vishaalsai**

- GitHub: [github.com/vishaalsai](https://github.com/vishaalsai)
- Project: [Customer Segmentation & Retention Analysis](https://github.com/vishaalsai/customer-retention-analysis)

---

## 📄 License

MIT License
