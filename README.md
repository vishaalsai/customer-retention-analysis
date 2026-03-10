# Customer Segmentation & Retention Analysis

> End-to-end data science project: customer segmentation, churn prediction, and lifetime value estimation — framed for a subscription-based streaming business (Spotify / Netflix context).

---

## Overview

Streaming platforms live and die by subscriber retention. This project applies unsupervised and supervised machine learning to a real transactional dataset to answer three core business questions:

1. **Who are our customers?** — RFM-based segmentation using K-Means clustering to identify High-Value, At-Risk, Dormant, and New cohorts.
2. **Who is about to leave?** — Churn prediction using XGBoost / LightGBM with feature engineering on listening/engagement patterns.
3. **How much is each customer worth?** — Customer Lifetime Value (CLV) estimation using the BG/NBD + Gamma-Gamma model via the `lifetimes` library.

Results are surfaced in an interactive **Streamlit dashboard** and tracked with **MLflow** for experiment reproducibility.

---

## Business Problem

A streaming company's growth depends not just on new user acquisition, but on **retaining and monetizing existing subscribers**. Industry benchmarks show that increasing customer retention by just 5% can boost profits by 25–95%. Yet most platforms struggle to move beyond vanity metrics (DAU/MAU) toward actionable, user-level intelligence.

This project simulates the analytical workflow a Data Scientist at Spotify or Netflix would run to:
- Identify high-value subscriber segments for targeted marketing
- Flag users showing early churn signals for proactive intervention
- Prioritize retention spend by estimated lifetime value

---

## Tech Stack

| Category | Tools |
|---|---|
| Data Wrangling | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| CLV Modeling | `lifetimes` (BG/NBD + Gamma-Gamma) |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Experiment Tracking | `mlflow` |
| Environment | Python 3.10+, Jupyter |

---

## Project Structure

```
spotify-retention-analysis/
├── data/
│   ├── raw/            # Source data (not tracked by git)
│   └── processed/      # Cleaned & feature-engineered outputs
├── notebooks/
│   └── exploration.ipynb   # EDA and prototyping
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Cleaning, feature engineering, RFM table
│   ├── segmentation.py     # K-Means clustering + segment labeling
│   ├── churn_model.py      # XGBoost/LightGBM churn classifier
│   └── clv.py              # BG/NBD + Gamma-Gamma CLV estimation
├── app/
│   └── streamlit_app.py    # Interactive dashboard
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/spotify-retention-analysis.git
cd spotify-retention-analysis
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Add raw data
Place the source dataset (e.g., `online_retail.xlsx`) inside `data/raw/`. This folder is excluded from version control.

### 4. Run notebooks
```bash
jupyter notebook notebooks/exploration.ipynb
```

### 5. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Project Phases Roadmap

- [x] **Phase 0** — Project setup, folder structure, README
- [ ] **Phase 1** — Data ingestion, cleaning, and exploratory data analysis (EDA)
- [ ] **Phase 2** — RFM feature engineering and customer segmentation (K-Means)
- [ ] **Phase 3** — Churn prediction model (XGBoost / LightGBM + MLflow tracking)
- [ ] **Phase 4** — Customer Lifetime Value estimation (BG/NBD + Gamma-Gamma)
- [ ] **Phase 5** — Streamlit dashboard (interactive segment explorer + CLV viewer)
- [ ] **Phase 6** — Model evaluation, explainability (SHAP), and business write-up
- [ ] **Phase 7** — Final polish: docstrings, tests, GitHub Actions CI

---

## Author

Built as a portfolio project demonstrating end-to-end data science — from raw transactional data to business-ready insights.
