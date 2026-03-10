# Customer Segmentation & Retention Analysis — E-Commerce

> End-to-end data science project: customer segmentation, churn prediction, and lifetime value estimation — applied to a real-world e-commerce transactional dataset.

---

## Overview

E-commerce platforms depend on repeat purchases to stay profitable. This project applies unsupervised and supervised machine learning to 1M+ real transactions to answer three core business questions:

1. **Who are our customers?** — RFM-based segmentation using K-Means clustering to identify High-Value Loyalists, At-Risk Mid-Tier, and Low-Engagement One-Time Buyers.
2. **Who is about to leave?** — Churn prediction using XGBoost / LightGBM with feature engineering on purchase recency, frequency, and order value patterns.
3. **How much is each customer worth?** — Customer Lifetime Value (CLV) estimation using the BG/NBD + Gamma-Gamma model via the `lifetimes` library.

Results are surfaced in an interactive **Streamlit dashboard** and tracked with **MLflow** for experiment reproducibility.

---

## Business Problem

For e-commerce companies like **Amazon**, **Shopify**, or **Flipkart**, growth depends not just on acquiring new buyers but on understanding and retaining existing ones. Industry research shows that returning customers spend 67% more than first-time buyers, and acquiring a new customer costs 5–7× more than retaining an existing one.

Yet most retail analytics teams struggle to move beyond aggregate dashboards toward actionable, customer-level intelligence. The core questions go unanswered:

> *"How do we identify which customers are about to churn, which are high-value, and what actions should we take for each segment?"*

This project simulates the analytical workflow a Data Scientist at an e-commerce company would run to:
- Identify high-value customer segments for targeted loyalty and upsell campaigns
- Flag customers showing early churn signals for proactive retention outreach
- Prioritize marketing spend by quantifying the economic value of each customer group

---

## Dataset

**UCI Online Retail II** — a publicly available dataset from the UCI Machine Learning Repository.

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | ~1 million transactions |
| Time Period | December 2009 – December 2011 |
| Geography | UK-based online retailer, customers worldwide |
| Key Fields | `CustomerID`, `InvoiceDate`, `Quantity`, `UnitPrice`, `Country`, `Description` |

The dataset represents a real wholesale giftware retailer and closely mirrors the transactional structure of modern e-commerce platforms.

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
customer-retention-analysis/
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
git clone https://github.com/<your-username>/customer-retention-analysis.git
cd customer-retention-analysis
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Add raw data
Download the **UCI Online Retail II** dataset and place it as `data/raw/online_retail_II.xlsx`. This folder is excluded from version control.

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
