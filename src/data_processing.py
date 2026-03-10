"""
data_processing.py
------------------
Module for loading, cleaning, and feature engineering of the UCI Online
Retail II dataset. Handles missing values, cancellations, outliers, and
computes RFM (Recency, Frequency, Monetary Value) features per customer
for e-commerce retention analysis.

Responsibilities:
    - Load raw transactional data from data/raw/
    - Remove cancelled orders, null CustomerIDs, and invalid quantities/prices
    - Parse and enrich date features (recency, day-of-week, month, etc.)
    - Build the RFM (Recency, Frequency, Monetary) table at the customer level
    - Save cleaned and feature-engineered datasets to data/processed/

Typical usage:
    from src.data_processing import load_data, build_rfm_table
    df = load_data("data/raw/online_retail_II.xlsx")
    rfm = build_rfm_table(df)
"""

# TODO (Phase 1): Implement load_data()
# TODO (Phase 1): Implement clean_data()
# TODO (Phase 1): Implement build_rfm_table()
# TODO (Phase 1): Implement save_processed()

pass
