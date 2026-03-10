"""
data_processing.py
------------------
Handles all data ingestion, cleaning, and feature engineering steps.

Responsibilities:
    - Load raw transactional data from data/raw/
    - Clean and validate records (nulls, duplicates, negative quantities/prices)
    - Parse and enrich date features (recency, day-of-week, month, etc.)
    - Build the RFM (Recency, Frequency, Monetary) table at the customer level
    - Save cleaned and feature-engineered datasets to data/processed/

Typical usage:
    from src.data_processing import load_data, build_rfm_table
    df = load_data("data/raw/online_retail.xlsx")
    rfm = build_rfm_table(df)
"""

# TODO (Phase 1): Implement load_data()
# TODO (Phase 1): Implement clean_data()
# TODO (Phase 1): Implement build_rfm_table()
# TODO (Phase 1): Implement save_processed()

pass
