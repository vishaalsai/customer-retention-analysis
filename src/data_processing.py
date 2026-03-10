"""
data_processing.py
------------------
Module for loading, cleaning, and feature engineering of the UCI Online
Retail II dataset. Handles missing values, cancellations, outliers, and
computes RFM (Recency, Frequency, Monetary Value) features per customer
for e-commerce retention analysis.

Dataset columns (after loading):
    Invoice      - Invoice number; starts with 'C' for cancellations
    StockCode    - Product code
    Description  - Product name
    Quantity     - Units per transaction (negative for returns)
    InvoiceDate  - Date and time of invoice
    Price        - Unit price in GBP
    CustomerID   - Unique customer identifier (renamed from 'Customer ID')
    Country      - Customer's country of residence

Usage:
    python src/data_processing.py
    --- or ---
    from src.data_processing import load_data, clean_data
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the UCI Online Retail II dataset from an Excel or CSV file.

    Supports two formats:
        - .xlsx  — The original UCI workbook with two sheets:
                   'Year 2009-2010' and 'Year 2010-2011', which are
                   concatenated into one DataFrame.
        - .csv   — A pre-exported single-file version of the dataset,
                   loaded directly (both years already combined).

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the .xlsx or .csv file.

    Returns
    -------
    pd.DataFrame
        Combined raw dataset.
    """
    print(f"[load_data] Reading: {filepath}")

    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".xlsx":
        sheet_2009 = pd.read_excel(filepath, sheet_name="Year 2009-2010", engine="openpyxl")
        sheet_2010 = pd.read_excel(filepath, sheet_name="Year 2010-2011", engine="openpyxl")
        df = pd.concat([sheet_2009, sheet_2010], ignore_index=True)
    elif ext == ".csv":
        # CSV export from UCI — all rows already in one file
        df = pd.read_csv(filepath, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Expected .xlsx or .csv")

    # Standardise column name: 'Customer ID' (with space) -> 'CustomerID'
    if "Customer ID" in df.columns:
        df.rename(columns={"Customer ID": "CustomerID"}, inplace=True)

    print(f"[load_data] Combined shape : {df.shape}")
    print(f"[load_data] Columns        : {list(df.columns)}")
    print(df.head(3))
    return df


# ---------------------------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a multi-step cleaning pipeline to the raw retail DataFrame.

    Cleaning steps (in order):
        1. Drop rows with null CustomerID  — guest/anonymous transactions
        2. Drop exact duplicate rows
        3. Remove cancelled invoices       — InvoiceNo starting with 'C'
        4. Remove rows with Quantity <= 0  — returns and data errors
        5. Remove rows with Price <= 0     — free items and data errors
        6. Parse InvoiceDate to datetime
        7. Engineer TotalPrice = Quantity * Price

    A cleaning report is printed showing rows removed at each step.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for analysis and feature engineering.
    """
    report = {}
    initial_rows = len(df)
    print("\n" + "=" * 55)
    print("  CLEANING REPORT")
    print("=" * 55)
    print(f"  Starting rows : {initial_rows:,}")

    # --- Step 1: Remove null CustomerID ---------------------------------
    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    removed = before - len(df)
    report["null_customer_id"] = removed
    print(f"  [-] Null CustomerID removed  : {removed:,}  (rows left: {len(df):,})")

    # --- Step 2: Remove duplicate rows ----------------------------------
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    report["duplicates"] = removed
    print(f"  [-] Duplicate rows removed   : {removed:,}  (rows left: {len(df):,})")

    # --- Step 3: Remove cancelled invoices ------------------------------
    before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    removed = before - len(df)
    report["cancellations"] = removed
    print(f"  [-] Cancelled invoices removed: {removed:,}  (rows left: {len(df):,})")

    # --- Step 4: Remove non-positive Quantity ---------------------------
    before = len(df)
    df = df[df["Quantity"] > 0]
    removed = before - len(df)
    report["bad_quantity"] = removed
    print(f"  [-] Quantity <= 0 removed    : {removed:,}  (rows left: {len(df):,})")

    # --- Step 5: Remove non-positive Price ------------------------------
    before = len(df)
    df = df[df["Price"] > 0]
    removed = before - len(df)
    report["bad_price"] = removed
    print(f"  [-] Price <= 0 removed       : {removed:,}  (rows left: {len(df):,})")

    # --- Step 6: Parse InvoiceDate to datetime --------------------------
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # --- Step 7: Engineer TotalPrice ------------------------------------
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    # --- Cast CustomerID to integer for consistency ---------------------
    df["CustomerID"] = df["CustomerID"].astype(int)

    total_removed = initial_rows - len(df)
    pct_removed   = (total_removed / initial_rows) * 100
    print(f"\n  Total rows removed : {total_removed:,}  ({pct_removed:.1f}% of raw data)")
    print(f"  Final shape        : {df.shape}")
    print("=" * 55 + "\n")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. DATA SUMMARY
# ---------------------------------------------------------------------------

def get_data_summary(df: pd.DataFrame) -> None:
    """
    Print a high-level summary of the cleaned dataset.

    Covers:
        - Date range of transactions
        - Unique customer, product, and invoice counts
        - Total revenue
        - Top 5 countries by transaction volume
        - Any remaining null values

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_data().
    """
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)

    date_min = df["InvoiceDate"].min().strftime("%Y-%m-%d")
    date_max = df["InvoiceDate"].max().strftime("%Y-%m-%d")
    print(f"  Date range         : {date_min}  ->  {date_max}")
    print(f"  Unique customers   : {df['CustomerID'].nunique():,}")
    print(f"  Unique products    : {df['StockCode'].nunique():,}")
    print(f"  Unique invoices    : {df['Invoice'].nunique():,}")
    print(f"  Total revenue (£)  : £{df['TotalPrice'].sum():,.2f}")

    print("\n  Top 5 countries by transaction volume:")
    top_countries = (
        df.groupby("Country")["Invoice"]
        .count()
        .sort_values(ascending=False)
        .head(5)
    )
    for country, count in top_countries.items():
        print(f"    {country:<25} {count:>8,}")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("\n  Remaining nulls    : None")
    else:
        print(f"\n  Remaining nulls:\n{nulls}")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# 4. SAVE PROCESSED DATA
# ---------------------------------------------------------------------------

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to a CSV file.

    Creates the output directory if it does not already exist.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to persist.
    output_path : str
        Relative or absolute path for the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[save_processed_data] Saved {len(df):,} rows -> {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Accepts either the .xlsx or .csv version of the UCI Online Retail II dataset.
    # Update RAW_PATH to match whichever format you have in data/raw/.
    RAW_PATH    = "data/raw/online_retail_II.csv"   # swap to .xlsx if using the Excel version
    OUTPUT_PATH = "data/processed/cleaned_retail.csv"

    df_raw   = load_data(RAW_PATH)
    df_clean = clean_data(df_raw)
    get_data_summary(df_clean)
    save_processed_data(df_clean, OUTPUT_PATH)
