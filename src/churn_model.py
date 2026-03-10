"""
churn_model.py
--------------
Module for building and evaluating a churn prediction model using XGBoost.
Defines churn based on customer inactivity (segment membership), trains a
supervised binary classifier, and outputs per-customer churn probabilities
with feature importance analysis.

Churn definition (derived from Phase 2 segmentation):
    Churned (1)  — Segment == "Dormant / At-Risk"
    Retained (0) — Segment == "Champions"

Pipeline:
    1. prepare_churn_data()      — Build feature matrix X and label vector y
    2. split_and_scale()         — Train/test split + StandardScaler
    3. train_xgboost()           — Fit XGBClassifier + log to MLflow
    4. evaluate_model()          — Metrics, confusion matrix, ROC curve
    5. get_feature_importance()  — Feature importance bar chart
    6. add_churn_probability()   — Score all customers with churn probability
    7. save_churn_results()      — Persist final DataFrame

Usage:
    python src/churn_model.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# MLflow configuration
# Using SQLite backend (filesystem tracking deprecated in MLflow 3.x)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT   = "churn_prediction"

# Feature columns used for modelling
FEATURE_COLS = ["Recency", "Frequency", "Monetary"]


# ---------------------------------------------------------------------------
# 1. PREPARE CHURN DATA
# ---------------------------------------------------------------------------

def prepare_churn_data(rfm: pd.DataFrame) -> tuple:
    """
    Build the feature matrix (X) and binary churn label (y) from the
    RFM segmentation output.

    Churn label encoding:
        1 = Dormant / At-Risk  (customer has churned or is at high risk)
        0 = Champions          (customer is retained and active)

    Features used: Recency, Frequency, Monetary (raw, unscaled).
    Scaling is handled separately in split_and_scale() to prevent data
    leakage from the test set into the scaler.

    Parameters
    ----------
    rfm : pd.DataFrame
        RFM DataFrame from Phase 2 with a 'Segment' column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (Recency, Frequency, Monetary).
    y : pd.Series
        Binary churn label (1 = churned, 0 = retained).
    """
    churn_map = {"Dormant / At-Risk": 1, "Champions": 0}
    y = rfm["Segment"].map(churn_map).astype(int)
    X = rfm[FEATURE_COLS].copy()

    n_total   = len(y)
    n_churned = y.sum()
    churn_rate = n_churned / n_total * 100

    print("\n" + "=" * 55)
    print("  CHURN LABEL DISTRIBUTION")
    print("=" * 55)
    print(f"  Total customers   : {n_total:,}")
    print(f"  Churned  (label=1): {n_churned:,}  ({churn_rate:.1f}%)")
    print(f"  Retained (label=0): {n_total - n_churned:,}  ({100 - churn_rate:.1f}%)")
    print(f"\n  Class imbalance ratio: {n_churned / (n_total - n_churned):.2f}:1 (churned:retained)")
    print("=" * 55)

    return X, y


# ---------------------------------------------------------------------------
# 2. SPLIT AND SCALE
# ---------------------------------------------------------------------------

def split_and_scale(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Perform stratified train/test split and apply StandardScaler.

    Stratification ensures the churn rate is equal in both splits,
    which is important when class imbalance is present.

    StandardScaler is fitted ONLY on the training set to prevent data
    leakage: the test set is transformed using training statistics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from prepare_churn_data().
    y : pd.Series
        Binary churn labels.

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled feature arrays.
    y_train, y_test : pd.Series
        Label arrays.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler (needed later to score new customers).
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    print("\n" + "=" * 55)
    print("  TRAIN / TEST SPLIT")
    print("=" * 55)
    print(f"  Training set : {len(y_train):,} customers"
          f"  (churn rate: {y_train.mean() * 100:.1f}%)")
    print(f"  Test set     : {len(y_test):,}  customers"
          f"  (churn rate: {y_test.mean() * 100:.1f}%)")
    print("  Stratification confirmed — churn rate equal in both splits.")
    print("=" * 55)

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# 3. TRAIN XGBOOST
# ---------------------------------------------------------------------------

def train_xgboost(X_train: np.ndarray, y_train: pd.Series) -> tuple:
    """
    Train an XGBoost binary classifier and log the experiment to MLflow.

    The MLflow run is left OPEN after this function returns so that
    evaluation metrics (computed in evaluate_model) can be logged to
    the same run. Call mlflow.end_run() after evaluate_model().

    Hyperparameters are chosen to balance model expressiveness against
    overfitting on a dataset of ~4,700 training samples:
        - Shallow trees (max_depth=4) to avoid memorising noise
        - Moderate learning rate (0.05) with more estimators (200)
        - Row and column subsampling (0.8) for variance reduction

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features from split_and_scale().
    y_train : pd.Series
        Training labels.

    Returns
    -------
    model : XGBClassifier
        Trained classifier.
    run_id : str
        MLflow run ID (used to retrieve the logged experiment later).
    """
    params = {
        "n_estimators"   : 200,
        "max_depth"      : 4,
        "learning_rate"  : 0.05,
        "subsample"      : 0.8,
        "colsample_bytree": 0.8,
        "random_state"   : 42,
        "eval_metric"    : "logloss",
        "verbosity"      : 0,
    }

    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    active = mlflow.active_run()
    if active:
        mlflow.end_run()

    run = mlflow.start_run()
    mlflow.log_params({k: v for k, v in params.items() if k != "verbosity"})

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    mlflow.xgboost.log_model(model, artifact_path="xgboost_churn_model")

    print("\n" + "=" * 55)
    print("  XGBOOST TRAINING COMPLETE")
    print("=" * 55)
    print(f"  MLflow experiment : {MLFLOW_EXPERIMENT}")
    print(f"  Run ID            : {run.info.run_id}")
    print(f"  Params logged     : {len(params) - 1}")
    print(f"  Model artifact    : xgboost_churn_model")
    print("  (Run still active — metrics will be logged after evaluation)")
    print("=" * 55)

    return model, run.info.run_id


# ---------------------------------------------------------------------------
# 4. EVALUATE MODEL
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
    output_dir: str = "data/processed",
) -> dict:
    """
    Evaluate the trained classifier and generate diagnostic plots.

    Metrics computed:
        - Accuracy  : overall correct predictions
        - Precision : of those predicted to churn, how many actually do
        - Recall    : of those who actually churned, how many we caught
        - F1 Score  : harmonic mean of precision and recall
        - ROC-AUC   : area under the ROC curve (threshold-independent)

    For churn detection, RECALL is the primary business metric:
    a missed churner (false negative) means lost revenue with no chance
    to intervene.  However, very low precision wastes campaign budget
    on customers who would have stayed anyway.

    Plots saved:
        - data/processed/confusion_matrix.png
        - data/processed/roc_curve.png

    Parameters
    ----------
    model : XGBClassifier
        Trained model from train_xgboost().
    X_test : np.ndarray
        Scaled test features.
    y_test : pd.Series
        True test labels.
    output_dir : str
        Directory for saving plot files.

    Returns
    -------
    dict
        All five metric values keyed by name.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : round(float(accuracy_score(y_test, y_pred)),  4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall"   : round(float(recall_score(y_test, y_pred)),    4),
        "f1"       : round(float(f1_score(y_test, y_pred)),        4),
        "roc_auc"  : round(float(roc_auc_score(y_test, y_prob)),   4),
    }

    # Log metrics to the active MLflow run and close it
    if mlflow.active_run():
        mlflow.log_metrics(metrics)
        mlflow.end_run()
        print("\n  [MLflow] Metrics logged and run closed.")

    # --- Print metrics table ------------------------------------------------
    print("\n" + "=" * 55)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  {'Metric':<12}  {'Score':>8}")
    print("  " + "-" * 24)
    for name, val in metrics.items():
        print(f"  {name:<12}  {val:>8.4f}")
    print("=" * 55)

    # --- Business interpretation -------------------------------------------
    recall_pct    = metrics["recall"]    * 100
    precision_pct = metrics["precision"] * 100
    print(f"""
  Business Interpretation
  -----------------------
  A recall of {recall_pct:.1f}% means we correctly identify {recall_pct:.1f}% of
  customers who will churn — giving us the opportunity to intervene
  before they are lost.

  A precision of {precision_pct:.1f}% means that {precision_pct:.1f}% of customers
  we flag as churning will actually churn. The remaining {100 - precision_pct:.1f}%
  are false alarms — customers we would spend retention budget on
  unnecessarily.

  For this use case, high recall is prioritised over high precision:
  the cost of missing a churner (lost lifetime value) far exceeds
  the cost of sending an unnecessary win-back email.
""")

    os.makedirs(output_dir, exist_ok=True)

    # --- Confusion matrix ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Retained (0)", "Churned (1)"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — XGBoost Churn Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}")

    # --- ROC curve ----------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"XGBoost (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Churn Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved        -> {roc_path}")

    return metrics


# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: XGBClassifier,
    feature_names: list,
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Extract and visualise XGBoost feature importances.

    XGBoost's default importance type is 'weight' (number of times a
    feature is used in a split across all trees). We use this alongside
    a business interpretation to explain which RFM dimension is the
    strongest churn signal.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model.
    feature_names : list
        Ordered list of feature names matching the training columns.
    output_dir : str
        Directory for saving the importance plot.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns, sorted
        descending by importance.
    """
    importances = model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["steelblue" if i == 0 else "lightsteelblue"
              for i in range(len(importance_df))]
    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color=colors[::-1],
        edgecolor="white",
    )
    ax.set_xlabel("Feature Importance (XGBoost weight)")
    ax.set_title("Feature Importance — Churn Prediction Model",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Feature importance plot saved -> {fi_path}")

    # --- Business interpretation -------------------------------------------
    top = importance_df.iloc[0]
    interpretations = {
        "Recency": (
            "Recency is the strongest predictor of churn. Customers who have not "
            "purchased recently are overwhelmingly likely to never return. This "
            "suggests re-engagement campaigns should be triggered within the first "
            "60–90 days of inactivity, before the customer fully disengages."
        ),
        "Frequency": (
            "Purchase frequency is the strongest predictor of churn. Customers who "
            "made only one or two purchases are at much higher risk than repeat buyers. "
            "This highlights the critical importance of the second-purchase conversion — "
            "the moment a one-time buyer becomes a repeat customer."
        ),
        "Monetary": (
            "Total spend is the strongest predictor of churn. Low-spend customers churn "
            "at a higher rate, likely because they never found enough value to become "
            "loyal. High-spend customers, by contrast, have demonstrated commitment and "
            "are worth defending with premium retention offers."
        ),
    }

    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE RESULTS")
    print("=" * 60)
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']:<12}: {row['Importance']:.4f}")
    print(f"\n  Most predictive feature: {top['Feature']}")
    print(f"\n  Why it makes business sense:")
    print(f"  {interpretations.get(top['Feature'], '')}")
    print("=" * 60)

    return importance_df


# ---------------------------------------------------------------------------
# 6. ADD CHURN PROBABILITY
# ---------------------------------------------------------------------------

def add_churn_probability(
    rfm: pd.DataFrame,
    model: XGBClassifier,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Score all customers with their predicted churn probability.

    The churn probability is the model's predicted probability that a
    customer belongs to class 1 (Dormant / At-Risk). This continuous
    score is more useful than a binary label because it supports:
        - Tiered intervention strategies (critical / high / medium / low)
        - Prioritisation of retention budget toward highest-risk customers

    Parameters
    ----------
    rfm : pd.DataFrame
        Full RFM DataFrame (all 5,878 customers).
    model : XGBClassifier
        Trained model from train_xgboost().
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on training data from split_and_scale().

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns:
            churn_probability — float [0, 1]
            churn_predicted   — int {0, 1}
    """
    X = rfm[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    rfm = rfm.copy()
    rfm["churn_probability"] = model.predict_proba(X_scaled)[:, 1]
    rfm["churn_predicted"]   = model.predict(X_scaled)

    probs = rfm["churn_probability"]
    print("\n" + "=" * 55)
    print("  CHURN PROBABILITY DISTRIBUTION")
    print("=" * 55)
    print(f"  Mean probability  : {probs.mean():.3f}")
    print(f"  Median probability: {probs.median():.3f}")
    print(f"  Std deviation     : {probs.std():.3f}")
    print()

    tiers = [
        ("Critical (0.8–1.0)", (probs >= 0.8).sum()),
        ("High     (0.6–0.8)", ((probs >= 0.6) & (probs < 0.8)).sum()),
        ("Medium   (0.4–0.6)", ((probs >= 0.4) & (probs < 0.6)).sum()),
        ("Low      (0.0–0.4)", (probs < 0.4).sum()),
    ]
    print(f"  {'Risk Tier':<22} {'Customers':>10}  {'% of Base':>10}")
    print("  " + "-" * 46)
    for tier, count in tiers:
        pct = count / len(rfm) * 100
        print(f"  {tier:<22} {count:>10,}  {pct:>9.1f}%")
    print("=" * 55)

    return rfm


# ---------------------------------------------------------------------------
# 7. SAVE CHURN RESULTS
# ---------------------------------------------------------------------------

def save_churn_results(rfm: pd.DataFrame, output_path: str) -> None:
    """
    Persist the scored RFM DataFrame (with churn probabilities) to CSV.

    Parameters
    ----------
    rfm : pd.DataFrame
        Scored DataFrame from add_churn_probability().
    output_path : str
        Destination file path (e.g. 'data/processed/churn_results.csv').
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rfm.to_csv(output_path, index=False)
    print(f"\n[save_churn_results] Saved {len(rfm):,} rows -> {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rfm = pd.read_csv(
        "data/processed/rfm_segments.csv",
        index_col=None,   # CustomerID is a regular column, not an index
    )

    X, y                               = prepare_churn_data(rfm)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    model, run_id                      = train_xgboost(X_train, y_train)
    metrics                            = evaluate_model(model, X_test, y_test)
    importance_df                      = get_feature_importance(model, X.columns.tolist())
    rfm                                = add_churn_probability(rfm, model, scaler)
    save_churn_results(rfm, "data/processed/churn_results.csv")

    print(f"\nPhase 3 complete. MLflow run ID: {run_id}")
