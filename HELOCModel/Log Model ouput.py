"""
UPDATED: Logistic Regression training + outputs

Changes you requested
1) Drop these manipulated columns BEFORE training:
   - "MaxDelq2PublicRecLast12M"
   - "MaxDelqEver"
2) Excel output now includes a "Coefficients" sheet with:
   - feature name
   - coefficient
   - odds ratio (exp(coef))
   - absolute coefficient (for quick sorting)
   - intercept (separate sheet row)

This is an update to your current script. :contentReference[oaicite:0]{index=0}
Do not run here. Code only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

import joblib


# -----------------------------
# Config
# -----------------------------
RAW_DATA_PATH = Path("heloc_dataset_v1.csv")
MANIP_DATA_PATH = Path("heloc_dataset_v1_manipulated.csv")

TARGET_COL = "RiskPerformance"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Adjustable classification threshold (set to what your team decides)
THRESHOLD = 0.775

# Drop these columns from manipulated dataset before training (duplicated information)
DROP_COLS = ["MaxDelq2PublicRecLast12M", "MaxDelqEver","NumInqLast6Mexcl7days","MSinceMostRecentTradeOpen","MaxDelq2PublicRecLast12M_4.0"]

# Output files
MODEL_PIPELINE_PATH = Path("heloc_logreg_pipeline.joblib")
MODEL_META_PATH = Path("heloc_logreg_metadata.json")
EXCEL_REPORT_PATH = Path("heloc_logreg_report.xlsx")


# -----------------------------
# Helpers
# -----------------------------
def encode_target_to_0_1(y: pd.Series) -> tuple[np.ndarray, dict]:
    """
    Convert RiskPerformance labels to 0/1 with explicit mapping:
      Good -> 0
      Bad  -> 1
    """
    y_str = y.astype(str).str.strip().str.lower()
    y01 = np.where(y_str == "bad", 1, 0)

    mapping = {
        "0": "Good",
        "1": "Bad",
        "positive_class_is_1": True,
        "note": "Predicted probability is P(Bad).",
    }
    return y01.astype(int), mapping


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """
    Numeric preprocessing:
      - median imputation for NaNs
      - standard scaling (LR benefits)
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def compute_metrics(y_true: np.ndarray, p_bad: np.ndarray, threshold: float) -> dict:
    y_pred = (p_bad >= threshold).astype(int)

    out = {
        "Threshold": threshold,
        "ROC_AUC": roc_auc_score(y_true, p_bad),
        "PR_AUC_AvgPrecision": average_precision_score(y_true, p_bad),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Pos1": precision_score(y_true, y_pred, zero_division=0),
        "Recall_Pos1": recall_score(y_true, y_pred, zero_division=0),
        "F1_Pos1": f1_score(y_true, y_pred, zero_division=0),
        "LogLoss": log_loss(y_true, np.vstack([1 - p_bad, p_bad]).T, labels=[0, 1]),
    }
    return out


def labeled_confusion_df(y_true: np.ndarray, p_bad: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Confusion matrix with explicit row/column meaning and class labels.
    Positive class = 1 = Bad
    Rows = Actual (Fact)
    Columns = Predicted
    """
    y_pred = (p_bad >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    idx = ["Actual 0 (Good)", "Actual 1 (Bad)"]
    cols = ["Pred 0 (Good)", "Pred 1 (Bad)"]
    df_cm = pd.DataFrame(cm, index=idx, columns=cols)
    return df_cm


def build_coefficients_df(pipeline: Pipeline, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exports coefficients aligned with feature_cols (the columns fed into the ColumnTransformer).
    NOTE: Because we scale features, these coefficients are on standardized features.
    """
    clf = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_.ravel()[0])

    coef_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Coefficient": coef,
            "Odds_Ratio_exp(coef)": np.exp(coef),
            "Abs_Coefficient": np.abs(coef),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    intercept_df = pd.DataFrame(
        {
            "Term": ["Intercept"],
            "Value": [intercept],
            "Note": ["Log-odds intercept (with standardized features)."],
        }
    )

    return coef_df, intercept_df


# -----------------------------
# Main training routine
# -----------------------------
def main():
    # 1) Load raw data only to build "input feature list"
    df_raw = pd.read_csv(RAW_DATA_PATH)
    if TARGET_COL not in df_raw.columns:
        raise ValueError(f"Raw dataset missing target column '{TARGET_COL}'.")

    raw_input_features = [c for c in df_raw.columns if c != TARGET_COL]

    # 2) Load manipulated data for training
    df = pd.read_csv(MANIP_DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Manipulated dataset missing target column '{TARGET_COL}'.")

    # Drop the duplicated columns (if present)
    drop_cols_present = [c for c in DROP_COLS if c in df.columns]
    if drop_cols_present:
        df = df.drop(columns=drop_cols_present)

    X = df.drop(columns=[TARGET_COL]).copy()
    y01, y_mapping = encode_target_to_0_1(df[TARGET_COL])

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y01,
        test_size=TEST_SIZE,
        stratify=y01,
        random_state=RANDOM_STATE,
    )

    # 4) Build pipeline
    feature_cols = list(X.columns)  # manipulated feature list after dropping
    preprocessor = make_preprocessor(feature_cols)

    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        penalty="l2",
        C=1,  # set to your selected C from CV
        class_weight="balanced",  # remove if you want no class weighting
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )

    # 5) Fit once on training set
    pipeline.fit(X_train, y_train)

    # 6) Evaluate on test
    p_bad_test = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, p_bad_test, threshold=THRESHOLD)
    cm_df = labeled_confusion_df(y_test, p_bad_test, threshold=THRESHOLD)

    # 7) Save pipeline
    joblib.dump(pipeline, MODEL_PIPELINE_PATH)

    # Save metadata (make sure manipulated feature list reflects dropped columns)
    metadata = {
        "model_file": str(MODEL_PIPELINE_PATH),
        "trained_on": "manipulated_dataset",
        "dropped_columns_from_manipulated": drop_cols_present,
        "training_features_manipulated_order": feature_cols,
        "raw_input_features_order": raw_input_features,
        "target_mapping": y_mapping,
        "threshold_on_P(Bad)": THRESHOLD,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "notes": [
            "Pipeline expects manipulated features at prediction time.",
            "If user provides RAW features, you must transform raw -> manipulated before calling pipeline.",
            "Coefficients are on standardized (scaled) features.",
        ],
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 8) Build Excel report tables
    metrics_df = pd.DataFrame([metrics])

    cm_legend_df = pd.DataFrame(
        {
            "Confusion Matrix Legend": [
                "Rows are ACTUAL (Fact)",
                "Columns are PREDICTED",
                "0 means Good",
                "1 means Bad",
                f"Threshold applied on P(Bad): {THRESHOLD}",
            ]
        }
    )

    manipulated_features_df = pd.DataFrame({"Manipulated_Training_Feature_Order": feature_cols})
    raw_input_features_df = pd.DataFrame({"Raw_Input_Feature_Order": raw_input_features})

    coef_df, intercept_df = build_coefficients_df(pipeline, feature_cols)

    # 9) Write Excel
    with pd.ExcelWriter(EXCEL_REPORT_PATH, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Performance", index=False)

        cm_legend_df.to_excel(writer, sheet_name="Confusion_Matrix", index=False, startrow=0)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix", startrow=len(cm_legend_df) + 2)

        raw_input_features_df.to_excel(writer, sheet_name="Raw_Input_Features", index=False)
        manipulated_features_df.to_excel(writer, sheet_name="Manipulated_Training_Features", index=False)

        coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
        intercept_df.to_excel(writer, sheet_name="Intercept", index=False)

        pd.DataFrame({"Metadata": [json.dumps(metadata, indent=2)]}).to_excel(
            writer, sheet_name="Model_Metadata", index=False
        )

    print(f"Saved pipeline: {MODEL_PIPELINE_PATH.resolve()}")
    print(f"Saved metadata: {MODEL_META_PATH.resolve()}")
    print(f"Saved report: {EXCEL_REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
