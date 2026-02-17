from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

import requests


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MODEL_JOBLIB = Path("heloc_logreg_pipeline.joblib")
MODEL_META_JSON = Path("heloc_logreg_metadata.json")
DATA_DICTIONARY_XLSX = Path("heloc_data_dictionary-2.xlsx")

RAW_INPUT_FILLED_XLSX = Path("raw_input_filled.xlsx")

# Keep this consistent with training time dropping
DROP_COLS = [
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumInqLast6Mexcl7days",
    "MSinceMostRecentTradeOpen",
    "MaxDelq2PublicRecLast12M_4.0",
]

# Gemini Flash 2.5
USE_GEMINI = True  # set True when you want to call Gemini
GEMINI_API_KEY = "AIzaSyBF5mkMbI95PWJarrWESol-zuOWikJvew8"  
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


# ------------------------------------------------------------
# Metadata and dictionary helpers
# ------------------------------------------------------------
def load_metadata_required(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    required_keys = [
        "training_features_manipulated_order",
        "raw_input_features_order",
        "threshold_on_P(Bad)",
        "target_mapping",
    ]
    missing = [k for k in required_keys if k not in meta]
    if missing:
        raise ValueError(f"Metadata missing required keys: {missing}")

    return meta


def load_feature_dictionary(dict_path: Path) -> Dict[str, str]:
    """
    Reads the data dictionary Excel and builds a mapping: feature_name -> description.
    Tries several common column names, falls back to first two columns.
    """
    if not dict_path.exists():
        return {}

    df = pd.read_excel(dict_path)

    possible_name_cols = ["Variable", "Feature", "Name", "Column", "Field", "Attribute"]
    possible_desc_cols = ["Description", "Definition", "Meaning", "Notes", "Details"]

    name_col = next((c for c in possible_name_cols if c in df.columns), None)
    desc_col = next((c for c in possible_desc_cols if c in df.columns), None)

    if name_col is None or desc_col is None:
        if df.shape[1] >= 2:
            name_col = df.columns[0]
            desc_col = df.columns[1]
        else:
            return {}

    mapping = {}
    for _, row in df.iterrows():
        k = str(row[name_col]).strip()
        v = str(row[desc_col]).strip()
        if k and k.lower() != "nan":
            mapping[k] = v

    return mapping


# ------------------------------------------------------------
# Input helpers
# ------------------------------------------------------------
def read_raw_input_from_excel(xlsx_path: Path, raw_features: List[str]) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input Excel not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    if df.shape[0] < 1:
        raise ValueError("Excel has no rows.")

    missing = [c for c in raw_features if c not in df.columns]
    if missing:
        raise ValueError(f"Excel is missing required raw columns: {missing}")

    one = df.loc[[0], raw_features].copy()
    return one


def read_raw_input_manually(raw_features: List[str]) -> pd.DataFrame:
    values = {}
    print("\nEnter raw feature values. Press Enter to leave blank.\n")

    for f in raw_features:
        s = input(f"{f}: ").strip()
        if s == "":
            values[f] = np.nan
        else:
            try:
                values[f] = float(s)
            except ValueError:
                print("Invalid number, stored as NaN.")
                values[f] = np.nan

    return pd.DataFrame([values], columns=raw_features)


# ------------------------------------------------------------
# RAW -> MANIPULATED transformation
# ------------------------------------------------------------
def raw_to_manipulated(
    raw_row: pd.DataFrame,
    expected_manip_features: List[str],
    drop_cols: List[str],
) -> pd.DataFrame:
    """
    Converts a single raw input row into a manipulated feature row expected by the trained model.

    Generic rules implemented
    1) For any raw feature with value -7 or -8:
       - set base feature to NaN
       - create flags feature_is_m7 and feature_is_m8 if expected
    2) ExternalRiskEstimate == -9:
       - set ExternalRiskEstimate to NaN
       - create NoBureau if expected
    3) CountMinus7 and CountMinus8 if expected
    4) Ensures DROP_COLS are not passed to the model (even if present in expected features)

    If your manipulated dataset creates more engineered features, extend this function.
    """
    if raw_row.shape[0] != 1:
        raise ValueError("raw_to_manipulated expects exactly one row.")

    raw = raw_row.iloc[0].to_dict()

    minus7_count = 0
    minus8_count = 0

    # Initialize all expected manipulated features as NaN
    out = {col: np.nan for col in expected_manip_features}

    # Pass through values that exist in both raw and expected manipulated
    for k, v in raw.items():
        if k in out:
            out[k] = v

    # Special codes -7 and -8
    for raw_col, v in raw.items():
        flag7 = f"{raw_col}_is_m7"
        flag8 = f"{raw_col}_is_m8"

        if flag7 in out and pd.isna(out[flag7]):
            out[flag7] = 0.0
        if flag8 in out and pd.isna(out[flag8]):
            out[flag8] = 0.0

        if pd.isna(v):
            continue

        if v == -7:
            minus7_count += 1
            if raw_col in out:
                out[raw_col] = np.nan
            if flag7 in out:
                out[flag7] = 1.0

        if v == -8:
            minus8_count += 1
            if raw_col in out:
                out[raw_col] = np.nan
            if flag8 in out:
                out[flag8] = 1.0

    # No bureau handling
    if "ExternalRiskEstimate" in raw and not pd.isna(raw["ExternalRiskEstimate"]):
        if raw["ExternalRiskEstimate"] == -9:
            if "ExternalRiskEstimate" in out:
                out["ExternalRiskEstimate"] = np.nan
            if "NoBureau" in out:
                out["NoBureau"] = 1.0
        else:
            if "NoBureau" in out and pd.isna(out["NoBureau"]):
                out["NoBureau"] = 0.0
    else:
        if "NoBureau" in out and pd.isna(out["NoBureau"]):
            out["NoBureau"] = 0.0

    # Counts
    if "CountMinus7" in out:
        out["CountMinus7"] = float(minus7_count)
    if "CountMinus8" in out:
        out["CountMinus8"] = float(minus8_count)

    # Force drop columns if they appear
    for c in drop_cols:
        if c in out:
            out.pop(c, None)

    # Ensure final columns match model expectation order (excluding dropped if any popped)
    final_cols = [c for c in expected_manip_features if c not in drop_cols]
    df_out = pd.DataFrame([[out.get(c, np.nan) for c in final_cols]], columns=final_cols)

    return df_out


# ------------------------------------------------------------
# Coefficient based reasons and suggestions
# ------------------------------------------------------------
def get_logreg_coefficients(pipeline) -> Tuple[np.ndarray, float]:
    clf = pipeline.named_steps.get("clf", None)
    if clf is None or not hasattr(clf, "coef_"):
        raise ValueError("Pipeline does not contain a logistic regression classifier named 'clf'.")
    return clf.coef_.ravel(), float(clf.intercept_.ravel()[0])


def compute_feature_contributions(
    pipeline,
    X_manip_one: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    contribution_j = coef_j * x_preprocessed_j
    Ranking by contribution_to_logit gives a consistent "drivers toward Bad" order.
    """
    coef, intercept = get_logreg_coefficients(pipeline)

    prep = pipeline.named_steps["prep"]
    x_proc = prep.transform(X_manip_one)

    x_vec = np.asarray(x_proc).ravel()
    if x_vec.shape[0] != coef.shape[0]:
        raise ValueError("Mismatch between preprocessed feature length and coefficient length.")

    contrib = coef * x_vec

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coef,
            "x_preprocessed": x_vec,
            "contribution_to_logit": contrib,
        }
    ).sort_values("contribution_to_logit", ascending=False)

    df["intercept"] = intercept
    df["logit_sum"] = intercept + df["contribution_to_logit"].sum()
    return df


def suggestion_for_feature(feature_name: str) -> Optional[str]:
    f = feature_name.lower()

    if "inq" in f or "inquir" in f:
        return "Reduce new credit inquiries and avoid multiple applications in a short period."
    if "util" in f or "revolv" in f:
        return "Lower revolving utilization by paying down balances and keeping utilization stable."
    if "delq" in f or "delin" in f:
        return "Avoid missed payments. Use autopay and rebuild on time payment history."
    if "publicrec" in f or "derog" in f:
        return "Resolve derogatory items if possible and keep all accounts current."
    if "age" in f or "msince" in f or "m_since" in f:
        return "Build longer credit history over time and keep older accounts in good standing."
    if "trade" in f:
        return "Avoid opening many new accounts quickly and maintain a stable credit mix."
    if "balance" in f:
        return "Reduce outstanding balances and keep debt manageable."
    return None


def build_rejection_package(
    pipeline,
    X_manip_one: pd.DataFrame,
    expected_manip_features: List[str],
    feature_desc: Dict[str, str],
    drop_cols: List[str],
    top_k: int = 6,
) -> dict:
    feature_names = [c for c in expected_manip_features if c not in drop_cols]
    contrib_df = compute_feature_contributions(pipeline, X_manip_one, feature_names)

    top = contrib_df.head(top_k).copy()
    reasons = []
    suggestions = []

    for _, row in top.iterrows():
        fname = row["feature"]
        desc = feature_desc.get(fname, "")
        reasons.append(
            {
                "feature": fname,
                "description": desc,
                "coef": float(row["coef"]),
                "contribution_to_logit": float(row["contribution_to_logit"]),
            }
        )

        sug = suggestion_for_feature(fname)
        if sug and sug not in suggestions:
            suggestions.append(sug)

    return {
        "top_drivers_toward_bad": reasons,
        "suggestions": suggestions[:5],
        "notes": [
            "Drivers are derived from logistic regression coefficients times the preprocessed input values.",
            "This is directional, not causal.",
        ],
    }


# ------------------------------------------------------------
# Gemini call
# ------------------------------------------------------------
def call_gemini_flash(explanation_package: dict) -> str:
    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY in script or as env var.")

    prompt = {
        "task": "Explain rejection and suggest improvements",
        "style": "brief and applicant friendly",
        "inputs": explanation_package,
        "output_requirements": {
            "length": "120 to 200 words",
            "format": "one short paragraph, then 3 bullet suggestions",
        },
        "constraints": [
            "Do not mention protected traits.",
            "Do not guarantee approval.",
            "Keep it practical and action focused.",
        ],
    }

    body = {
        "contents": [{"role": "user", "parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}]
    }

    r = requests.post(
        GEMINI_ENDPOINT,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=60,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text}")

    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(data, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not MODEL_JOBLIB.exists():
        raise FileNotFoundError(f"Model joblib not found: {MODEL_JOBLIB}")

    pipeline = joblib.load(MODEL_JOBLIB)
    metadata = load_metadata_required(MODEL_META_JSON)

    expected_manip_features = list(metadata["training_features_manipulated_order"])
    raw_features = list(metadata["raw_input_features_order"])
    threshold = float(metadata["threshold_on_P(Bad)"])

    target_mapping = metadata["target_mapping"]
    label_0 = target_mapping.get("0", "Good")
    label_1 = target_mapping.get("1", "Bad")

    feature_desc = load_feature_dictionary(DATA_DICTIONARY_XLSX)

    print("\nHELOC Inference")
    print(f"Loaded model: {MODEL_JOBLIB.name}")
    print(f"Threshold on P({label_1}): {threshold}")
    print(f"Output meaning: 0 = {label_0}, 1 = {label_1}")

    print("\nChoose input method")
    print("1) Read raw inputs from filled Excel")
    print("2) Enter raw inputs manually in terminal")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        raw_one = read_raw_input_from_excel(RAW_INPUT_FILLED_XLSX, raw_features)
    elif choice == "2":
        raw_one = read_raw_input_manually(raw_features)
    else:
        print("Invalid choice.")
        return

    X_manip_one = raw_to_manipulated(
        raw_row=raw_one,
        expected_manip_features=expected_manip_features,
        drop_cols=DROP_COLS,
    )

    p_bad = float(pipeline.predict_proba(X_manip_one)[:, 1][0])
    y_pred = 1 if p_bad >= threshold else 0

    decision_text = (
        f"{label_1} (1) -> reject automatically"
        if y_pred == 1
        else f"{label_0} (0) -> approved for more officer review"
    )

    print("\nPrediction Result")
    print(f"P({label_1}): {p_bad:.4f}")
    print(f"Predicted label: {y_pred}")
    print(f"Decision: {decision_text}")

    if y_pred == 1:
        package = build_rejection_package(
            pipeline=pipeline,
            X_manip_one=X_manip_one,
            expected_manip_features=expected_manip_features,
            feature_desc=feature_desc,
            drop_cols=DROP_COLS,
            top_k=6,
        )

        print("\nRejection package (for LLM or UI)")
        print(json.dumps(package, indent=2, ensure_ascii=False))

        if USE_GEMINI:
            explanation = call_gemini_flash(package)
            print("\nGemini explanation")
            print(explanation)

    print("\nDone.")


if __name__ == "__main__":
    main()
