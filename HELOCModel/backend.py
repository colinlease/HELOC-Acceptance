# backend.py  - HELOC Application Portal Backend

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import joblib
import requests

# Paths (works locally + on Streamlit Cloud)
# Put artifacts either:
#   (A) in same folder as backend.py  OR
#   (B) in ./artifacts/

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = (BASE_DIR / "artifacts") if (BASE_DIR / "artifacts").exists() else BASE_DIR

MODEL_JOBLIB = ARTIFACT_DIR / "heloc_logreg_pipeline.joblib"
MODEL_META_JSON = ARTIFACT_DIR / "heloc_logreg_metadata.json"
DATA_DICTIONARY_XLSX = ARTIFACT_DIR / "heloc_data_dictionary-2.xlsx"

# Keep this consistent with training time dropping
DROP_COLS = [
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumInqLast6Mexcl7days",
    "MSinceMostRecentTradeOpen",
    "MaxDelq2PublicRecLast12M_4.0",
]

# Gemini Flash 2.5
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Metadata and dictionary helpers
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

# RAW -> MANIPULATED transformation
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

    # Ensure final columns match model expectation order (excluding dropped)
    final_cols = [c for c in expected_manip_features if c not in drop_cols]
    df_out = pd.DataFrame([[out.get(c, np.nan) for c in final_cols]], columns=final_cols)

    return df_out


# Coefficient based reasons and suggestions
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


def build_admin_diagnostics(
    pipeline,
    X_manip_one: pd.DataFrame,
    expected_manip_features: List[str],
    feature_desc: Dict[str, str],
    drop_cols: List[str],
    decision: str,
    top_k: int = 5,
) -> dict:
    """Builds internal diagnostics for admin display.

    Includes:
      - NoBureau / CountMinus7 / CountMinus8 special-code summary
      - Top-k feature contributions aligned to the decision:
          * decision == 'deny'    -> strongest drivers toward Bad (largest positive contributions)
          * decision == 'forward' -> strongest drivers toward Good (most negative contributions)

    Notes: contribution_to_logit = coef * x_preprocessed.
    Positive contributions push toward the model's positive class (Bad).
    Negative contributions push toward Good.
    """
    # --- Special code summary (from manipulated features) ---
    def _safe_float(col: str) -> float:
        if col in X_manip_one.columns:
            v = X_manip_one.iloc[0][col]
            if pd.isna(v):
                return 0.0
            return float(v)
        return 0.0

    no_bureau = int(_safe_float("NoBureau") > 0)
    count_m7 = int(round(_safe_float("CountMinus7")))
    count_m8 = int(round(_safe_float("CountMinus8")))

    if (no_bureau == 0) and (count_m7 == 0) and (count_m8 == 0):
        special_codes_message = "No special codes detected"
        special_codes_detected = False
    else:
        special_codes_detected = True
        parts = []
        parts.append(f"NoBureau: {'Yes' if no_bureau else 'No'}")
        parts.append(f"-7 codes: {count_m7}")
        parts.append(f"-8 codes: {count_m8}")
        special_codes_message = " | ".join(parts)

    # --- Contribution list aligned to decision ---
    feature_names = [c for c in expected_manip_features if c not in drop_cols]
    contrib_df = compute_feature_contributions(pipeline, X_manip_one, feature_names)

    if decision == "deny":
        # Largest positive contributions push toward Bad
        picked = contrib_df.sort_values("contribution_to_logit", ascending=False).head(top_k)
        direction = "toward_bad"
    else:
        # Most negative contributions push toward Good
        picked = contrib_df.sort_values("contribution_to_logit", ascending=True).head(top_k)
        direction = "toward_good"

    top_contributors: List[dict] = []
    for _, row in picked.iterrows():
        fname = str(row["feature"])
        top_contributors.append(
            {
                "feature": fname,
                "description": feature_desc.get(fname, ""),
                "coef": float(row["coef"]),
                "contribution_to_logit": float(row["contribution_to_logit"]),
                "direction": direction,
            }
        )

    return {
        "special_codes": {
            "NoBureau": no_bureau,
            "CountMinus7": count_m7,
            "CountMinus8": count_m8,
            "detected": special_codes_detected,
            "message": special_codes_message,
        },
        "top_contributors": top_contributors,
        "notes": [
            "Internal diagnostics for grading/troubleshooting.",
            "Feature contributions are coef * preprocessed_value (logit space).",
        ],
    }


# Gemini call (NO hardcoded key)
def call_gemini_flash(explanation_package: dict, gemini_api_key: Optional[str] = None) -> str:
    api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY env var or Streamlit secrets.")

    prompt = {
        "role": "You are a loan decision assistant. Your job is to explain THIS specific denial using ONLY the provided inputs.",
        "task": (
            "Write a borrower-friendly denial explanation grounded in the provided inputs. "
            "Identify the 2–3 most important reasons this application was denied, using the actual fields and values. "
            "Then give 3 tailored next-step suggestions that directly correspond to those reasons."
            "Do NOT reference the actual field names, use consumer-friendly language to explain why the application was denied."
        ),
        "inputs": explanation_package,
        "output_requirements": {
            "length": "120 to 180 words",
            "format": [
            "One short paragraph summarizing the decision",
            "Then a section titled 'Main reasons' with 2–4 bullets",
            "Then a section titled 'What you can do next' with 2-4 bullets"
            ]
        },
        "hard_rules": [
            "Use ONLY the provided inputs. Do not invent facts or assume missing information.",
            "If NoBureau=1 or CountMinus7/8>0, explicitly say the credit file appears thin/insufficient.",
            "Do not mention protected traits. Do not guarantee approval.",
            "Do not reference the actual field names in your answer, explain in plain language why the application was denied."
        ],
        "style": {
            "tone": "plain English, applicant-friendly, specific",
            "avoid": ["generic advice not linked to a reason", "jargon like 'log-odds'"]
        }
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

# Artifact loading (cache this in Streamlit)
def load_artifacts() -> Tuple[Any, dict, Dict[str, str]]:
    if not MODEL_JOBLIB.exists():
        raise FileNotFoundError(f"Model joblib not found: {MODEL_JOBLIB}")
    if not MODEL_META_JSON.exists():
        raise FileNotFoundError(f"Metadata json not found: {MODEL_META_JSON}")

    pipeline = joblib.load(MODEL_JOBLIB)
    metadata = load_metadata_required(MODEL_META_JSON)
    feature_desc = load_feature_dictionary(DATA_DICTIONARY_XLSX)

    return pipeline, metadata, feature_desc


# Streamlit-friendly scoring entry point
def score_application(
    raw_input: Union[Dict[str, Any], pd.DataFrame],
    pipeline: Any,
    metadata: dict,
    feature_desc: Dict[str, str],
    use_gemini: bool = True,
    gemini_api_key: Optional[str] = None,
    top_k: int = 6,
) -> dict:
    expected_manip_features = list(metadata["training_features_manipulated_order"])
    raw_features = list(metadata["raw_input_features_order"])
    threshold = float(metadata["threshold_on_P(Bad)"])

    target_mapping = metadata["target_mapping"]
    label_0 = target_mapping.get("0", "Good")
    label_1 = target_mapping.get("1", "Bad")

    # normalize input -> 1-row df
    if isinstance(raw_input, dict):
        raw_one = pd.DataFrame([raw_input], columns=raw_features)
    else:
        if raw_input.shape[0] < 1:
            raise ValueError("Uploaded data has no rows.")
        missing = [c for c in raw_features if c not in raw_input.columns]
        if missing:
            raise ValueError(f"Missing required raw columns: {missing}")
        raw_one = raw_input.loc[[raw_input.index[0]], raw_features].copy()
        raw_one.index = [0]

    X_manip_one = raw_to_manipulated(
        raw_row=raw_one,
        expected_manip_features=expected_manip_features,
        drop_cols=DROP_COLS,
    )

    p_bad = float(pipeline.predict_proba(X_manip_one)[:, 1][0])
    y_pred = 1 if p_bad >= threshold else 0

    decision_text = (
        "Unfortunately, we are not able to approve your application at this time."
        if y_pred == 1
        else "Your application has received initial approval and will be forwarded to a loan officer for final review."
    )

    decision = "deny" if y_pred == 1 else "forward"

    result = {
        "p_bad": p_bad,
        "threshold": threshold,
        "y_pred": y_pred,
        "decision": decision,
        "decision_text": decision_text,
        "label_0": label_0,
        "label_1": label_1,
    }

    result["admin_diagnostics"] = build_admin_diagnostics(
        pipeline=pipeline,
        X_manip_one=X_manip_one,
        expected_manip_features=expected_manip_features,
        feature_desc=feature_desc,
        drop_cols=DROP_COLS,
        decision=decision,
        top_k=5,
    )

    if y_pred == 1:
        package = build_rejection_package(
            pipeline=pipeline,
            X_manip_one=X_manip_one,
            expected_manip_features=expected_manip_features,
            feature_desc=feature_desc,
            drop_cols=DROP_COLS,
            top_k=top_k,
        )
        result["rejection_package"] = package

        if use_gemini:
            result["gemini_explanation"] = call_gemini_flash(package, gemini_api_key=gemini_api_key)

    return result
