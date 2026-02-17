# app.py
from __future__ import annotations
from pathlib import Path

import pandas as pd
import streamlit as st

from backend import load_artifacts, score_application

st.set_page_config(page_title="HELOC DSS", layout="wide")

st.title("HELOC Automated Application Portal")
# Make download buttons look like hyperlinks
st.markdown(
    """
    <style>
      .stDownloadButton > button {
        background: none !important;
        border: none !important;
        padding: 0 !important;
        color: #1f77b4 !important;
        text-decoration: underline !important;
        font-weight: 400 !important;
      }
      .stDownloadButton > button:hover {
        color: #0f5c8c !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Cache model artifacts
# ------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()

pipeline, metadata, feature_desc = get_artifacts()
raw_fields = list(metadata["raw_input_features_order"])

# Constant decision threshold from metadata (used by backend)
DEFAULT_THRESHOLD = metadata.get("threshold_on_P(Bad)")

# Gemini key from Streamlit secrets (preferred) or env var fallback
gemini_api_key = None
if "GEMINI_API_KEY" in st.secrets:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]

# ------------------------------------------------------------
# Admin / Reviewer sidebar (for grading / diagnostics)
# ------------------------------------------------------------
st.sidebar.title("Admin / Reviewer Panel")
st.sidebar.caption(
    "Internal diagnostic information. "
    "Not visible to applicants."
)

# Threshold / cutoff (populated after scoring)
_admin_threshold_slot = st.sidebar.empty()
if DEFAULT_THRESHOLD is not None:
    _admin_threshold_slot.metric("Threshold / Cutoff", f"{float(DEFAULT_THRESHOLD):.4f}")
else:
    _admin_threshold_slot.metric("Threshold / Cutoff", "—")

_admin_results_section = st.sidebar.empty()

_admin_diagnostics_section = st.sidebar.empty()

# ------------------------------------------------------------
# Main-page input method selector (no sidebar)
# ------------------------------------------------------------
st.subheader("Provide applicant data")
mode = st.radio(
    "Input method",
    ["Upload CSV/XLSX", "Manual entry"],
    index=0,
    horizontal=True,
)

if not gemini_api_key:
    st.warning(
        "Explanations are currently unavailable because the application is missing its explanation service key. "
        "You can still score applications."
    )


# ------------------------------------------------------------
# Input builders
# ------------------------------------------------------------
def manual_entry_ui() -> dict:
    st.subheader("Enter your credit information manually")
    st.caption("Enter raw values exactly as the template expects. Special codes like -7, -8, -9 are allowed.")

    # Curated groups for a consumer-friendly manual entry experience
    FIELD_GROUPS: list[tuple[str, str, list[str], bool]] = [
        (
            "Credit score / overall risk",
            "Your overall credit risk estimate.",
            ["ExternalRiskEstimate"],
            True,
        ),
        (
            "Credit history length / age of file",
            "How long you’ve had credit accounts and how recently they’ve been opened.",
            [
                "MSinceOldestTradeOpen",
                "MSinceMostRecentTradeOpen",
                "AverageMInFile",
            ],
            False,
        ),
        (
            "Delinquencies + public records",
            "Late payments, derogatory events, and severity of delinquencies.",
            [
                "NumTrades60Ever2DerogPubRec",
                "NumTrades90Ever2DerogPubRec",
                "MSinceMostRecentDelq",
                "MaxDelq2PublicRecLast12M",
                "MaxDelqEver",
                "PercentTradesNeverDelq",
            ],
            False,
        ),
        (
            "Credit activity / inquiries",
            "How often you’ve recently applied for credit.",
            [
                "MSinceMostRecentInqexcl7days",
                "NumInqLast6M",
                "NumInqLast6Mexcl7days",
            ],
            False,
        ),
        (
            "Accounts / trade counts and mix",
            "How many accounts you have and what types of credit they represent.",
            [
                "NumTotalTrades",
                "NumTradesOpeninLast12M",
                "NumSatisfactoryTrades",
                "PercentInstallTrades",
            ],
            False,
        ),
        (
            "Utilization / balances",
            "How much of your available credit you’re using and how many accounts carry balances.",
            [
                "NetFractionRevolvingBurden",
                "NetFractionInstallBurden",
                "NumRevolvingTradesWBalance",
                "NumInstallTradesWBalance",
                "NumBank2NatlTradesWHighUtilization",
                "PercentTradesWBalance",
            ],
            False,
        ),
    ]

    raw: dict[str, float] = {}

    # Track which fields we render so we can safely catch any extras
    rendered: set[str] = set()

    for title, blurb, fields, expanded in FIELD_GROUPS:
        # Only keep fields that actually exist in the model metadata
        fields_in_model = [f for f in fields if f in raw_fields]
        if not fields_in_model:
            continue

        with st.expander(f"{title} — {blurb}", expanded=expanded):
            cols = st.columns(3)
            for i, f in enumerate(fields_in_model):
                rendered.add(f)
                with cols[i % 3]:
                    raw[f] = st.number_input(
                        f,
                        value=int(st.session_state.get(f"man_{f}", 0.0)),
                        step=1,
                        format="%d",
                        key=f"man_{f}",
                        help=(feature_desc.get(f) if isinstance(feature_desc, dict) else None),
                    )

    # If there are any fields in metadata that aren’t covered above, show them in an "Other" section
    remaining = [f for f in raw_fields if f not in rendered]
    if remaining:
        with st.expander("Other / advanced — Additional fields used by the model", expanded=False):
            cols = st.columns(3)
            for i, f in enumerate(remaining):
                with cols[i % 3]:
                    raw[f] = st.number_input(
                        f,
                        value=int(st.session_state.get(f"man_{f}", 0.0)),
                        step=1,
                        format="%d",
                        key=f"man_{f}",
                        help=(feature_desc.get(f) if isinstance(feature_desc, dict) else None),
                    )

    return raw


def upload_ui() -> pd.DataFrame | None:
    st.subheader("Upload an Excel or CSV file containing your credit information")
    # Allow users to download the Excel template and an example filled template
    template_path = Path(__file__).resolve().parent / "raw_input_template.xlsx"
    example_path = Path(__file__).resolve().parent / "raw_input_filled.xlsx"

    c_dl1, c_dl2, _spacer = st.columns([3, 3, 7])

    with c_dl1:
        if template_path.exists():
            template_bytes = template_path.read_bytes()
            st.download_button(
                label="Download template file",
                data=template_bytes,
                file_name="raw_input_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Template file is not available right now.")

    with c_dl2:
        if example_path.exists():
            example_bytes = example_path.read_bytes()
            st.download_button(
                label="Download example template",
                data=example_bytes,
                file_name="raw_input_filled.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Example file is not available right now.")
    st.caption("Download the template above and fill in your credit information.")
    up = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if not up:
        return None

    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up)
    else:
        df = pd.read_excel(up)

    st.write("Preview:")
    st.dataframe(df.head())

    return df


# ------------------------------------------------------------
# Collect input
# ------------------------------------------------------------
raw_input_obj = None
if mode == "Manual entry":
    raw_input_obj = manual_entry_ui()
else:
    raw_input_obj = upload_ui()

st.divider()

# ------------------------------------------------------------
# Score
# ------------------------------------------------------------
if st.button("Score Application", type="primary", disabled=(raw_input_obj is None)):
    # Temporary progress UI (only visible while scoring runs)
    _progress_slot = st.empty()
    _progress = _progress_slot.progress(0.5)
    try:
        result = score_application(
            raw_input=raw_input_obj,
            pipeline=pipeline,
            metadata=metadata,
            feature_desc=feature_desc,
            use_gemini=bool(gemini_api_key),
            gemini_api_key=gemini_api_key,
        )
        _progress.progress(1.0)

        p_bad = result["p_bad"]
        label_1 = result["label_1"]
        with _admin_results_section.container():
            st.subheader("Application Results")
            st.metric(f"P({label_1})", f"{p_bad:.4f}")
            st.metric("Decision", "DENY" if result["decision"] == "deny" else "FORWARD")
            st.divider()

        # Admin diagnostics (special codes + top contributors)
        admin_diag = result.get("admin_diagnostics") or {}
        special = admin_diag.get("special_codes") or {}
        contributors = admin_diag.get("top_contributors") or []

        with _admin_diagnostics_section.container():
            # Special codes summary
            st.subheader("Special Codes")
            if special.get("detected"):
                nb = "Yes" if int(special.get("NoBureau", 0)) else "No"
                m7 = int(special.get("CountMinus7", 0))
                m8 = int(special.get("CountMinus8", 0))
                st.markdown(
                    f"""
                    <div style='line-height:1.2; margin:0; padding:0;'>
                      <div style='margin:0; padding:0;'>NoBureau: {nb}</div>
                      <div style='margin:0; padding:0;'>-7 codes: {m7}</div>
                      <div style='margin:0; padding:0;'>-8 codes: {m8}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.write("No special codes detected")

            st.divider()
            # Top Contributors
            st.subheader("Top Contributors")
            if contributors:
                # Scale bars by absolute contribution for a simple visual cue
                abs_vals = [abs(float(c.get("contribution_to_logit", 0.0))) for c in contributors]
                max_abs = max(abs_vals) if abs_vals else 0.0

                for i, c in enumerate(contributors, start=1):
                    fname = str(c.get("feature", ""))
                    desc = str(c.get("description", ""))
                    contrib = float(c.get("contribution_to_logit", 0.0))
                    direction = str(c.get("direction", ""))

                    risk_text = "Increased risk" if direction == "toward_bad" else "Reduced risk"

                    desc_line = (
                        f"<div style='margin:0; padding:0; line-height:1.2; color: rgba(49, 51, 63, 0.8); font-size: 0.85rem;'>{desc}</div>"
                        if desc else ""
                    )
                    risk_line = (
                        f"<div style='margin:0; padding:0; line-height:1.2; color: rgba(49, 51, 63, 0.8); font-size: 0.85rem;'>{risk_text}</div>"
                    )
                    contrib_line = (
                        f"<div style='margin:0; padding:0; line-height:1.2; color: rgba(49, 51, 63, 0.8); font-size: 0.85rem;'>Contribution: {contrib:+.4f}</div>"
                    )

                    st.markdown(
                        f"""
                        <div style='margin-bottom: 0.6rem; padding-left: 0.5rem; border-left: 3px solid rgba(49, 51, 63, 0.20);'>
                          <div style='margin:0; padding:0; line-height:1.2; font-weight:600; color: rgba(49, 51, 63, 0.70);'>{fname}</div>
                          {desc_line}
                          {risk_line}
                          {contrib_line}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            else:
                st.write("Contributor details unavailable.")

        threshold = result["threshold"]
        _admin_threshold_slot.metric("Threshold / Cutoff", f"{threshold:.4f}")
        decision = result["decision"]
        label_1 = result["label_1"]

        c1, _spacer = st.columns([1, 2])
        c1.metric("Decision", "DENY" if decision == "deny" else "FORWARD")

        if decision == "deny":
            st.error(result["decision_text"])
        else:
            st.success(result["decision_text"])

        # If denied, show explanations
        if decision == "deny":
            gem_text = result.get("gemini_explanation")
            st.subheader("Explanation of Denial")
            if gem_text:
                st.write(gem_text)
            else:
                st.warning("We couldn’t generate an explanation right now. Please try again.")

    except Exception as e:
        st.exception(e)
    finally:
        _progress_slot.empty()
