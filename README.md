# HELOC Automated Application Portal (Streamlit)

A Streamlit web app that scores a HELOC application using a trained **scikit-learn pipeline** (logistic regression classifier) and, when enabled, generates an applicant-friendly denial explanation using **Gemini**.

## What the app does

### Input methods (main page)
The user chooses one of two workflows:

1) **Upload CSV/XLSX (default)**
- Users can download two Excel files directly in the app:
  - `raw_input_template.xlsx` (blank template)
  - `raw_input_filled.xlsx` (example filled template)
- The user can fill out their credit information in the template file and re-upload the completed file to the application.

2) **Manual entry**
- The app shows grouped input sections (expanders) and collects the raw fields expected by the model.
- Inputs are configured as whole-number numeric fields (`step=1`, `format="%d"`).
- Special credit codes like `-7`, `-8`, and `-9` are allowed.

### Scoring (main page)
- Clicking **Score Application** runs the model and shows a progress indicator while scoring.
- The main (customer-facing) page shows:
  - **Decision**: `FORWARD` or `DENY`
  - A decision message (success/error styling)
  - If denied, **Explanation of Denial** (Gemini output if available)

### Admin / Reviewer sidebar
The sidebar is always present and shows:
- **Threshold / Cutoff** (loaded from model metadata at app start)
- After scoring:
  - **P(Bad)** (model probability for the positive class)
  - **Decision**
  - **Special Codes** summary (NoBureau / -7 / -8 counts)
  - **Top Contributors** list (per-feature logistic regression contribution diagnostics)

## Model and decisioning

### Model type
- The model is a **logistic regression classifier** with L2 regularization (C=1), wrapped in a scikit-learn **pipeline**.
- The hyperparameter C was tuned using cross validation, and logistic regression was chosen over SVM, Random Forest and Neural Networks due to its comparable performance and superior explainability. 
- The model produces `P(Bad)` via `predict_proba(...)` (probability of the positive class).

### Decision rule
- A constant threshold is read from metadata:
  - `metadata["threshold_on_P(Bad)"]`
- Decision logic:
  - `DENY` if `P(Bad) >= threshold`
  - `FORWARD` otherwise
  - The current threshold is set to 0.7750 in order to reduce false positives (denials of qualified borrowers).
  - Given that loan officers will manually approve accepted applications, the model is skewed to only deny applicants if they are extremely unqualified, leading to a false positive rate of 5%.

## Data Cleaning & Feature Engineering (Special Codes + Delinquency Codes)
Data cleaning was done by explicitly handling credit-bureau **special codes** `-7`, `-8`, and `-9`, which represent non-standard meanings (e.g., no credit file available or no relevant trades) rather than true numeric values. The dataset was scanned to quantify where these codes occurred and to understand their patterns; notably, rows containing `-9` were found to have `-9` across all numeric bureau fields, indicating **no credit history**, and were captured with a `NoBureau` flag. Because `-7` and `-8` appeared only in specific columns, borrower-level summary features (`CountMinus7`, `CountMinus8`) were created along with targeted indicator variables (e.g., `FEATURE_is_m7`, `FEATURE_is_m8`) for selected high-impact fields. After engineering these indicators, special-code values in the original numeric features were replaced with `NaN` so that modeling steps (e.g., imputation) could treat them as missing rather than ordinal values. Finally, delinquency-code fields (`MaxDelqEver`, `MaxDelq2PublicRecLast12M`) were one-hot encoded and the original code columns were dropped, since their integer codes are categorical/non-linear (e.g., the largest code represents “all other”) and should not be modeled as monotonic numeric quantities.

## Explanations and diagnostics

### Applicant-facing denial explanation (Gemini)
- If a Gemini API key is configured, the backend calls Gemini to produce an applicant-friendly denial explanation.
- When the model returns `DENY`, the app sends Gemini a structured explanation_package containing the applicant’s processed inputs plus the model-derived denial drivers (feature contributions) and required output/constraints so Gemini can write a borrower-friendly denial explanation.
- If Gemini does not return an explanation, the app shows a friendly warning message.

### Admin diagnostics returned by the backend
The backend returns an `admin_diagnostics` object in the scoring result, including:

1) **Special codes**
- `NoBureau` (Yes/No)
- `CountMinus7` (integer)
- `CountMinus8` (integer)
- If none are triggered, the UI shows: **"No special codes detected"**

2) **Top contributors (logistic regression contribution analysis)**
- The backend computes per-feature contributions in logit space:
  - `contribution_to_logit = coef * preprocessed_value`
- The returned list is aligned to the final decision:
  - If `DENY`: top contributors **toward Bad**
  - If `FORWARD`: top contributors **toward Good**
- The sidebar displays each contributor with:
  - feature name
  - description (if available)
  - “Increased risk” / “Reduced risk”
  - contribution value

## Key files

- `HELOCModel/app.py`
  - Streamlit UI (input workflows, downloads, scoring button, sidebar rendering)
- `HELOCModel/backend.py`
  - Artifact loading (`load_artifacts`)
  - Scoring (`score_application`)
  - Admin diagnostics packaging (`admin_diagnostics`)
  - Gemini call for denial explanation
- `HELOCModel/HELOC Model.py`
  - Loads saved logit model and transforms an applicant's scores to determine the application decision
- `HELOCModel/Log Model Output.py`
  - Original model training and metadata aggragation
- `HELOCModel/raw_input_template.xlsx`
  - Downloadable template used for uploads
- `HELOCModel/raw_input_filled.xlsx`
  - Downloadable example template used for uploads

## Configuration (Gemini)

### Streamlit Secrets
Set your Gemini key in Streamlit secrets as:

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
