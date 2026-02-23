from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("heloc_dataset_v1_manipulated.csv")
OUTPUT_XLSX = Path("model_comparison_results.xlsx")

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50

DROP_COLS = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]

FAST_MODE = True 
CV_FOLDS = 3 if FAST_MODE else 5
CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

SVM_N_ITER = 30 if FAST_MODE else 60
ANN_N_ITER = 30 if FAST_MODE else 30

# If scipy is not installed, set USE_SCIPY_DISTS=False to use fixed candidate lists instead
USE_SCIPY_DISTS = True


# -----------------------------
# Helpers
# -----------------------------
def get_feature_target(df: pd.DataFrame, target_col: str = "RiskPerformance"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=[target_col] + drop_cols).copy()

    # Force consistent mapping for threshold logic:
    # y = 1 means Bad, y = 0 means Good
    y_raw = df[target_col].copy().astype(str)
    y = (y_raw.str.strip().str.lower() == "bad").astype(int).to_numpy()

    return X, y


def make_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    numeric_features = list(X.columns)

    if scale:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def proba_or_score(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model has neither predict_proba nor decision_function.")


def eval_on_test(best_estimator, X_test, y_test, threshold: float = DEFAULT_THRESHOLD) -> dict:
    p_bad = proba_or_score(best_estimator, X_test)  # this is P(Bad=1)
    y_pred = (p_bad >= threshold).astype(int)

    out = {}
    out["ROC_AUC"] = roc_auc_score(y_test, p_bad)
    out["AvgPrecision_PR_AUC"] = average_precision_score(y_test, p_bad)
    out["Accuracy"] = accuracy_score(y_test, y_pred)
    out["Precision"] = precision_score(y_test, y_pred, zero_division=0)
    out["Recall"] = recall_score(y_test, y_pred, zero_division=0)
    out["F1"] = f1_score(y_test, y_pred, zero_division=0)
    out["LogLoss"] = log_loss(y_test, np.vstack([1 - p_bad, p_bad]).T, labels=[0, 1])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    out["TN"] = tn
    out["FP"] = fp
    out["FN"] = fn
    out["TP"] = tp
    return out


def top_cv_rows(search, model_name: str, top_n: int = 10) -> pd.DataFrame:
    cv_res = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False).head(top_n)
    keep_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
    ]
    param_cols = [c for c in cv_res.columns if c.startswith("param_")]
    out = cv_res[keep_cols + param_cols].copy()
    out.insert(0, "Model", model_name)
    return out


# -----------------------------
# Model search builders
# -----------------------------
def run_logistic_regression(X_train, y_train, X_template):
    preprocessor = make_preprocessor(X_template, scale=True)

    clf = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
        class_weight="balanced",
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    param_grid = {
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 1, 10],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs


def run_decision_tree(X_train, y_train, X_template):
    preprocessor = make_preprocessor(X_template, scale=False)

    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    param_grid = {
        "clf__max_depth": [None, 4, 8],
        "clf__min_samples_leaf": [1, 10, 30],
        "clf__ccp_alpha": [0.0, 0.001],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs


def run_random_forest(X_train, y_train, X_template):
    preprocessor = make_preprocessor(X_template, scale=False)

    clf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    param_grid = {
        "clf__n_estimators": [300],
        "clf__max_depth": [None, 12],
        "clf__min_samples_leaf": [1, 10],
        "clf__max_features": ["sqrt", 0.5],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs


def run_svm(X_train, y_train, X_template):
    preprocessor = make_preprocessor(X_template, scale=True)

    base_clf = SVC(
        probability=False,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", base_clf)])

    if USE_SCIPY_DISTS:
        from scipy.stats import loguniform

        param_dist = {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": loguniform(1e-2, 1e2),
            "clf__gamma": ["scale", "auto"],
        }
    else:
        param_dist = {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto"],
        }

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=SVM_N_ITER,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    rs.fit(X_train, y_train)

    best_params = rs.best_params_

    final_clf = SVC(
        probability=True,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        kernel=best_params["clf__kernel"],
        C=best_params["clf__C"],
        gamma=best_params.get("clf__gamma", "scale"),
    )
    final_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", final_clf)])
    final_pipe.fit(X_train, y_train)

    class _Wrap:
        best_estimator_ = final_pipe
        best_params_ = best_params
        best_score_ = rs.best_score_
        cv_results_ = rs.cv_results_

    return _Wrap()


def run_ann(X_train, y_train, X_template):
    preprocessor = make_preprocessor(X_template, scale=True)

    clf = MLPClassifier(
        random_state=RANDOM_STATE,
        max_iter=1000 if FAST_MODE else 2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    if USE_SCIPY_DISTS:
        from scipy.stats import loguniform

        param_dist = {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "clf__alpha": loguniform(1e-5, 1e-2),
            "clf__learning_rate_init": loguniform(1e-4, 1e-2),
        }
    else:
        param_dist = {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        }

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=ANN_N_ITER,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    rs.fit(X_train, y_train)
    return rs


# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(DATA_PATH)
    X, y = get_feature_target(df, target_col="RiskPerformance")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    runners = {
        "LogisticRegression": run_logistic_regression,
        "DecisionTree": run_decision_tree,
        "RandomForest": run_random_forest,
        "SVM": run_svm,
        "ANN": run_ann,
    }

    results_rows = []
    best_params_rows = []
    cv_rows = []

    for model_name, runner in runners.items():
        search = runner(X_train, y_train, X_template=X_train)

        best_est = search.best_estimator_
        best_params = search.best_params_
        best_cv_auc = search.best_score_

        test_metrics = eval_on_test(best_est, X_test, y_test, threshold=DEFAULT_THRESHOLD)

        row = {"Model": model_name, "CV_Folds": CV_FOLDS, "BestCV_ROC_AUC": best_cv_auc}
        row.update(test_metrics)
        results_rows.append(row)

        bp = {"Model": model_name}
        bp.update(best_params)
        best_params_rows.append(bp)

        cv_rows.append(top_cv_rows(search, model_name=model_name, top_n=10))

    results_df = pd.DataFrame(results_rows).sort_values("ROC_AUC", ascending=False)
    best_params_df = pd.DataFrame(best_params_rows)
    cv_top_df = pd.concat(cv_rows, axis=0, ignore_index=True)
    features_df = pd.DataFrame({"Feature": list(X.columns)})

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Model_Comparison", index=False)
        best_params_df.to_excel(writer, sheet_name="Best_Params", index=False)
        cv_top_df.to_excel(writer, sheet_name="CV_Top", index=False)
        features_df.to_excel(writer, sheet_name="Features", index=False)

    print(f"Saved results to: {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
