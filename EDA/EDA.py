# EDA helper: counts of special values (-7, -8, -9) by column,
# Creates manipulated dataset with engineered features.

import os
import pandas as pd

SPECIAL_CODES = [-7, -8, -9]
CSV_FILENAME = "heloc_dataset_v1.csv"


def main():
    
    # ----------------- EDA -------------------

    # Load CSV from the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILENAME)

    df = pd.read_csv(csv_path)
    original_cols = df.columns.tolist()

    # Identify numeric columns only (so non-numeric labels like "RiskPerformance" don't break row-level checks)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Build a count table: for each column, count occurrences of -7, -8, -9
    counts = pd.DataFrame(index=df.columns)
    for code in SPECIAL_CODES:
        counts[f"count_{code}"] = (df == code).sum(axis=0)

    # Add a total special count per column
    counts["count_special_total"] = counts[[f"count_{c}" for c in SPECIAL_CODES]].sum(axis=1)

    # Print the table (sorted by most special values)
    counts_sorted = counts.sort_values("count_special_total", ascending=False)
    print("\n=== Special value counts by column (-7, -8, -9) ===")
    print(counts_sorted.to_string())

    # Row-level: counts where ALL numeric columns are the SAME special code (-7 vs -8 vs -9)
    # and where ALL numeric columns are any special code (-7/-8/-9).
    if numeric_cols:
        print("\n=== Rows where ALL numeric fields are special (by code) ===")
        print(f"Numeric columns considered: {len(numeric_cols)}")
        print(f"Total rows: {len(df)}")

        # Helper to compute % Good in RiskPerformance for a given mask
        def pct_good(mask: pd.Series) -> float | None:
            if "RiskPerformance" not in df.columns:
                return None
            if mask.sum() == 0:
                return 0.0
            rp = df.loc[mask, "RiskPerformance"].astype(str).str.strip().str.lower()
            return float((rp == "good").mean() * 100)

        # Masks for each specific special code
        all_minus7_mask = (df[numeric_cols] == -7).all(axis=1)
        all_minus8_mask = (df[numeric_cols] == -8).all(axis=1)
        all_minus9_mask = (df[numeric_cols] == -9).all(axis=1)

        # Mask for any special code in every numeric column
        all_special_mask = df[numeric_cols].isin(SPECIAL_CODES).all(axis=1)

        # Print counts + % Good for each case
        for code, mask in [(-7, all_minus7_mask), (-8, all_minus8_mask), (-9, all_minus9_mask)]:
            n_rows = int(mask.sum())
            good_pct = pct_good(mask)
            print(f"\n-- All numeric fields == {code} --")
            print(f"Rows: {n_rows}")
            if good_pct is None:
                print("% RiskPerformance == Good: (RiskPerformance column not found)")
            else:
                print(f"% RiskPerformance == Good: {good_pct:.2f}%")

        # Also print the combined 'all special' case for completeness
        n_all_special_rows = int(all_special_mask.sum())
        good_pct_all_special = pct_good(all_special_mask)
        print("\n-- All numeric fields in {-7, -8, -9} (any mix) --")
        print(f"Rows: {n_all_special_rows}")
        if good_pct_all_special is None:
            print("% RiskPerformance == Good: (RiskPerformance column not found)")
        else:
            print(f"% RiskPerformance == Good: {good_pct_all_special:.2f}%")
    else:
        print("\nNo numeric columns found, so row-level special-code counts were not computed.")

    # Overall totals across the dataset 
    total_special = int((df.isin(SPECIAL_CODES)).sum().sum())
    print("\n=== Overall totals ===")
    print(f"Total special entries (-7/-8/-9) across entire dataframe: {total_special}")

    # -------- Manipulated File ----------------

    if numeric_cols:
        no_bureau_mask = (df[numeric_cols] == -9).all(axis=1)
        df["NoBureau"] = no_bureau_mask.astype(int)
    else:
        df["NoBureau"] = 0

    # Row-level counts of special codes across numeric fields
    # (CountMinus7 captures "condition not met" prevalence; CountMinus8 captures thin-file/invalid-trade prevalence)
    if numeric_cols:
        df["CountMinus7"] = (df[numeric_cols] == -7).sum(axis=1).astype(int)
        df["CountMinus8"] = (df[numeric_cols] == -8).sum(axis=1).astype(int)
    else:
        df["CountMinus7"] = 0
        df["CountMinus8"] = 0

    # Create targeted indicator columns for special codes (-7/-8) in selected features
    # NOTE: These indicators are created regardless of NoBureau; NoBureau rows will naturally have 0s here

    # Columns to add indicators (based on analysis of which columns contain special fields)
    indicator_specs = {
        "MSinceMostRecentDelq": [-7, -8],
        "NetFractionInstallBurden": [-8],
        "MSinceMostRecentInqexcl7days": [-7, -8],
        "NumInstallTradesWBalance": [-8],
        "NumBank2NatlTradesWHighUtilization": [-8],
        "MSinceOldestTradeOpen": [-8],
        "NetFractionRevolvingBurden": [-8],
        "NumRevolvingTradesWBalance": [-8],
        "PercentTradesWBalance": [-8],
    }

    for col, codes in indicator_specs.items():
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not found; skipping indicator creation.")
            continue
        for code in codes:
            suffix = "m7" if code == -7 else ("m8" if code == -8 else f"m{abs(code)}")
            ind_col = f"{col}_is_{suffix}"
            df[ind_col] = (df[col] == code).astype(int)

    # Replace special codes (-7, -8, -9) with NaN for modeling-friendly numeric features.
    # Keep engineered indicator columns and summary columns intact.
    engineered_keep_cols = {"NoBureau", "CountMinus7", "CountMinus8"}
    engineered_keep_cols.update([c for c in df.columns if c.endswith("_is_m7") or c.endswith("_is_m8")])

    # Apply replacement only to original numeric columns 
    numeric_original_cols = [c for c in numeric_cols if c in original_cols and c != "RiskPerformance" and c not in engineered_keep_cols]

    if numeric_original_cols:
        df[numeric_original_cols] = df[numeric_original_cols].replace({-7: pd.NA, -8: pd.NA, -9: pd.NA})

    # One-hot encode delinquency code fields (keep originals; add dummies for modeling)

    one_hot_cols = ["MaxDelqEver", "MaxDelq2PublicRecLast12M"]
    for col in one_hot_cols:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not found; skipping one-hot encoding.")
            continue
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True, dtype=int)
        # Avoid accidental overwrite if re-running in an interactive session
        dummies = dummies[[c for c in dummies.columns if c not in df.columns]]
        df = pd.concat([df, dummies], axis=1)

    # Drop original delinquency code columns (keep one-hot indicators)
    df = df.drop(columns=[c for c in one_hot_cols if c in df.columns])

    # Reorder columns so engineered features are easy to find in the output file
    engineered_cols = [c for c in df.columns if c not in original_cols]

    if "RiskPerformance" in df.columns:
        remaining_original = [c for c in original_cols if c != "RiskPerformance" and c in df.columns]
        df = df[["RiskPerformance"] + engineered_cols + remaining_original]
    else:
        remaining_original = [c for c in original_cols if c in df.columns]
        df = df[engineered_cols + remaining_original]

    print("\n=== Engineered columns added ===")
    print(engineered_cols)
    print(f"Final dataframe shape (rows, cols): {df.shape}")

    # Save manipulated dataset to the same folder with a clear name
    base_name, ext = os.path.splitext(CSV_FILENAME)
    out_filename = f"{base_name}_manipulated{ext if ext else '.csv'}"
    out_path = os.path.join(script_dir, out_filename)

    df.to_csv(out_path, index=False)

    print("\n=== Data manipulation complete ===")
    print(f"Added column: NoBureau (1 if all numeric fields are -9, else 0)")
    print(f"NoBureau == 1 count: {int(df['NoBureau'].sum())}")
    print(f"Wrote manipulated dataset to: {out_path}")


if __name__ == "__main__":
    main()
