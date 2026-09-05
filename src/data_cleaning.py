import numpy as np
import pandas as pd


# =========================================================
# 1. Local Time-Series Cleaning
# =========================================================

def clean_series_by_local_outlier_then_fill(
    s: pd.Series,
    iqr_k: float = 3.0,
) -> pd.Series:
    """Mask local IQR outliers, then interpolate and fill missing values."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    valid = s.dropna()

    if len(valid) >= 5:
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        if pd.notna(iqr) and iqr > 0:
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
            s = s.mask((s < lower) | (s > upper))

    s = s.interpolate(method="linear", limit_direction="both")
    s = s.ffill().bfill()

    return s


# =========================================================
# 2. Duplicate Key Consolidation
# =========================================================

def consolidate_duplicate_compressed_keys(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    target_col: str = "AVG_REMOVAL_RATE",
) -> pd.DataFrame:

    df = df.copy()
    key_cols = ["WAFER_ID", "STAGE"]

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    unique_dup_keys = int(
        df.loc[dup_mask, key_cols].drop_duplicates().shape[0]
    )

    print(f"\n=== Consolidating duplicate keys for {dataset_name} ===")
    print("Rows before consolidation:", len(df))
    print("Duplicate key count:", unique_dup_keys)

    if unique_dup_keys == 0:
        print("No duplicate keys found.")
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and col not in key_cols
    ]

    agg_dict = {}

    for col in numeric_cols:
        if col == target_col:
            continue

        if col == "SOURCE_FILE_NUNIQUE":
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "mean"

    for col in non_numeric_cols:
        agg_dict[col] = "first"

    consolidated = (
        df.groupby(key_cols, as_index=False)
        .agg(agg_dict)
    )

    print("Rows after consolidation:", len(consolidated))

    return consolidated


# =========================================================
# 3. Fixed Target-Range Filtering
# =========================================================

def get_drop_keys_by_target_range(
    label_df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
    lower: float = 0,
    upper: float = 200,
):

    target = pd.to_numeric(label_df[target_col], errors="coerce")
    mask = ~target.between(lower, upper)

    drop_df = label_df.loc[
        mask,
        ["WAFER_ID", "STAGE", target_col],
    ].copy()

    drop_keys = (
        drop_df[["WAFER_ID", "STAGE"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return drop_df.reset_index(drop=True), drop_keys


def drop_rows_by_keys(
    df: pd.DataFrame,
    drop_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows matching the supplied WAFER_ID and STAGE pairs."""
    if len(drop_keys) == 0:
        return df.copy().reset_index(drop=True)

    out = df.merge(
        drop_keys.assign(_drop_flag=1),
        on=["WAFER_ID", "STAGE"],
        how="left",
    )

    out = (
        out.loc[out["_drop_flag"].isna()]
        .drop(columns="_drop_flag")
    )

    return out.reset_index(drop=True)


# =========================================================
# 4. Target IQR Outlier Handling
# =========================================================

def compute_iqr_bounds(y: pd.Series, k: float = 1.5):
    """Return lower bound, upper bound, Q1, Q3, and IQR."""
    y = pd.to_numeric(y, errors="coerce").dropna()

    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper, q1, q3, iqr


def detect_target_outliers_iqr(
    df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
    k: float = 10.0,
):

    y = pd.to_numeric(df[target_col], errors="coerce").dropna()

    lower_iqr, upper_iqr, q1, q3, iqr = compute_iqr_bounds(y, k=k)

    lower_quantile = y.quantile(0.001)
    upper_quantile = y.quantile(0.999)

    lower = min(lower_iqr, lower_quantile)
    upper = max(upper_iqr, upper_quantile)

    y_full = pd.to_numeric(df[target_col], errors="coerce")
    mask = (y_full < lower) | (y_full > upper)

    print("\n=== Target Extreme Outlier Detection ===")
    print(f"Q1                : {q1:.6f}")
    print(f"Q3                : {q3:.6f}")
    print(f"IQR               : {iqr:.6f}")
    print(f"Lower IQR bound   : {lower_iqr:.6f}")
    print(f"Upper IQR bound   : {upper_iqr:.6f}")
    print(f"Lower final bound : {lower:.6f}")
    print(f"Upper final bound : {upper:.6f}")
    print(f"Outlier count     : {int(mask.sum())} / {len(df)}")

    return mask, lower, upper


def remove_target_outliers_iqr(
    df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
    k: float = 10.0,
):
    """Return retained rows, rejected rows, mask, and target bounds."""
    mask, lower, upper = detect_target_outliers_iqr(
        df,
        target_col=target_col,
        k=k,
    )

    df_clean = df.loc[~mask].reset_index(drop=True)
    df_outliers = df.loc[mask].reset_index(drop=True)

    return df_clean, df_outliers, mask, lower, upper


def flag_target_outliers_by_bounds(
    df: pd.DataFrame,
    lower: float,
    upper: float,
    target_col: str = "AVG_REMOVAL_RATE",
    flag_col: str = "TARGET_OUTLIER_FLAG",
):

    out = df.copy()
    y = pd.to_numeric(out[target_col], errors="coerce")

    mask = ((y < lower) | (y > upper)).fillna(False)
    out[flag_col] = mask.astype(int)

    return out.reset_index(drop=True), mask


# =========================================================
# 5. Feature-Level Cleaning
# =========================================================

def fit_preprocessor_from_train_best(
    train_df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
    iqr_clip_multiplier: float = 4.0,
):

    df = train_df.copy()

    if "STAGE" in df.columns:
        df["STAGE"] = (
            df["STAGE"].astype(str).str.strip().str.upper()
        )

    protected_non_numeric_cols = ["WAFER_ID", "STAGE"]

    for col in df.columns:
        if col not in protected_non_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != target_col
    ]

    stats = {}

    for col in feature_numeric_cols:
        s = df[col].dropna()

        if len(s) < 5:
            stats[col] = {"lower": np.nan, "upper": np.nan}
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            stats[col] = {"lower": np.nan, "upper": np.nan}
        else:
            stats[col] = {
                "lower": q1 - iqr_clip_multiplier * iqr,
                "upper": q3 + iqr_clip_multiplier * iqr,
            }

    return {
        "feature_numeric_cols": feature_numeric_cols,
        "stats": stats,
    }


def apply_preprocessor_best(
    df: pd.DataFrame,
    prep: dict,
    target_col: str = "AVG_REMOVAL_RATE",
) -> pd.DataFrame:

    out = df.copy()

    if "STAGE" in out.columns:
        out["STAGE"] = (
            out["STAGE"].astype(str).str.strip().str.upper()
        )

    protected_non_numeric_cols = ["WAFER_ID", "STAGE"]

    for col in out.columns:
        if col not in protected_non_numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    feature_numeric_cols = prep["feature_numeric_cols"]

    for col in feature_numeric_cols:
        if col not in out.columns:
            out[col] = np.nan

    for col in feature_numeric_cols:
        col_series = pd.to_numeric(out[col], errors="coerce")
        lower = prep["stats"][col]["lower"]
        upper = prep["stats"][col]["upper"]

        cleaned = col_series.copy()

        if pd.notna(lower) and pd.notna(upper):
            cleaned = cleaned.mask(
                (cleaned < lower) | (cleaned > upper)
            )

        cleaned = cleaned.interpolate(
            method="linear",
            limit_direction="both",
        )
        cleaned = cleaned.ffill().bfill()

        out[col] = cleaned

    return out.copy()