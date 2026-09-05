import numpy as np
import pandas as pd

from data import (
    list_csv_files_in_local_folder,
    load_single_csv_from_local,
    standardize_column_names,
    standardize_keys,
)
from data_cleaning import clean_series_by_local_outlier_then_fill


# =========================================================
# 1. Basic Statistical and Temporal Features
# =========================================================

def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_valid_value(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan


def last_valid_value(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) > 0 else np.nan


def safe_quantile(s: pd.Series, q: float):
    s = s.dropna()
    return s.quantile(q) if len(s) > 0 else np.nan


def slope_feature(x: pd.Series) -> float:
    """Calculate slope against observation position, not elapsed time."""
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) < 2:
        return np.nan

    idx = np.arange(len(x))
    coef = np.polyfit(idx, x.values, 1)[0]

    return float(coef)


def lag1_autocorr_feature(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) < 3:
        return np.nan

    return x.autocorr(lag=1)


def first_half_mean(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan

    mid = max(1, len(x) // 2)

    return x.iloc[:mid].mean()


def second_half_mean(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan

    mid = max(1, len(x) // 2)

    return x.iloc[mid:].mean()


# =========================================================
# 2. Segment Features
# =========================================================

def segment_slices(n: int, segment_count: int = 3):
    """Build positional segments using the notebook's original rule."""
    if n <= 0:
        return []

    idx = np.linspace(0, n, segment_count + 1).astype(int)
    segments = []

    for i in range(segment_count):
        start = idx[i]
        end = idx[i + 1]

        if end <= start:
            end = min(n, start + 1)

        segments.append((start, end))

    return segments


def segment_mean(
    x: pd.Series,
    seg_idx: int,
    seg_count: int = 3,
):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan

    segs = segment_slices(len(x), seg_count)

    if seg_idx >= len(segs):
        return np.nan

    start, end = segs[seg_idx]

    return x.iloc[start:end].mean()


def segment_std(
    x: pd.Series,
    seg_idx: int,
    seg_count: int = 3,
):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan

    segs = segment_slices(len(x), seg_count)

    if seg_idx >= len(segs):
        return np.nan

    start, end = segs[seg_idx]

    return x.iloc[start:end].std()


def segment_median(
    x: pd.Series,
    seg_idx: int,
    seg_count: int = 3,
):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan

    segs = segment_slices(len(x), seg_count)

    if seg_idx >= len(segs):
        return np.nan

    start, end = segs[seg_idx]

    return x.iloc[start:end].median()


# =========================================================
# 3. Process-Specific Features
# =========================================================

def upper_tail_ratio(x: pd.Series, q: float = 0.90):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan

    threshold = x.quantile(q)

    return float((x >= threshold).mean())


def spike_count(x: pd.Series, z: float = 2.5):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) < 3:
        return np.nan

    median = x.median()
    mad = np.median(np.abs(x - median))

    if mad == 0 or pd.isna(mad):
        return 0

    robust_z = 0.6745 * (x - median) / mad

    return int((np.abs(robust_z) > z).sum())


def low_flow_ratio(x: pd.Series, q: float = 0.10):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan

    threshold = x.quantile(q)

    return float((x <= threshold).mean())


def stability_feature(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) < 2:
        return np.nan

    median = x.median()

    if pd.isna(median) or abs(median) < 1e-8:
        return np.nan

    return float(x.std() / abs(median))


# =========================================================
# 4. Feature Profiles and Statistical Plans
# =========================================================

def detect_feature_profile(
    col_name: str,
    s: pd.Series,
) -> str:
    name = str(col_name).upper()
    valid = pd.to_numeric(s, errors="coerce").dropna()

    if "ROTATION" in name or "RPM" in name:
        return "rotation"

    if "PRESSURE" in name:
        return "pressure"

    if "SLURRY_FLOW" in name or (
        "FLOW" in name and "SLURRY" in name
    ):
        return "slurry_flow"

    if any(
        keyword in name
        for keyword in ["USAGE", "COUNT", "CYCLE", "TIME", "DURATION"]
    ):
        return "usage_time"

    if len(valid) > 0 and valid.nunique() <= 6:
        return "quasi_static"

    return "general"


def get_stage_stat_plan(stage: str, profile: str) -> dict:
    """Return the original profile-based feature plan.

    The notebook uses the same plan for stages A and B.
    The stage argument is retained for compatibility.
    """
    stage = str(stage).upper()

    common_stats = [
        "median",
        "q10",
        "q25",
        "q75",
        "q90",
        "iqr",
        "robust_range",
        "valid_ratio",
        "missing_ratio",
    ]

    if profile == "rotation":
        return {
            "basic": common_stats,
            "trend": ["std", "slope"],
            "autocorr": True,
            "segment": True,
            "half": False,
            "custom": [],
        }

    if profile == "pressure":
        return {
            "basic": common_stats,
            "trend": ["std"],
            "autocorr": False,
            "segment": True,
            "half": False,
            "custom": ["upper_tail_ratio", "spike_count"],
        }

    if profile == "slurry_flow":
        return {
            "basic": common_stats,
            "trend": ["std", "slope"],
            "autocorr": False,
            "segment": True,
            "half": False,
            "custom": ["low_flow_ratio", "stability"],
        }

    if profile == "usage_time":
        return {
            "basic": [
                "median",
                "q25",
                "q75",
                "valid_ratio",
                "missing_ratio",
            ],
            "trend": ["first", "last", "delta", "slope"],
            "autocorr": False,
            "segment": False,
            "half": False,
            "custom": [],
        }

    if profile == "quasi_static":
        return {
            "basic": [
                "median",
                "q25",
                "q75",
                "valid_ratio",
                "missing_ratio",
            ],
            "trend": ["std", "zero_ratio", "nunique_ratio"],
            "autocorr": False,
            "segment": False,
            "half": False,
            "custom": [],
        }

    return {
        "basic": common_stats,
        "trend": ["std", "slope"],
        "autocorr": False,
        "segment": True,
        "half": False,
        "custom": [],
    }


# =========================================================
# 5. Single Wafer-Stage Compression
# =========================================================

def compress_one_group_best(
    group_df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
    segment_count: int = 3,
    local_iqr_k: float = 3.0,
) -> dict:

    result = {}

    wafer_id = group_df["WAFER_ID"].iloc[0]
    stage = str(group_df["STAGE"].iloc[0]).upper()

    result["WAFER_ID"] = wafer_id
    result["STAGE"] = stage
    result["N_TIMESTEPS"] = len(group_df)

    if "SOURCE_FILE" in group_df.columns:
        result["SOURCE_FILE_NUNIQUE"] = (
            group_df["SOURCE_FILE"].nunique()
        )

    if "TIMESTAMP" in group_df.columns:
        ts = pd.to_datetime(
            group_df["TIMESTAMP"],
            errors="coerce",
        )
        ts_valid = ts.dropna()

        result["DURATION_SEC"] = (
            (ts_valid.max() - ts_valid.min()).total_seconds()
            if len(ts_valid) > 0
            else np.nan
        )
        result["START_TIME_ORD"] = (
            ts_valid.min().value
            if len(ts_valid) > 0
            else np.nan
        )
    else:
        result["DURATION_SEC"] = np.nan
        result["START_TIME_ORD"] = np.nan

    exclude_cols = {
        "WAFER_ID",
        "STAGE",
        "TIMESTAMP",
        "SOURCE_FILE",
        "ROW_IN_FILE",
        target_col,
    }

    numeric_candidate_cols = []

    for col in group_df.columns:
        if col in exclude_cols:
            continue

        col_numeric = pd.to_numeric(
            group_df[col],
            errors="coerce",
        )

        if col_numeric.notna().sum() > 0:
            numeric_candidate_cols.append(col)

    for col in numeric_candidate_cols:
        raw_s = safe_numeric_series(group_df[col])
        s = clean_series_by_local_outlier_then_fill(
            raw_s,
            iqr_k=local_iqr_k,
        )
        valid_cleaned = s.dropna()

        profile = detect_feature_profile(col, raw_s)
        plan = get_stage_stat_plan(stage, profile)

        valid_ratio = raw_s.notna().mean()
        missing_ratio = raw_s.isna().mean()

        # Basic statistics
        if "median" in plan["basic"]:
            result[f"{col}_median"] = s.median()

        if "min" in plan["basic"]:
            result[f"{col}_min"] = s.min()

        if "max" in plan["basic"]:
            result[f"{col}_max"] = s.max()

        if "q25" in plan["basic"]:
            result[f"{col}_q25"] = safe_quantile(s, 0.25)

        if "q75" in plan["basic"]:
            result[f"{col}_q75"] = safe_quantile(s, 0.75)

        if "range" in plan["basic"]:
            result[f"{col}_range"] = (
                s.max() - s.min()
                if s.notna().sum() > 0
                else np.nan
            )

        if "valid_ratio" in plan["basic"]:
            result[f"{col}_valid_ratio"] = valid_ratio

        if "missing_ratio" in plan["basic"]:
            result[f"{col}_missing_ratio"] = missing_ratio

        if "q10" in plan["basic"]:
            result[f"{col}_q10"] = safe_quantile(s, 0.10)

        if "q90" in plan["basic"]:
            result[f"{col}_q90"] = safe_quantile(s, 0.90)

        if "iqr" in plan["basic"]:
            q25 = safe_quantile(s, 0.25)
            q75 = safe_quantile(s, 0.75)

            result[f"{col}_iqr"] = (
                q75 - q25
                if pd.notna(q25) and pd.notna(q75)
                else np.nan
            )

        if "robust_range" in plan["basic"]:
            q10 = safe_quantile(s, 0.10)
            q90 = safe_quantile(s, 0.90)

            result[f"{col}_robust_range"] = (
                q90 - q10
                if pd.notna(q10) and pd.notna(q90)
                else np.nan
            )

        # Trend and variation statistics
        if "first" in plan["trend"]:
            result[f"{col}_first"] = first_valid_value(s)

        if "last" in plan["trend"]:
            result[f"{col}_last"] = last_valid_value(s)

        if "delta" in plan["trend"]:
            result[f"{col}_delta"] = (
                last_valid_value(s) - first_valid_value(s)
                if s.notna().sum() > 0
                else np.nan
            )

        if "std" in plan["trend"]:
            result[f"{col}_std"] = s.std()

        if "slope" in plan["trend"]:
            result[f"{col}_slope"] = (
                slope_feature(s)
                if len(valid_cleaned) >= 3
                else np.nan
            )

        if "zero_ratio" in plan["trend"]:
            result[f"{col}_zero_ratio"] = (
                (valid_cleaned == 0).mean()
                if len(valid_cleaned) > 0
                else np.nan
            )

        if "nunique_ratio" in plan["trend"]:
            result[f"{col}_nunique_ratio"] = (
                valid_cleaned.nunique(dropna=True)
                / len(valid_cleaned)
                if len(valid_cleaned) > 0
                else np.nan
            )

        if plan["autocorr"]:
            result[f"{col}_lag1_autocorr"] = (
                lag1_autocorr_feature(s)
                if len(valid_cleaned) >= 3
                else np.nan
            )

        if plan["half"]:
            result[f"{col}_first_half_mean"] = first_half_mean(s)
            result[f"{col}_second_half_mean"] = second_half_mean(s)

        # Segment statistics
        if plan["segment"]:
            for seg_idx in range(segment_count):
                prefix = f"{col}_seg{seg_idx + 1}"

                result[f"{prefix}_mean"] = segment_mean(
                    s, seg_idx, segment_count
                )
                result[f"{prefix}_std"] = segment_std(
                    s, seg_idx, segment_count
                )

                if profile == "slurry_flow":
                    result[f"{prefix}_median"] = segment_median(
                        s, seg_idx, segment_count
                    )

        # Process-specific statistics
        if "upper_tail_ratio" in plan["custom"]:
            result[f"{col}_upper_tail_ratio"] = upper_tail_ratio(s)

        if "spike_count" in plan["custom"]:
            result[f"{col}_spike_count"] = spike_count(s)

        if "low_flow_ratio" in plan["custom"]:
            result[f"{col}_low_flow_ratio"] = low_flow_ratio(s)

        if "stability" in plan["custom"]:
            result[f"{col}_stability"] = stability_feature(s)

    return result


# =========================================================
# 6. Incremental Folder Compression
# =========================================================

def compress_timeseries_folder_incremental_best(
    folder_path,
    dataset_name: str = "Dataset",
    verbose: bool = False,
    max_files=None,
    target_col: str = "AVG_REMOVAL_RATE",
    segment_count: int = 3,
    local_iqr_k: float = 3.0,
    show_file_progress: bool = False,
    show_summary_progress: bool = True,
) -> pd.DataFrame:

    csv_files = list_csv_files_in_local_folder(
        folder_path,
        max_files=max_files,
    )

    key_cols = ["WAFER_ID", "STAGE"]
    compressed_rows = []
    skipped_empty_files = []
    skipped_no_key_files = []

    print(f"\n=== Incremental compression for {dataset_name} ===")
    print("CSV file count:", len(csv_files))

    for idx, file_info in enumerate(csv_files, start=1):
        file_name = file_info.name

        if verbose and show_file_progress:
            print(
                f"[{idx}/{len(csv_files)}] "
                f"Loading and compressing {file_name} ..."
            )
        elif show_summary_progress and (
            idx % 25 == 0 or idx == len(csv_files)
        ):
            print(f"Processed {idx}/{len(csv_files)} files")

        df = load_single_csv_from_local(file_info)
        df = standardize_column_names(df)

        if df.empty:
            skipped_empty_files.append(file_name)
            continue

        missing_keys = [
            key for key in key_cols
            if key not in df.columns
        ]

        if len(missing_keys) > 0:
            skipped_no_key_files.append((file_name, missing_keys))
            continue

        df = standardize_keys(df)
        df["SOURCE_FILE"] = file_name

        if "TIMESTAMP" in df.columns:
            df["TIMESTAMP"] = pd.to_datetime(
                df["TIMESTAMP"],
                errors="coerce",
            )
            df = df.sort_values(
                ["WAFER_ID", "STAGE", "TIMESTAMP"],
                kind="stable",
            )
        else:
            df = df.sort_values(
                ["WAFER_ID", "STAGE"],
                kind="stable",
            )

        grouped = df.groupby(
            ["WAFER_ID", "STAGE"],
            sort=False,
            dropna=False,
        )

        for _, group_df in grouped:
            compressed_rows.append(
                compress_one_group_best(
                    group_df,
                    target_col=target_col,
                    segment_count=segment_count,
                    local_iqr_k=local_iqr_k,
                )
            )

    compressed_df = pd.DataFrame(compressed_rows)

    print(
        f"{dataset_name} compressed raw shape:",
        compressed_df.shape,
    )
    print("Skipped empty files:", len(skipped_empty_files))
    print("Skipped files without keys:", len(skipped_no_key_files))

    return compressed_df


# =========================================================
# 7. Compatibility Helper
# =========================================================

def feature_engineering_compressed_best(
    df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
) -> pd.DataFrame:
    """Return a copy, matching the notebook's pass-through helper."""
    return df.copy()