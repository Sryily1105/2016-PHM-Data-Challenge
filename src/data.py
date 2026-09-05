from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. CSV Loading
# =========================================================

def list_csv_files_in_local_folder(folder_path, max_files=None):
    """Return CSV files sorted by filename, excluding subdirectories.

    Set max_files to None to include all files.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    csv_files = sorted(
        folder_path.glob("*.csv"),
        key=lambda path: path.name,
    )

    if max_files is not None:
        if not isinstance(max_files, int) or max_files < 0:
            raise ValueError("max_files must be a non-negative integer or None.")

        csv_files = csv_files[:max_files]

    return csv_files


def load_single_csv_from_local(file_path, **read_csv_kwargs):
    """Load a CSV file with optional pandas.read_csv arguments."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    return pd.read_csv(file_path, **read_csv_kwargs)


# =========================================================
# 2. Column and Key Standardization
# =========================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace and uppercase column names."""
    result = df.copy()
    result.columns = [
        str(column).strip().upper()
        for column in result.columns
    ]
    return result


def standardize_stage_value(value):
    """Convert a stage value to a stripped uppercase string."""
    if pd.isna(value):
        return np.nan

    return str(value).strip().upper()


def standardize_wafer_id(value):
    """Normalize wafer IDs, converting integer-like values to integer strings.

    For example, 123.0 and "123" both become "123".
    Other values are preserved as stripped strings.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    try:
        numeric_value = float(text)

        if numeric_value.is_integer():
            return str(int(numeric_value))
    except (ValueError, OverflowError):
        pass

    return text


def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize existing WAFER_ID and STAGE columns.

    Call standardize_column_names first to normalize column names.
    """
    result = df.copy()

    if "WAFER_ID" in result.columns:
        result["WAFER_ID"] = result["WAFER_ID"].apply(
            standardize_wafer_id
        )

    if "STAGE" in result.columns:
        result["STAGE"] = result["STAGE"].apply(
            standardize_stage_value
        )

    return result


# =========================================================
# 3. Label Preparation
# =========================================================

def prepare_label_df_strict(
    label_df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
    tol: float = 1e-8,
    drop_inconsistent: bool = True,
) -> pd.DataFrame:
    
    target_col = str(target_col).strip().upper()

    labels = standardize_column_names(label_df)
    labels = standardize_keys(labels)

    key_cols = ["WAFER_ID", "STAGE"]
    required_cols = key_cols + [target_col]

    missing_cols = [
        column
        for column in required_cols
        if column not in labels.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Required columns missing from label data: {missing_cols}"
        )

    labels = labels[required_cols].copy()
    labels[target_col] = pd.to_numeric(
        labels[target_col],
        errors="coerce",
    )
    labels = labels.dropna(subset=required_cols)

    grouped = (
        labels.groupby(key_cols)[target_col]
        .agg(["median", "min", "max"])
        .reset_index()
    )

    if drop_inconsistent:
        label_range = grouped["max"] - grouped["min"]
        consistent_mask = label_range.fillna(0) <= tol
        grouped = grouped.loc[consistent_mask]

    return (
        grouped[key_cols + ["median"]]
        .rename(columns={"median": target_col})
        .reset_index(drop=True)
    )


# =========================================================
# 4. Feature and Label Merging
# =========================================================

def merge_features_with_labels_by_keys(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    target_col: str = "AVG_REMOVAL_RATE",
    dataset_name: str = "Dataset",
) -> pd.DataFrame:

    target_col = str(target_col).strip().upper()
    key_cols = ["WAFER_ID", "STAGE"]

    for table_name, df, required_cols in [
        ("feature table", feature_df, key_cols),
        ("label table", label_df, key_cols + [target_col]),
    ]:
        missing_cols = [
            column
            for column in required_cols
            if column not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"{dataset_name}: required columns missing from "
                f"{table_name}: {missing_cols}"
            )

    if target_col in feature_df.columns:
        raise ValueError(
            f"{dataset_name}: the feature table already contains "
            f"{target_col}. Check whether labels have already been merged."
        )

    return pd.merge(
        feature_df.copy(),
        label_df.copy(),
        on=key_cols,
        how="left",
        validate="1:1",
    )