import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# =========================================================
# 1. EDA Utilities
# =========================================================

def safe_corr(x, y):
    """Calculate Pearson correlation after removing missing pairs."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(tmp) < 2:
        return np.nan

    if (
        tmp["x"].std(ddof=0) == 0
        or tmp["y"].std(ddof=0) == 0
    ):
        return np.nan

    return float(tmp["x"].corr(tmp["y"]))


def pick_cols_by_keywords(cols, keywords):
    """Select column names containing any supplied keyword."""
    out = []

    for col in cols:
        name = str(col).upper()

        if any(keyword in name for keyword in keywords):
            out.append(col)

    return out


def prepare_eda_tables(
    train_merged_clean,
    target_col="AVG_REMOVAL_RATE",
):
    """Prepare the original CMP EDA data and correlation tables."""
    eda_df = train_merged_clean.copy()

    eda_df["STAGE"] = (
        eda_df["STAGE"].astype(str).str.strip().str.upper()
    )
    eda_df[target_col] = pd.to_numeric(
        eda_df[target_col],
        errors="coerce",
    )

    numeric_cols = (
        eda_df.select_dtypes(include=[np.number])
        .columns.tolist()
    )
    numeric_feature_cols = [
        col for col in numeric_cols
        if col != target_col
    ]

    process_keywords = [
        "PRESSURE",
        "ROTATION",
        "FLOW",
        "SLURRY",
        "RIPPLE",
        "CENTER_AIR_BAG",
        "EDGE_AIR_BAG",
        "MAIN_OUTER_AIR_BAG",
        "RETAINER_RING",
        "CHAMBER",
    ]
    consumable_keywords = [
        "USAGE",
        "DRESSER",
        "POLISHING_TABLE",
        "BACKING_FILM",
        "MEMBRANE",
        "PRESSURIZED_SHEET",
    ]

    process_cols = pick_cols_by_keywords(
        numeric_feature_cols,
        process_keywords,
    )
    consumable_cols = pick_cols_by_keywords(
        numeric_feature_cols,
        consumable_keywords,
    )

    corr_rows = []

    for feature_type, columns in [
        ("process", process_cols),
        ("consumable", consumable_cols),
    ]:
        for col in columns:
            corr_val = safe_corr(
                eda_df[col],
                eda_df[target_col],
            )

            corr_rows.append({
                "feature": col,
                "feature_type": feature_type,
                "corr_with_target": corr_val,
                "abs_corr_with_target": (
                    abs(corr_val)
                    if pd.notna(corr_val)
                    else np.nan
                ),
            })

    eda_corr_df = (
        pd.DataFrame(
            corr_rows,
            columns=[
                "feature",
                "feature_type",
                "corr_with_target",
                "abs_corr_with_target",
            ],
        )
        .sort_values("abs_corr_with_target", ascending=False)
        .reset_index(drop=True)
    )

    top_process = (
        eda_corr_df.loc[
            eda_corr_df["feature_type"] == "process"
        ]
        .head(3)["feature"]
        .tolist()
    )

    top_consumable = (
        eda_corr_df.loc[
            eda_corr_df["feature_type"] == "consumable"
        ]
        .head(3)["feature"]
        .tolist()
    )

    hypothesis_rows = [
        {
            "feature": col,
            "corr_with_target": safe_corr(
                eda_df[col],
                eda_df[target_col],
            ),
        }
        for col in top_consumable + top_process
    ]

    hypothesis_df = (
        pd.DataFrame(
            hypothesis_rows,
            columns=["feature", "corr_with_target"],
        )
        .sort_values("corr_with_target", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "eda_df": eda_df,
        "eda_corr_df": eda_corr_df,
        "hypothesis_df": hypothesis_df,
        "top_process": top_process,
        "top_consumable": top_consumable,
    }


# =========================================================
# 2. Target Distribution Plots
# =========================================================

def plot_target_fixed_version(
    df_before,
    df_after,
    target_col="AVG_REMOVAL_RATE",
    bins=30,
    outlier_k=10,
):

    y_before = pd.to_numeric(
        df_before[target_col],
        errors="coerce",
    ).dropna()

    y_after = pd.to_numeric(
        df_after[target_col],
        errors="coerce",
    ).dropna()

    q1 = y_before.quantile(0.25)
    q3 = y_before.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - outlier_k * iqr
    upper = q3 + outlier_k * iqr

    y_before_outliers = y_before[
        (y_before < lower) | (y_before > upper)
    ]

    plot_upper = min(
        200,
        float(np.nanpercentile(y_after, 99.5)) + 5,
    )
    plot_lower = max(
        0,
        float(np.nanpercentile(y_after, 0.5)) - 5,
    )
    bin_edges = np.linspace(plot_lower, plot_upper, 45)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        y_after,
        bins=bin_edges,
        color="#4C78A8",
        edgecolor="white",
        alpha=0.85,
    )

    ax.set_xlim(plot_lower, plot_upper)
    ax.set_title(
        f"After Removing Extreme Outliers: {target_col}"
    )
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()

    print("\n=== y_after quick check ===")
    print(y_after.describe())

    print("\n=== Range counts ===")
    print("50-100 :", ((y_after >= 50) & (y_after <= 100)).sum())
    print("100-130:", ((y_after > 100) & (y_after <= 130)).sum())
    print("130-140:", ((y_after > 130) & (y_after < 140)).sum())
    print("140-160:", ((y_after >= 140) & (y_after <= 160)).sum())

    print("\n=== Plain-IQR outlier values before cleaning ===")

    if len(y_before_outliers) == 0:
        print("No outliers detected.")
    else:
        print(np.sort(y_before_outliers.values))

    return fig


def plot_validation_target_preserved(
    y_valid_raw,
    target_col="AVG_REMOVAL_RATE",
    bins=30,
):
    """Plot the supplied validation targets without further filtering."""
    y_valid_raw = pd.to_numeric(
        y_valid_raw,
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(y_valid_raw, bins=bins)
    ax.set_title(
        f"Validation Target (Original / Preserved): {target_col}"
    )
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")

    fig.tight_layout()
    plt.show()

    return fig


def plot_stage_distribution(
    eda_df,
    target_col="AVG_REMOVAL_RATE",
):
    """Plot target histograms by stage."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    stages = sorted(eda_df["STAGE"].dropna().unique())

    for stage_name, color in zip(stages, colors):
        y_stage = pd.to_numeric(
            eda_df.loc[
                eda_df["STAGE"] == stage_name,
                target_col,
            ],
            errors="coerce",
        ).dropna()

        if len(y_stage) > 0:
            ax.hist(
                y_stage,
                bins=30,
                alpha=0.45,
                label=f"STAGE {stage_name}",
                color=color,
            )

    ax.set_title(f"Stage-wise Distribution of {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()
    plt.show()

    return fig


# =========================================================
# 3. EDA Heatmap and Scatter Plots
# =========================================================

def plot_eda_correlation_heatmap(
    eda_df,
    top_process,
    top_consumable,
    target_col="AVG_REMOVAL_RATE",
):
    """Plot correlations among the selected EDA variables."""
    heatmap_cols = top_process + top_consumable + [target_col]
    heatmap_cols = [
        col for col in heatmap_cols
        if col in eda_df.columns
    ]

    corr_mat = eda_df[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))

    heatmap = ax.imshow(
        corr_mat,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    fig.colorbar(heatmap, ax=ax, label="Correlation")

    ax.set_xticks(range(len(corr_mat.columns)))
    ax.set_xticklabels(
        corr_mat.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(corr_mat.columns)))
    ax.set_yticklabels(corr_mat.columns)
    ax.set_title("Correlation Heatmap of Top EDA Variables")

    fig.tight_layout()
    plt.show()

    return fig


def plot_eda_feature_scatter(
    eda_df,
    top_process,
    top_consumable,
    target_col="AVG_REMOVAL_RATE",
):
    """Plot the top consumable and process features against the target."""
    scatter_consumable = (
        top_consumable[0] if len(top_consumable) > 0 else None
    )
    scatter_process = (
        top_process[0] if len(top_process) > 0 else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for ax, feature in zip(
        axes,
        [scatter_consumable, scatter_process],
    ):
        if feature is None:
            ax.axis("off")
            continue

        tmp = eda_df[[feature, target_col, "STAGE"]].copy()
        tmp[feature] = pd.to_numeric(
            tmp[feature], errors="coerce"
        )
        tmp[target_col] = pd.to_numeric(
            tmp[target_col], errors="coerce"
        )
        tmp = tmp.dropna()

        stages = sorted(tmp["STAGE"].dropna().unique())

        for stage_name, color in zip(stages, colors):
            part = tmp[tmp["STAGE"] == stage_name]

            ax.scatter(
                part[feature],
                part[target_col],
                s=20,
                alpha=0.55,
                label=f"STAGE {stage_name}",
                color=color,
            )

        ax.set_title(f"{feature} vs {target_col}")
        ax.set_xlabel(feature)
        ax.set_ylabel(target_col)
        ax.legend()

    fig.tight_layout()
    plt.show()

    return fig


# =========================================================
# 4. Actual vs Predicted Scatter Plot
# =========================================================

def plot_scatter(
    y_true,
    group_labels,
    pred,
    dataset_name,
    model_name,
    group_order=("low", "high"),
    group_color_map=None,
):
    """Plot original-scale predictions against actual removal rates."""
    if group_color_map is None:
        group_color_map = {
            "low": "blue",
            "high": "red",
        }

    y_true = np.asarray(y_true, dtype=float)
    group_labels = np.asarray(group_labels)
    pred = np.asarray(pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for group_name in group_order:
        idx = np.where(group_labels == group_name)[0]

        if len(idx) == 0:
            continue

        ax.scatter(
            y_true[idx],
            pred[idx],
            s=24,
            alpha=0.75,
            color=group_color_map[group_name],
            label=f"{group_name} group (n={len(idx)})",
        )

    min_v = min(np.min(y_true), np.min(pred))
    max_v = max(np.max(y_true), np.max(pred))

    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        "k--",
        linewidth=1.3,
        label="Perfect Prediction",
    )

    ax.set_xlabel("Actual Removal Rate")
    ax.set_ylabel("Predicted Removal Rate")
    ax.set_title(f"{dataset_name} ({model_name})")
    ax.legend()

    fig.tight_layout()
    plt.show()

    return fig


# =========================================================
# 5. Feature Importance Extraction
# =========================================================

def get_xgb_feature_importance(fitted_pack, feature_names):
    """Extract gain importance from the native XGBoost model."""
    booster = fitted_pack["model"]
    score_dict = booster.get_score(importance_type="gain")

    mapped = {}

    for i, col in enumerate(feature_names):
        mapped[col] = float(
            score_dict.get(f"f{i}", 0.0)
        )

    return pd.Series(mapped).sort_values(ascending=False)


def get_rf_feature_importance(fitted_pipe, feature_names):
    """Extract Random Forest impurity-based feature importance."""
    model = fitted_pipe.named_steps["model"]

    return pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)


def get_svr_permutation_importance(
    fitted_pipe,
    X_eval,
    y_eval_true_model_scale,
    random_state=42,
):

    result = permutation_importance(
        fitted_pipe,
        X_eval,
        y_eval_true_model_scale,
        n_repeats=5,
        random_state=random_state,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    return pd.Series(
        result.importances_mean,
        index=X_eval.columns,
    ).sort_values(ascending=False)


# =========================================================
# 6. Feature Importance Plots
# =========================================================

def plot_group_feature_importance(
    group_feature_importance,
    base_model_order=("XGBoost", "RandomForest", "SVR"),
    group_order=("low", "high"),
    top_n=15,
):

    figures = {}

    for model_name in base_model_order:
        fig, axes = plt.subplots(
            1,
            len(group_order),
            figsize=(14, 5),
            squeeze=False,
        )

        for ax, group_name in zip(axes[0], group_order):
            imp = (
                group_feature_importance[group_name][model_name]
                .dropna()
                .head(top_n)
                .sort_values(ascending=True)
            )

            ax.barh(
                imp.index,
                imp.values,
                color="#4C78A8",
            )
            ax.set_title(f"{model_name} - {group_name} group")
            ax.set_xlabel("Importance")
            ax.grid(axis="x", alpha=0.25)

        fig.tight_layout()
        plt.show()

        figures[model_name] = fig

    return figures