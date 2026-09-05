import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer


# =========================================================
# 1. Feature Complexity Ranking
# =========================================================

def feature_complexity_score(feature_name):

    name = str(feature_name).lower()

    if any(
        tag in name
        for tag in [
            "_median",
            "_mean",
            "_q25",
            "_q75",
            "_q10",
            "_q90",
            "_std",
        ]
    ):
        return 1

    if any(
        tag in name
        for tag in [
            "_first",
            "_last",
            "_delta",
            "_slope",
            "_iqr",
            "_robust_range",
        ]
    ):
        return 2

    if any(
        tag in name
        for tag in [
            "_seg1_",
            "_seg2_",
            "_seg3_",
            "_upper_tail_ratio",
            "_spike_count",
            "_stability",
            "_lag1_autocorr",
        ]
    ):
        return 3

    return 2


# =========================================================
# 2. Correlation-Based Prefilter
# =========================================================

def prefilter_features_by_correlation_global(
    X_df,
    corr_threshold=0.8,
):

    if X_df.shape[1] <= 1:
        return list(X_df.columns), pd.DataFrame()

    corr_mat = X_df.corr(method="pearson").abs()
    missing_map = X_df.isna().mean().to_dict()

    kept = list(X_df.columns)
    decision_rows = []
    cols = list(X_df.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1 = cols[i]
            f2 = cols[j]

            if f1 not in kept or f2 not in kept:
                continue

            corr_val = corr_mat.loc[f1, f2]

            if pd.isna(corr_val) or corr_val < corr_threshold:
                continue

            miss1 = float(missing_map.get(f1, 1.0))
            miss2 = float(missing_map.get(f2, 1.0))

            comp1 = feature_complexity_score(f1)
            comp2 = feature_complexity_score(f2)

            if abs(miss1 - miss2) > 1e-12:
                keep_feature = f1 if miss1 < miss2 else f2
                drop_feature = f2 if miss1 < miss2 else f1
                rule_used = "missing_ratio"

            elif comp1 != comp2:
                keep_feature = f1 if comp1 < comp2 else f2
                drop_feature = f2 if comp1 < comp2 else f1
                rule_used = "interpretability"

            else:
                keep_feature = f1
                drop_feature = f2
                rule_used = "tie_break_feature_order"

            kept.remove(drop_feature)

            decision_rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "abs_corr": float(corr_val),
                "missing_ratio_1": miss1,
                "missing_ratio_2": miss2,
                "complexity_1": comp1,
                "complexity_2": comp2,
                "keep_feature": keep_feature,
                "drop_feature": drop_feature,
                "rule_used": rule_used,
            })

    return kept, pd.DataFrame(decision_rows)


# =========================================================
# 3. Feature-Target Correlation Report
# =========================================================

def build_feature_correlation_table(
    X_df,
    y_series,
    group_labels,
    group_order=("low", "high"),
):

    rows = []

    def safe_corr(x, y):
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()

        if len(tmp) < 2:
            return np.nan

        if (
            tmp["x"].std(ddof=0) == 0
            or tmp["y"].std(ddof=0) == 0
        ):
            return np.nan

        return float(tmp["x"].corr(tmp["y"]))

    for col in X_df.columns:
        row = {"feature": col}
        row["corr_all_train"] = safe_corr(
            X_df[col],
            y_series,
        )

        for g in group_order:
            idx = np.where(np.asarray(group_labels) == g)[0]

            row[f"corr_{g}_train"] = safe_corr(
                X_df.iloc[idx][col].reset_index(drop=True),
                y_series.iloc[idx].reset_index(drop=True),
            )

        rows.append(row)

    corr_df = pd.DataFrame(rows)
    corr_df["abs_corr_all_train"] = (
        corr_df["corr_all_train"].abs()
    )

    corr_df = (
        corr_df.sort_values(
            "abs_corr_all_train",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return corr_df


# =========================================================
# 4. XGBoost Cumulative Importance Selection
# =========================================================

def select_features_by_cumulative_importance(
    X_df,
    y_series,
    group_name,
    cutoff=0.80,
    min_features=8,
    seed=42,
):

    imputer = SimpleImputer(strategy="median")

    X_imp = pd.DataFrame(
        imputer.fit_transform(X_df),
        columns=X_df.columns,
    )

    if group_name == "low":
        selector_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "verbosity": 0,
            "seed": seed,
            "n_estimators": 700,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.90,
        }
    else:
        selector_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "verbosity": 0,
            "seed": seed,
            "n_estimators": 550,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
        }

    selector = xgb.XGBRegressor(**selector_params)
    selector.fit(X_imp, y_series)

    imp = pd.Series(
        selector.feature_importances_,
        index=X_df.columns,
    ).sort_values(ascending=False)

    if imp.sum() <= 0:
        imp_norm = pd.Series(
            np.repeat(1.0 / len(imp), len(imp)),
            index=imp.index,
        )
    else:
        imp_norm = imp / imp.sum()

    imp_cum = imp_norm.cumsum()

    selected_count = (
        np.where(imp_cum.values >= cutoff)[0][0] + 1
    )
    selected_count = max(min_features, selected_count)

    selected_cols = imp.index[:selected_count].tolist()

    detail_df = pd.DataFrame({
        "feature": imp.index,
        "importance": imp.values,
        "normalized_importance": imp_norm.values,
        "cumulative_importance": imp_cum.values,
        "selected": [
            i < selected_count
            for i in range(len(imp))
        ],
    })

    return selected_cols, detail_df