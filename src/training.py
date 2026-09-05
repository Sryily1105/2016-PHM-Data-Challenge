import json

import numpy as np
import pandas as pd
import optuna

from feature_selection import select_features_by_cumulative_importance
from model_building import (
    suggest_params,
    fit_model,
    predict_model,
    transform_target,
    inverse_target_transform,
    normalize_weight_dict,
    apply_weighted_ensemble,
    rmse,
    mae,
)


# =========================================================
# 1. Sliding-Window Splits
# =========================================================

def sliding_window_split(
    n_samples,
    n_splits=5,
    train_ratio=0.8,
    test_ratio=0.2,
):
    """Build fixed-length sliding windows using the original CMP rule."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    if abs(train_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + test_ratio must equal 1.0")

    if n_samples <= n_splits:
        raise ValueError(
            f"n_samples={n_samples} is too small "
            f"for n_splits={n_splits}"
        )

    ratio = train_ratio / test_ratio
    test_size = int(n_samples // (n_splits + ratio))

    if test_size <= 0:
        raise ValueError(
            f"test_size too small for n_samples={n_samples}, "
            f"n_splits={n_splits}"
        )

    step_size = test_size
    train_size = int(n_samples - n_splits * test_size)

    if train_size <= 0:
        raise ValueError(
            f"train_size too small: train_size={train_size}, "
            f"n_samples={n_samples}, n_splits={n_splits}"
        )

    splits = []

    for fold_id in range(n_splits):
        train_start = fold_id * step_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n_samples:
            raise ValueError(
                f"Fold {fold_id + 1} exceeds sample range: "
                f"test_end={test_end}, n_samples={n_samples}"
            )

        tr_idx = np.arange(train_start, train_end, dtype=int)
        te_idx = np.arange(test_start, test_end, dtype=int)

        splits.append((tr_idx, te_idx))

    return splits, train_size, test_size


# =========================================================
# 2. Bayesian Hyperparameter Tuning
# =========================================================

def tune_model_with_bayes(
    model_name,
    group_name,
    X_df,
    y_series_model_scale,
    inner_folds,
    seed,
    n_trials,
    use_log_target=False,
    n_jobs=-1,
    show_fold_log=False,
):

    def objective(trial):
        params = suggest_params(trial, model_name, group_name)

        inner_splits, _, _ = sliding_window_split(
            n_samples=len(X_df),
            n_splits=inner_folds,
            train_ratio=0.8,
            test_ratio=0.2,
        )

        score_list = []

        for fold_id, (tr_idx, va_idx) in enumerate(
            inner_splits,
            start=1,
        ):
            if show_fold_log:
                print(
                    f"    Inner Fold {fold_id}: "
                    f"train={len(tr_idx)} | valid={len(va_idx)} | "
                    f"train_idx=[{tr_idx[0]}:{tr_idx[-1]}] | "
                    f"valid_idx=[{va_idx[0]}:{va_idx[-1]}]"
                )

            X_tr = X_df.iloc[tr_idx].reset_index(drop=True)
            X_va = X_df.iloc[va_idx].reset_index(drop=True)

            y_tr = (
                y_series_model_scale.iloc[tr_idx]
                .reset_index(drop=True)
            )
            y_va = (
                y_series_model_scale.iloc[va_idx]
                .reset_index(drop=True)
            )

            y_tr_original_scale = inverse_target_transform(
                y_tr.values,
                use_log_target=use_log_target,
            )

            selected_cols_inner, _ = (
                select_features_by_cumulative_importance(
                    X_df=X_tr,
                    y_series=y_tr_original_scale,
                    group_name=group_name,
                    cutoff=0.80,
                    min_features=8,
                    seed=seed + fold_id,
                )
            )

            X_tr_sel = (
                X_tr[selected_cols_inner]
                .reset_index(drop=True)
            )
            X_va_sel = (
                X_va[selected_cols_inner]
                .reset_index(drop=True)
            )

            fitted = fit_model(
                model_name,
                X_tr_sel,
                y_tr,
                params,
                seed + fold_id,
                n_jobs=n_jobs,
            )

            pred = predict_model(
                model_name,
                fitted,
                X_va_sel,
                use_log_target=use_log_target,
            )

            true = inverse_target_transform(
                y_va.values,
                use_log_target=use_log_target,
            )

            fold_rmse = rmse(true, pred)
            true_std = float(np.std(true))
            pred_std = float(np.std(pred))

            flat_penalty = max(
                0.0,
                0.5 * true_std - pred_std,
            )

            score_list.append(
                fold_rmse + 0.8 * flat_penalty
            )

        return float(np.mean(score_list))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    return study.best_params, float(study.best_value)


# =========================================================
# 3. Group-Wise Nested CV and Final Refit
# =========================================================

def train_groupwise_models(
    X_train,
    y_train,
    final_train_group,
    train_time_order,
    group_order=("low", "high"),
    base_model_order=("XGBoost", "RandomForest", "SVR"),
    outer_folds=5,
    inner_folds=4,
    n_trials=15,
    random_state=42,
    use_log_target=False,
    n_jobs=-1,
    show_fold_log=False,
):

    y_train = pd.Series(y_train).reset_index(drop=True)
    train_time_order = (
        pd.Series(train_time_order)
        .reset_index(drop=True)
    )
    final_train_group = np.asarray(final_train_group)

    group_outer_rows = []
    group_param_rows = []
    feature_selection_rows = []

    group_model_artifacts = {
        model_name: {}
        for model_name in base_model_order
    }
    group_selected_features = {}

    for group_name in group_order:
        idx_group = np.where(
            final_train_group == group_name
        )[0]

        X_group = (
            X_train.iloc[idx_group]
            .reset_index(drop=True)
        )
        y_group = (
            y_train.iloc[idx_group]
            .reset_index(drop=True)
        )
        time_group = (
            train_time_order.iloc[idx_group]
            .reset_index(drop=True)
        )

        if time_group.isna().any():
            raise ValueError(
                f"{group_name} group contains missing "
                "START_TIME_ORD values and cannot be time-sorted."
            )

        sort_idx = np.argsort(time_group.values)

        X_group = (
            X_group.iloc[sort_idx]
            .reset_index(drop=True)
        )
        y_group = (
            y_group.iloc[sort_idx]
            .reset_index(drop=True)
        )

        y_group_model = transform_target(
            y_group,
            use_log_target=use_log_target,
        )

        print("\n" + "=" * 90)
        print(
            f"TRAIN GROUP = {group_name} | "
            f"samples = {len(X_group)}"
        )
        print("=" * 90)

        outer_splits, _, _ = sliding_window_split(
            n_samples=len(X_group),
            n_splits=outer_folds,
            train_ratio=0.8,
            test_ratio=0.2,
        )

        # Nested cross-validation
        for outer_fold, (tr_idx, te_idx) in enumerate(
            outer_splits,
            start=1,
        ):
            if show_fold_log:
                print(
                    f"  Outer Fold {outer_fold}: "
                    f"train={len(tr_idx)} | test={len(te_idx)} | "
                    f"train_idx=[{tr_idx[0]}:{tr_idx[-1]}] | "
                    f"test_idx=[{te_idx[0]}:{te_idx[-1]}]"
                )

            X_outer_tr = (
                X_group.iloc[tr_idx]
                .reset_index(drop=True)
            )
            X_outer_te = (
                X_group.iloc[te_idx]
                .reset_index(drop=True)
            )
            y_outer_tr = (
                y_group.iloc[tr_idx]
                .reset_index(drop=True)
            )
            y_outer_te = (
                y_group.iloc[te_idx]
                .reset_index(drop=True)
            )

            y_outer_tr_model = transform_target(
                y_outer_tr,
                use_log_target=use_log_target,
            )

            selected_cols_fold, detail_df_fold = (
                select_features_by_cumulative_importance(
                    X_df=X_outer_tr,
                    y_series=y_outer_tr,
                    group_name=group_name,
                    cutoff=0.80,
                    min_features=8,
                    seed=random_state + outer_fold,
                )
            )

            detail_df_fold["group"] = group_name
            detail_df_fold["outer_fold"] = outer_fold
            feature_selection_rows.append(detail_df_fold)

            X_outer_tr_sel = (
                X_outer_tr[selected_cols_fold]
                .reset_index(drop=True)
            )
            X_outer_te_sel = (
                X_outer_te[selected_cols_fold]
                .reset_index(drop=True)
            )

            for model_name in base_model_order:
                if show_fold_log:
                    print(f"  Tuning {model_name} ...")

                best_params, best_inner_rmse = (
                    tune_model_with_bayes(
                        model_name=model_name,
                        group_name=group_name,
                        X_df=X_outer_tr,
                        y_series_model_scale=y_outer_tr_model,
                        inner_folds=inner_folds,
                        seed=(
                            random_state
                            + outer_fold * 100
                            + len(model_name)
                            + len(group_name)
                        ),
                        n_trials=n_trials,
                        use_log_target=use_log_target,
                        n_jobs=n_jobs,
                        show_fold_log=show_fold_log,
                    )
                )

                fitted_obj = fit_model(
                    model_name=model_name,
                    X_tr=X_outer_tr_sel,
                    y_tr=y_outer_tr_model,
                    params=best_params,
                    seed=(
                        random_state
                        + outer_fold * 1000
                        + len(group_name)
                    ),
                    n_jobs=n_jobs,
                )

                pred_outer = predict_model(
                    model_name,
                    fitted_obj,
                    X_outer_te_sel,
                    use_log_target=use_log_target,
                )

                group_outer_rows.append({
                    "group": group_name,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "inner_best_rmse": best_inner_rmse,
                    "outer_rmse": rmse(
                        y_outer_te.values,
                        pred_outer,
                    ),
                    "outer_mae": mae(
                        y_outer_te.values,
                        pred_outer,
                    ),
                    "best_params": json.dumps(
                        best_params,
                        ensure_ascii=False,
                    ),
                    "n_selected_features": len(
                        selected_cols_fold
                    ),
                    "selected_features": json.dumps(
                        selected_cols_fold,
                        ensure_ascii=False,
                    ),
                })

        # Feature selection on the full training group
        selected_cols_final, detail_df_final = (
            select_features_by_cumulative_importance(
                X_df=X_group,
                y_series=y_group,
                group_name=group_name,
                cutoff=0.80,
                min_features=8,
                seed=random_state + 900,
            )
        )

        detail_df_final["group"] = group_name
        detail_df_final["outer_fold"] = "final_refit"
        feature_selection_rows.append(detail_df_final)

        group_selected_features[group_name] = (
            selected_cols_final
        )

        X_group_sel = (
            X_group[selected_cols_final]
            .reset_index(drop=True)
        )

        # Retune and refit each model on the full training group
        for model_name in base_model_order:
            if show_fold_log:
                print(
                    f"\nRefit {model_name} on full "
                    f"TRAIN GROUP = {group_name}"
                )

            best_params, best_inner_rmse = (
                tune_model_with_bayes(
                    model_name=model_name,
                    group_name=group_name,
                    X_df=X_group,
                    y_series_model_scale=y_group_model,
                    inner_folds=inner_folds,
                    seed=(
                        random_state
                        + 9000
                        + len(model_name)
                        + len(group_name)
                    ),
                    n_trials=n_trials,
                    use_log_target=use_log_target,
                    n_jobs=n_jobs,
                    show_fold_log=show_fold_log,
                )
            )

            fitted_obj = fit_model(
                model_name=model_name,
                X_tr=X_group_sel,
                y_tr=y_group_model,
                params=best_params,
                seed=(
                    random_state
                    + 9500
                    + len(group_name)
                ),
                n_jobs=n_jobs,
            )

            group_model_artifacts[model_name][group_name] = (
                fitted_obj
            )

            group_param_rows.append({
                "group": group_name,
                "model": model_name,
                "best_inner_rmse_train": best_inner_rmse,
                "best_params": json.dumps(
                    best_params,
                    ensure_ascii=False,
                ),
            })

    return {
        "group_model_artifacts": group_model_artifacts,
        "group_selected_features": group_selected_features,
        "group_outer_result_df": pd.DataFrame(group_outer_rows),
        "group_param_df": pd.DataFrame(group_param_rows),
        "feature_selection_df": pd.concat(
            feature_selection_rows,
            axis=0,
        ).reset_index(drop=True),
    }


# =========================================================
# 4. Group-Routed Prediction
# =========================================================

def predict_by_true_group(
    X_df,
    true_group_labels,
    model_name,
    group_model_artifacts,
    group_selected_features,
    group_order=("low", "high"),
    use_log_target=False,
):
    """Route each row to its supplied group's fitted model."""
    preds = np.zeros(len(X_df), dtype=float)
    true_group_labels = np.asarray(true_group_labels)

    for group_name in group_order:
        idx = np.where(
            true_group_labels == group_name
        )[0]

        if len(idx) == 0:
            continue

        fitted_obj = (
            group_model_artifacts[model_name][group_name]
        )
        selected_cols = group_selected_features[group_name]

        preds[idx] = predict_model(
            model_name=model_name,
            fitted_obj=fitted_obj,
            X_te=(
                X_df.iloc[idx][selected_cols]
                .reset_index(drop=True)
            ),
            use_log_target=use_log_target,
        )

    return preds


def build_prediction_map(
    X_df,
    group_labels,
    group_model_artifacts,
    group_selected_features,
    base_model_order=("XGBoost", "RandomForest", "SVR"),
    group_order=("low", "high"),
    use_log_target=False,
):
    """Generate base-model predictions and their arithmetic mean."""
    pred_map = {}

    for model_name in base_model_order:
        pred_map[model_name] = predict_by_true_group(
            X_df=X_df,
            true_group_labels=group_labels,
            model_name=model_name,
            group_model_artifacts=group_model_artifacts,
            group_selected_features=group_selected_features,
            group_order=group_order,
            use_log_target=use_log_target,
        )

    pred_map["EnsembleMean"] = np.mean(
        np.column_stack([
            pred_map[model_name]
            for model_name in base_model_order
        ]),
        axis=1,
    )

    return pred_map


# =========================================================
# 5. Validation-Based Ensemble Weight Tuning
# =========================================================

def tune_groupwise_ensemble_weights(
    valid_pred_map,
    y_valid_true,
    valid_group_labels,
    model_names,
    n_trials=60,
    seed=42,
    group_order=("low", "high"),
):

    y_valid_true = np.asarray(y_valid_true, dtype=float)
    valid_group_labels = np.asarray(valid_group_labels)

    def objective(trial):
        weights_by_group = {}

        for group_name in group_order:
            raw = {
                model_name: trial.suggest_float(
                    f"{group_name}_{model_name}_weight",
                    0.0,
                    4.0,
                )
                for model_name in model_names
            }

            weights_by_group[group_name] = (
                normalize_weight_dict(raw)
            )

        pred = apply_weighted_ensemble(
            valid_pred_map,
            valid_group_labels,
            weights_by_group,
            model_names,
            group_order=group_order,
        )

        return rmse(y_valid_true, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_weights = {}
    rows = []

    for group_name in group_order:
        raw = {
            model_name: study.best_params[
                f"{group_name}_{model_name}_weight"
            ]
            for model_name in model_names
        }

        best_weights[group_name] = normalize_weight_dict(raw)

        row = {
            "group": group_name,
            "validation_rmse": float(study.best_value),
        }

        row.update({
            f"weight_{model_name}": (
                best_weights[group_name][model_name]
            )
            for model_name in model_names
        })

        rows.append(row)

    return best_weights, pd.DataFrame(rows)