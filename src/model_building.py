import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# =========================================================
# 1. Target Transformation
# =========================================================

def transform_target(y, use_log_target=False):
    """Convert targets to the model scale."""
    y = pd.Series(y).astype(float).reset_index(drop=True)

    return np.log1p(y) if use_log_target else y.copy()


def inverse_target_transform(arr, use_log_target=False):
    """Convert predictions back to the original target scale."""
    arr = np.asarray(arr, dtype=float)

    return np.expm1(arr) if use_log_target else arr


# =========================================================
# 2. Hyperparameter Search Spaces
# =========================================================

def suggest_params(trial, model_name, group_name):
    """Suggest parameters using the original CMP search spaces."""
    if model_name == "XGBoost":
        if group_name == "low":
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 650, 900
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 2, 4
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.04, log=True
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.80, 1.00
                ),
            }

        else:
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 500, 750
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 2, 3
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.04, log=True
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.85, 1.00
                ),
            }

    if model_name == "RandomForest":
        if group_name == "low":
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 250, 380
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 12, 22
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 6
                ),
                "max_features": trial.suggest_float(
                    "max_features", 0.25, 0.45
                ),
            }

        else:
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 220, 340
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 6, 12
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 4
                ),
                "max_features": trial.suggest_float(
                    "max_features", 0.20, 0.40
                ),
            }

    if model_name == "SVR":
        if group_name == "low":
            return {
                "kernel": "rbf",
                "C": trial.suggest_float(
                    "C", 30.0, 100.0, log=True
                ),
                "epsilon": trial.suggest_float(
                    "epsilon", 0.003, 0.02, log=True
                ),
                "gamma": trial.suggest_float(
                    "gamma", 0.001, 0.01, log=True
                ),
            }

        else:
            return {
                "kernel": "rbf",
                "C": trial.suggest_float(
                    "C", 3.0, 20.0, log=True
                ),
                "epsilon": trial.suggest_float(
                    "epsilon", 0.02, 0.08, log=True
                ),
                "gamma": trial.suggest_float(
                    "gamma", 2e-4, 2e-3, log=True
                ),
            }

    raise ValueError(f"Unknown model_name: {model_name}")


# =========================================================
# 3. Native XGBoost Training and Prediction
# =========================================================

def fit_xgb_native(X_tr, y_tr, params, seed):
    """Fit native XGBoost after median imputation.

    y_tr must be a pandas Series already on the intended model scale.
    """
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)

    dtrain = xgb.DMatrix(
        X_tr_imp,
        label=y_tr.values,
    )

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": seed,
        "max_depth": params["max_depth"],
        "eta": params["learning_rate"],
        "subsample": 0.85,
        "colsample_bytree": params["colsample_bytree"],
    }

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=params["n_estimators"],
        verbose_eval=False,
    )

    return {
        "imputer": imputer,
        "model": model,
    }


def predict_xgb_native(
    fitted_pack,
    X_te,
    use_log_target=False,
):
    """Predict with fitted XGBoost and return original-scale values."""
    X_te_imp = fitted_pack["imputer"].transform(X_te)
    dtest = xgb.DMatrix(X_te_imp)

    pred_model_scale = fitted_pack["model"].predict(dtest)

    return inverse_target_transform(
        pred_model_scale,
        use_log_target=use_log_target,
    )


# =========================================================
# 4. Scikit-Learn Pipelines
# =========================================================

def build_sklearn_pipeline(
    model_name,
    params,
    seed,
    n_jobs=-1,
):
    """Build the original SVR or Random Forest pipeline."""
    if model_name == "SVR":
        model = SVR(**params)

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    if model_name == "RandomForest":
        model = RandomForestRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            bootstrap=True,
            criterion="squared_error",
            min_samples_leaf=1,
            **params,
        )

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    raise ValueError(
        f"Unknown sklearn model_name: {model_name}"
    )


# =========================================================
# 5. Unified Training and Prediction Interfaces
# =========================================================

def fit_model(
    model_name,
    X_tr,
    y_tr,
    params,
    seed,
    n_jobs=-1,
):
    if model_name == "XGBoost":
        return fit_xgb_native(
            X_tr,
            y_tr,
            params,
            seed,
        )

    pipe = build_sklearn_pipeline(
        model_name,
        params,
        seed,
        n_jobs=n_jobs,
    )
    pipe.fit(X_tr, y_tr)

    return pipe


def predict_model(
    model_name,
    fitted_obj,
    X_te,
    use_log_target=False,
):

    if model_name == "XGBoost":
        return predict_xgb_native(
            fitted_obj,
            X_te,
            use_log_target=use_log_target,
        )

    pred_model_scale = fitted_obj.predict(X_te)

    return inverse_target_transform(
        pred_model_scale,
        use_log_target=use_log_target,
    )


# =========================================================
# 6. Ensemble Utilities
# =========================================================

def normalize_weight_dict(raw_weight_dict):

    clipped = {
        key: max(0.0, float(value))
        for key, value in raw_weight_dict.items()
    }
    total = sum(clipped.values())

    if total <= 0:
        return {
            key: 1.0 / len(clipped)
            for key in clipped
        }

    return {
        key: value / total
        for key, value in clipped.items()
    }


def apply_weighted_ensemble(
    pred_map,
    group_labels,
    weights_by_group,
    model_names,
    group_order=("low", "high"),
):

    group_labels = np.asarray(group_labels)
    out = np.zeros(len(group_labels), dtype=float)

    for group_name in group_order:
        idx = np.where(group_labels == group_name)[0]

        if len(idx) == 0:
            continue

        weights = weights_by_group[group_name]
        acc = np.zeros(len(idx), dtype=float)

        for model_name in model_names:
            acc += (
                weights[model_name]
                * np.asarray(pred_map[model_name])[idx]
            )

        out[idx] = acc

    return out

# =========================================================
# 7. Evaluation Metrics
# =========================================================

def rmse(y_true, y_pred):
    """Calculate root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Calculate mean absolute error."""
    return float(
        np.mean(
            np.abs(np.asarray(y_true) - np.asarray(y_pred))
        )
    )


def mse(y_true, y_pred):
    """Calculate mean squared error."""
    return float(
        np.mean(
            (np.asarray(y_true) - np.asarray(y_pred)) ** 2
        )
    )


def squared_error_stats(y_true, y_pred):

    sq_err = (
        np.asarray(y_true, dtype=float)
        - np.asarray(y_pred, dtype=float)
    ) ** 2

    if len(sq_err) == 0:
        return {
            "mse_mean": np.nan,
            "mse_std": np.nan,
        }

    if len(sq_err) == 1:
        return {
            "mse_mean": float(np.mean(sq_err)),
            "mse_std": 0.0,
        }

    return {
        "mse_mean": float(np.mean(sq_err)),
        "mse_std": float(np.std(sq_err, ddof=1)),
    }