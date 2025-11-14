"""
Model Training Pipeline for Rent Price Prediction - CuántoRento GT
Phase 4 - CRISP-DM: Modeling with MLflow

Implements:
- MLflow experiment setup
- Training + automated hyperparameter tuning (RandomizedSearchCV)
- Logging params/metrics/artifacts (scaler, feature importance, residual plot)
- Logging model with signature & input_example
- Registering best model in Model Registry (safe version)
- Saving results.json
"""

import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# Optional models
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except:
    LGBM_AVAILABLE = False


# =====================================================
# Utilities
# =====================================================
def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def save_json_results(results, path="results.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def log_feature_importance(model, feature_names, artifact_path="feature_importance.csv"):
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": fi})
        df = df.sort_values("importance", ascending=False)
        df.to_csv(artifact_path, index=False)
        mlflow.log_artifact(artifact_path)


def log_residual_plot(y_true, y_pred, model_name, artifact_path="residuals.png"):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=1)
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    plt.title(f"True vs Pred - {model_name}")
    plt.tight_layout()
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)


# =====================================================
# Tuning helper
# =====================================================
def tune_model(model, param_distributions, X, y, n_iter=20, random_state=42, cv=3, scoring="neg_mean_squared_error"):
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_, rs.cv_results_


# =====================================================
# Train + log a model
# =====================================================
def train_and_log_model(model, model_name, X_train, y_train, X_val, y_val,
                        params, scaler_path=None, tune=False, tune_space=None):

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Log scaler
        if scaler_path is not None and os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path)

        # Optional tuning
        best_params = params.copy()
        if tune and tune_space is not None:
            try:
                best_estimator, best_params, _ = tune_model(
                    model, tune_space, X_train, y_train
                )
                model = best_estimator
                for k, v in best_params.items():
                    mlflow.log_param(f"tuned_{k}", v)
            except Exception as e:
                mlflow.log_param("tuning_error", str(e))

        # Log base params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        model.fit(X_train, y_train)

        # Predict and metrics
        preds = model.predict(X_val)
        mae, rmse, r2 = eval_metrics(y_val, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Feature importance
        log_feature_importance(model, X_train.columns)

        # Residual plot
        log_residual_plot(y_val, preds, model_name)

        # Log per-run results.json
        run_results = {
            "model_name": model_name,
            "run_id": run_id,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "params": best_params
        }
        mlflow.log_artifact(save_json_results(run_results, f"results_{model_name}.json"))

        # Save model with signature
        try:
            input_example = X_train.head(3)
            signature = infer_signature(X_train, model.predict(X_train.head(3)))
            mlflow.sklearn.log_model(
                model,
                model_name,
                signature=signature,
                input_example=input_example,
            )
        except Exception:
            mlflow.sklearn.log_model(model, model_name)

        return run_results


# =====================================================
# Run all experiments
# =====================================================
def run_experiments(X_train, y_train, X_val, y_val, scaler_path="../data/processed/scaler.pkl"):
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("rent-price-modeling")

    results = {}

    # -------- Linear Regression --------
    lr_params = {"fit_intercept": True}
    lr = LinearRegression(**lr_params)
    results["LinearRegression"] = train_and_log_model(
        lr, "LinearRegression", X_train, y_train, X_val, y_val,
        lr_params, scaler_path=scaler_path, tune=False
    )

    # -------- Random Forest --------
    rf_params = {"random_state": 42}
    rf = RandomForestRegressor(random_state=42)
    rf_space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [6, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        # FIXED: remove deprecated "auto"
        "max_features": ["sqrt", "log2", None]
    }
    results["RandomForest"] = train_and_log_model(
        rf, "RandomForest", X_train, y_train, X_val, y_val,
        rf_params, scaler_path=scaler_path, tune=True, tune_space=rf_space
    )

    # -------- XGBoost --------
    if XGB_AVAILABLE:
        xgb_params = {"random_state": 42}
        xgb = XGBRegressor(random_state=42, verbosity=0)
        xgb_space = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }
        results["XGBoost"] = train_and_log_model(
            xgb, "XGBoost", X_train, y_train, X_val, y_val,
            xgb_params, scaler_path=scaler_path, tune=True, tune_space=xgb_space
        )

    # -------- LightGBM --------
    if LGBM_AVAILABLE:
        lgb_params = {"random_state": 42}
        lgb = LGBMRegressor(random_state=42)
        lgb_space = {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.01, 0.05],
            "num_leaves": [31, 50, 80],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }
        results["LightGBM"] = train_and_log_model(
            lgb, "LightGBM", X_train, y_train, X_val, y_val,
            lgb_params, scaler_path=scaler_path, tune=True, tune_space=lgb_space
        )

    # Save aggregated
    mlflow.log_artifact(save_json_results(results, "all_results.json"))
    return results


# =====================================================
# Safe MLflow registration of best model
# =====================================================
def register_best_model(results, registered_name="RentPriceModel"):
    client = mlflow.tracking.MlflowClient()

    # Select best model by RMSE
    best_model_name = min(results, key=lambda m: results[m]["rmse"])
    best_run_id = results[best_model_name]["run_id"]

    print(f"\n>> Best model selected: {best_model_name}")

    # Ensure model exists
    try:
        client.get_registered_model(registered_name)
        print(f">> Registered model '{registered_name}' exists.")
    except:
        print(f">> Creating registered model '{registered_name}'...")
        client.create_registered_model(registered_name)

    # Register new version
    model_source = f"runs:/{best_run_id}/{best_model_name}"
    mv = client.create_model_version(
        name=registered_name,
        source=model_source,
        run_id=best_run_id
    )

    print(f">> Registered version {mv.version} for {registered_name}")
    return mv


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    from data_preparation import prepare_data

    df = pd.read_csv("../data/raw/rent_guatemala.csv")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(df)

    # Save scaler for MLflow
    os.makedirs("../data/processed", exist_ok=True)
    joblib.dump(scaler, "../data/processed/scaler.pkl")

    results = run_experiments(X_train, y_train, X_val, y_val,
                              scaler_path="../data/processed/scaler.pkl")

    mv = register_best_model(results)

    print("\nFinal Results Summary:")
    for model, v in results.items():
        print(model, v)
