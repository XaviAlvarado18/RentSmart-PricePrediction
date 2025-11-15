"""
Comprehensive Model Evaluation Pipeline for Rent Price Prediction - CuántoRento GT
Phase 5 - CRISP-DM: Evaluation

Implements:
- Classification metrics (Precision, Recall, F1-Score, AUC-ROC)
- Confusion matrix analysis
- ROC and Precision-Recall curves
- Feature importance and interpretability (SHAP optional)
- Temporal cross-validation
- Error analysis and edge cases
- Business impact evaluation
"""

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


# =====================================================
# 1. PRICE DISCRETIZATION
# =====================================================

def discretize_prices(y, bins='quantile', n_bins=3, labels=None):
    """
    Discretize continuous prices into categories for classification evaluation.
    
    Args:
        y: Continuous price values
        bins: 'quantile' or array of bin edges
        n_bins: Number of bins (if bins='quantile')
        labels: Category labels (default: ['bajo', 'medio', 'alto'])
    
    Returns:
        y_discrete: Discretized labels
        bin_edges: Bin edges used
        labels: Category labels
    """
    if labels is None:
        labels = ['bajo', 'medio', 'alto']
    
    if bins == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(y, quantiles)
        bin_edges[0] = y.min() - 1
        bin_edges[-1] = y.max() + 1
    else:
        bin_edges = bins
    
    y_discrete = pd.cut(y, bins=bin_edges, labels=labels, include_lowest=True)
    
    return y_discrete, bin_edges, labels


def predict_discrete(model, X, bin_edges, labels):
    """
    Convert continuous predictions to discrete categories.
    
    Args:
        model: Trained regression model
        X: Features
        bin_edges: Bin edges from discretization
        labels: Category labels
    
    Returns:
        y_pred_discrete: Discretized predictions
        y_pred_continuous: Continuous predictions
    """
    y_pred_continuous = model.predict(X)
    y_pred_discrete = pd.cut(y_pred_continuous, bins=bin_edges, labels=labels, include_lowest=True)
    return y_pred_discrete, y_pred_continuous


# =====================================================
# 2. CLASSIFICATION METRICS
# =====================================================

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None, average='weighted'):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC)
        average: Averaging strategy for multiclass
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1_score'] = float(f1)
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class.tolist()
    metrics['recall_per_class'] = recall_per_class.tolist()
    metrics['f1_per_class'] = f1_per_class.tolist()
    
    if y_pred_proba is not None:
        try:
            auc_roc_ovr = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
            metrics['auc_roc_ovr'] = float(auc_roc_ovr)
        except Exception as e:
            metrics['auc_roc_ovr'] = None
            metrics['auc_roc_error'] = str(e)
    
    return metrics


# =====================================================
# 3. CONFUSION MATRIX ANALYSIS
# =====================================================

def plot_confusion_matrix(y_true, y_pred, labels, normalize=True, save_path=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        normalize: Whether to normalize
        save_path: Path to save figure
    
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm_normalized
        fmt = '.2f'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_to_plot = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()
    
    return cm


# =====================================================
# 4. ROC AND PRECISION-RECALL CURVES
# =====================================================

def plot_roc_curves(y_true, y_pred_proba, labels, save_path=None):
    """
    Plot ROC curves for multiclass classification (One-vs-Rest).
    
    Args:
        y_true: True labels (encoded as integers)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        labels: Label names
        save_path: Path to save figure
    """
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    plt.figure(figsize=(10, 8))
    
    auc_scores = {}
    
    for i, label in enumerate(labels):
        if n_classes == 2 and i == 0:
            continue
        
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            auc_scores[label] = float(auc)
            
            plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})', linewidth=2)
        except Exception:
            auc_scores[label] = None
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()
    
    return auc_scores


def plot_precision_recall_curves(y_true, y_pred_proba, labels, save_path=None):
    """
    Plot Precision-Recall curves for multiclass classification.
    
    Args:
        y_true: True labels (encoded as integers)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        labels: Label names
        save_path: Path to save figure
    """
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    plt.figure(figsize=(10, 8))
    
    ap_scores = {}
    
    for i, label in enumerate(labels):
        if n_classes == 2 and i == 0:
            continue
        
        try:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            ap_scores[label] = float(ap)
            
            plt.plot(recall, precision, label=f'{label} (AP = {ap:.3f})', linewidth=2)
        except Exception:
            ap_scores[label] = None
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()
    
    return ap_scores


# =====================================================
# 5. FEATURE IMPORTANCE & INTERPRETABILITY
# =====================================================

def calculate_advanced_feature_importance(model, X, y, feature_names, method='both', n_repeats=10):
    """
    Calculate feature importance using multiple methods.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        feature_names: Feature names
        method: 'native', 'permutation', or 'both'
        n_repeats: Number of repeats for permutation importance
    
    Returns:
        DataFrame with feature importances
    """
    importance_data = {'feature': feature_names}
    
    if method in ['native', 'both']:
        if hasattr(model, 'feature_importances_'):
            importance_data['native_importance'] = model.feature_importances_
        else:
            importance_data['native_importance'] = np.zeros(len(feature_names))
    
    if method in ['permutation', 'both']:
        try:
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, scoring='neg_mean_squared_error'
            )
            importance_data['permutation_importance'] = perm_importance.importances_mean
            importance_data['permutation_std'] = perm_importance.importances_std
        except Exception as e:
            importance_data['permutation_importance'] = np.zeros(len(feature_names))
            importance_data['permutation_std'] = np.zeros(len(feature_names))
    
    if SHAP_AVAILABLE and method in ['shap', 'both']:
        try:
            explainer = shap.TreeExplainer(model) if hasattr(model, 'predict') else shap.KernelExplainer(model.predict, X.sample(min(100, len(X))))
            shap_values = explainer.shap_values(X.sample(min(500, len(X))))
            
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            importance_data['shap_importance'] = shap_importance
        except Exception as e:
            importance_data['shap_importance'] = np.zeros(len(feature_names))
    
    df = pd.DataFrame(importance_data)
    
    if 'native_importance' in df.columns:
        df = df.sort_values('native_importance', ascending=False)
    elif 'permutation_importance' in df.columns:
        df = df.sort_values('permutation_importance', ascending=False)
    
    return df


def plot_feature_importance(importance_df, top_n=20, method='native', save_path=None):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importances
        top_n: Number of top features to show
        method: Column name to use for importance
        save_path: Path to save figure
    """
    if method not in importance_df.columns:
        method = importance_df.columns[1]
    
    df_plot = importance_df.head(top_n).sort_values(method, ascending=True)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(df_plot)), df_plot[method].values)
    plt.yticks(range(len(df_plot)), df_plot['feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance ({method})', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()


# =====================================================
# 6. TEMPORAL CROSS-VALIDATION
# =====================================================

def temporal_cross_validation(model, X, y, n_splits=5, discrete=False, bin_edges=None, labels=None):
    """
    Perform temporal cross-validation.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        n_splits: Number of splits
        discrete: Whether to also evaluate as classification
        bin_edges: Bin edges for discretization
        labels: Labels for discretization
    
    Returns:
        Dictionary with CV results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'regression': {'mae': [], 'rmse': [], 'r2': []},
        'classification': {'accuracy': [], 'f1': [], 'precision': [], 'recall': []} if discrete else None
    }
    
    fold = 0
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        model_cv = type(model)(**model.get_params())
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(X_test_cv)
        
        mae = mean_absolute_error(y_test_cv, y_pred_cv)
        rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
        r2 = r2_score(y_test_cv, y_pred_cv)
        
        cv_results['regression']['mae'].append(mae)
        cv_results['regression']['rmse'].append(rmse)
        cv_results['regression']['r2'].append(r2)
        
        if discrete and bin_edges is not None:
            y_test_discrete = pd.cut(y_test_cv, bins=bin_edges, labels=labels, include_lowest=True)
            y_pred_discrete = pd.cut(y_pred_cv, bins=bin_edges, labels=labels, include_lowest=True)
            
            acc = accuracy_score(y_test_discrete, y_pred_discrete)
            f1 = f1_score(y_test_discrete, y_pred_discrete, average='weighted', zero_division=0)
            prec = precision_score(y_test_discrete, y_pred_discrete, average='weighted', zero_division=0)
            rec = recall_score(y_test_discrete, y_pred_discrete, average='weighted', zero_division=0)
            
            cv_results['classification']['accuracy'].append(acc)
            cv_results['classification']['f1'].append(f1)
            cv_results['classification']['precision'].append(prec)
            cv_results['classification']['recall'].append(rec)
    
    metrics_to_process = list(cv_results['regression'].keys())
    for metric in metrics_to_process:
        if isinstance(cv_results['regression'][metric], list):
            cv_results['regression'][metric + '_mean'] = np.mean(cv_results['regression'][metric])
            cv_results['regression'][metric + '_std'] = np.std(cv_results['regression'][metric])
    
    if discrete and cv_results['classification']:
        metrics_to_process_clf = list(cv_results['classification'].keys())
        for metric in metrics_to_process_clf:
            if isinstance(cv_results['classification'][metric], list):
                cv_results['classification'][metric + '_mean'] = np.mean(cv_results['classification'][metric])
                cv_results['classification'][metric + '_std'] = np.std(cv_results['classification'][metric])
    
    return cv_results


def plot_cv_results(cv_results, save_path=None):
    """
    Plot cross-validation results.
    
    Args:
        cv_results: Results from temporal_cross_validation
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['mae', 'rmse', 'r2']
    titles = ['MAE', 'RMSE', 'R²']
    
    for ax, metric, title in zip(axes, metrics, titles):
        data = cv_results['regression'][metric]
        ax.boxplot(data, vert=True)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} Across Folds', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()


# =====================================================
# 7. ERROR ANALYSIS & EDGE CASES
# =====================================================

def analyze_errors(y_true, y_pred, X, metadata_df=None):
    """
    Analyze prediction errors by segments.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        X: Features (may need original features with categorical info)
        metadata_df: DataFrame with metadata (zone, property_type, etc.)
    
    Returns:
        DataFrame with error analysis
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / y_true) * 100
    
    error_df = pd.DataFrame({
        'true_price': y_true,
        'pred_price': y_pred,
        'error': errors,
        'abs_error': abs_errors,
        'pct_error': pct_errors
    })
    
    if metadata_df is not None and len(metadata_df) == len(error_df):
        for col in metadata_df.columns:
            if col not in error_df.columns:
                error_df[col] = metadata_df[col].values
    
    error_df['is_outlier'] = error_df['abs_error'] > error_df['abs_error'].quantile(0.95)
    error_df['error_type'] = 'underestimate'
    error_df.loc[error_df['error'] > 0, 'error_type'] = 'overestimate'
    error_df.loc[error_df['abs_error'] < error_df['abs_error'].quantile(0.25), 'error_type'] = 'good'
    
    return error_df


def plot_error_analysis(error_df, segment_col, save_path=None):
    """
    Plot error analysis by segment.
    
    Args:
        error_df: DataFrame from analyze_errors
        segment_col: Column to segment by
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    segments = error_df[segment_col].unique()
    
    data_by_segment = [error_df[error_df[segment_col] == seg]['pct_error'].values for seg in segments]
    
    axes[0].boxplot(data_by_segment, labels=segments)
    axes[0].set_ylabel('Percentage Error (%)', fontsize=12)
    axes[0].set_xlabel(segment_col, fontsize=12)
    axes[0].set_title('Error Distribution by Segment', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    mean_errors = error_df.groupby(segment_col)['pct_error'].mean().sort_values()
    axes[1].barh(range(len(mean_errors)), mean_errors.values)
    axes[1].set_yticks(range(len(mean_errors)))
    axes[1].set_yticklabels(mean_errors.index)
    axes[1].set_xlabel('Mean Percentage Error (%)', fontsize=12)
    axes[1].set_title('Mean Error by Segment', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()


def identify_edge_cases(error_df, top_n=10):
    """
    Identify edge cases with highest errors.
    
    Args:
        error_df: DataFrame from analyze_errors
        top_n: Number of cases to return
    
    Returns:
        DataFrame with top edge cases
    """
    top_over = error_df.nlargest(top_n, 'abs_error')
    top_under = error_df.nsmallest(top_n, 'error')[error_df['error'] < 0].nlargest(top_n, 'abs_error')
    
    edge_cases = pd.concat([top_over, top_under]).drop_duplicates()
    edge_cases = edge_cases.sort_values('abs_error', ascending=False).head(top_n)
    
    return edge_cases


# =====================================================
# 8. BUSINESS IMPACT EVALUATION
# =====================================================

def calculate_business_impact(y_true, y_pred, config=None):
    """
    Calculate expected business impact.
    
    Args:
        y_true: True prices
        y_pred: Predicted prices
        config: Configuration dictionary with business parameters
    
    Returns:
        Dictionary with business metrics
    """
    if config is None:
        config = {
            'commission_rate': 0.10,
            'acceptable_error_threshold': 0.10,
            'days_on_market_reduction': 0.15,
            'avg_monthly_listings': 100
        }
    
    abs_errors = np.abs(y_pred - y_true)
    pct_errors = (abs_errors / y_true) * 100
    
    within_threshold = (pct_errors <= config['acceptable_error_threshold'] * 100).sum()
    accuracy_rate = within_threshold / len(y_true)
    
    avg_price = np.mean(y_true)
    avg_commission = avg_price * config['commission_rate']
    
    expected_reduction_dom = config['days_on_market_reduction']
    
    monthly_listings = config['avg_monthly_listings']
    expected_improvement = accuracy_rate * monthly_listings
    additional_revenue = expected_improvement * avg_commission * expected_reduction_dom
    
    impact = {
        'accuracy_within_threshold': float(accuracy_rate),
        'avg_price': float(avg_price),
        'avg_commission_per_listing': float(avg_commission),
        'expected_dom_reduction': float(expected_reduction_dom),
        'monthly_listings': monthly_listings,
        'expected_additional_revenue_monthly': float(additional_revenue),
        'mean_pct_error': float(np.mean(pct_errors)),
        'median_pct_error': float(np.median(pct_errors))
    }
    
    return impact


def plot_business_metrics(business_metrics, save_path=None):
    """
    Plot business impact metrics.
    
    Args:
        business_metrics: Dictionary from calculate_business_impact
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_to_plot = {
        'Accuracy Rate': business_metrics['accuracy_within_threshold'],
        'Mean Error %': business_metrics['mean_pct_error']
    }
    
    axes[0].bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=['green', 'orange'])
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Model Performance Metrics', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    
    revenue_metrics = {
        'Monthly Additional Revenue': business_metrics['expected_additional_revenue_monthly'],
        'Avg Commission per Listing': business_metrics['avg_commission_per_listing']
    }
    
    axes[1].bar(revenue_metrics.keys(), revenue_metrics.values(), color=['blue', 'purple'])
    axes[1].set_ylabel('Revenue (GTQ)', fontsize=12)
    axes[1].set_title('Business Impact', fontsize=14)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()


# =====================================================
# 9. COMPREHENSIVE EVALUATION FUNCTION
# =====================================================

def comprehensive_evaluation(
    model, X_train, y_train, X_val, y_val, X_test, y_test,
    model_name, metadata_df=None, mlflow_log=True, 
    n_cv_splits=5, business_config=None, use_shap=False
):
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        model_name: Name of the model
        metadata_df: Metadata for error analysis
        mlflow_log: Whether to log to MLflow
        n_cv_splits: Number of CV splits
        business_config: Business impact configuration
        use_shap: Whether to use SHAP (slower)
    
    Returns:
        Dictionary with all evaluation results
    """
    results = {
        'model_name': model_name,
        'regression_metrics': {},
        'classification_metrics': {},
        'feature_importance': None,
        'cv_results': None,
        'error_analysis': None,
        'business_impact': None
    }
    
    artifact_dir = f"eval_artifacts_{model_name}"
    os.makedirs(artifact_dir, exist_ok=True)
    
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    results['regression_metrics'] = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }
    
    if mlflow_log:
        mlflow.log_metric("eval_mae", mae)
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.log_metric("eval_r2", r2)
        mlflow.log_metric("eval_mape", mape)
    
    y_test_discrete, bin_edges, labels = discretize_prices(y_test, bins='quantile', n_bins=3)
    y_pred_discrete, _ = predict_discrete(model, X_test, bin_edges, labels)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test_discrete)
    y_pred_encoded = le.transform(y_pred_discrete)
    
    y_pred_proba = np.zeros((len(y_pred_test), 3))
    for i in range(3):
        dist = np.abs(y_pred_test - np.median(y_test[y_test_encoded == i]))
        y_pred_proba[:, i] = 1 / (1 + dist)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    classification_metrics = calculate_classification_metrics(
        y_test_encoded, y_pred_encoded, y_pred_proba
    )
    results['classification_metrics'] = classification_metrics
    
    if mlflow_log:
        for key, value in classification_metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"eval_{key}", value)
    
    plot_confusion_matrix(
        y_test_encoded, y_pred_encoded, range(3),
        save_path=os.path.join(artifact_dir, 'confusion_matrix.png')
    )
    
    plot_roc_curves(
        y_test_encoded, y_pred_proba, labels,
        save_path=os.path.join(artifact_dir, 'roc_curves.png')
    )
    
    plot_precision_recall_curves(
        y_test_encoded, y_pred_proba, labels,
        save_path=os.path.join(artifact_dir, 'pr_curves.png')
    )
    
    if mlflow_log:
        X_val_sample = X_val if len(X_val) < 1000 else X_val.sample(1000, random_state=42)
        if isinstance(y_val, pd.Series):
            y_val_sample = y_val.loc[X_val_sample.index]
        else:
            sample_indices = X_val_sample.index if isinstance(X_val, pd.DataFrame) else range(len(X_val_sample))
            y_val_sample = y_val[sample_indices]
        importance_df = calculate_advanced_feature_importance(
            model, X_val_sample, y_val_sample,
            X_val.columns.tolist() if isinstance(X_val, pd.DataFrame) else list(range(X_val.shape[1])),
            method='both' if use_shap and SHAP_AVAILABLE else 'both'
        )
    else:
        importance_df = calculate_advanced_feature_importance(
            model, X_val, y_val, 
            X_val.columns.tolist() if isinstance(X_val, pd.DataFrame) else list(range(X_val.shape[1])),
            method='native'
        )
    
    results['feature_importance'] = importance_df.to_dict('records')
    importance_df.to_csv(os.path.join(artifact_dir, 'feature_importance.csv'), index=False)
    plot_feature_importance(
        importance_df, top_n=20, method='native_importance' if 'native_importance' in importance_df.columns else importance_df.columns[1],
        save_path=os.path.join(artifact_dir, 'feature_importance.png')
    )
    
    if mlflow_log:
        mlflow.log_artifact(os.path.join(artifact_dir, 'feature_importance.csv'))
    
    cv_results = temporal_cross_validation(
        model, X_train, y_train, n_splits=n_cv_splits,
        discrete=True, bin_edges=bin_edges, labels=labels
    )
    results['cv_results'] = {
        k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} 
        if isinstance(v, dict) else v
        for k, v in cv_results.items()
    }
    
    plot_cv_results(cv_results, save_path=os.path.join(artifact_dir, 'cv_results.png'))
    
    error_df = analyze_errors(y_test, y_pred_test, X_test, metadata_df)
    results['error_analysis'] = {
        'mean_abs_error': float(error_df['abs_error'].mean()),
        'mean_pct_error': float(error_df['pct_error'].mean()),
        'outliers_count': int(error_df['is_outlier'].sum()),
        'edge_cases': identify_edge_cases(error_df, top_n=10).to_dict('records')
    }
    
    if metadata_df is not None and 'is_premium_zone' in metadata_df.columns:
        plot_error_analysis(
            error_df, 'is_premium_zone',
            save_path=os.path.join(artifact_dir, 'error_analysis_zone.png')
        )
    elif metadata_df is not None and len(metadata_df.columns) > 0:
        plot_error_analysis(
            error_df, metadata_df.columns[0],
            save_path=os.path.join(artifact_dir, 'error_analysis_segment.png')
        )
    
    business_impact = calculate_business_impact(y_test, y_pred_test, business_config)
    results['business_impact'] = business_impact
    
    plot_business_metrics(business_impact, save_path=os.path.join(artifact_dir, 'business_impact.png'))
    
    if mlflow_log:
        mlflow.log_metric("eval_accuracy_rate", business_impact['accuracy_within_threshold'])
        mlflow.log_metric("eval_expected_revenue_monthly", business_impact['expected_additional_revenue_monthly'])
        mlflow.log_artifacts(artifact_dir)
    
    return results


# =====================================================
# 10. REPORT GENERATION
# =====================================================

def generate_evaluation_report(results_dict, save_path="evaluation_report.json"):
    """
    Generate comprehensive evaluation report.
    
    Args:
        results_dict: Results from comprehensive_evaluation
        save_path: Path to save report
    """
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    return save_path


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    from data_preparation import prepare_data
    import joblib
    
    print("Loading data...")
    df = pd.read_csv("../data/raw/rent_guatemala.csv")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(df)
    
    print("Loading model...")
    client = mlflow.tracking.MlflowClient()
    model = mlflow.sklearn.load_model("models:/RentPriceModel/Production")
    
    print("Extracting metadata...")
    df_processed = pd.read_csv("../data/raw/rent_guatemala.csv")
    df_processed = df_processed.sort_values("listing_id").reset_index(drop=True)
    df_processed = df_processed.iloc[df_processed.index >= len(X_train) + len(X_val)]
    metadata_test = df_processed[['is_premium_zone', 'property_type', 'zone']].copy() if all(col in df_processed.columns for col in ['is_premium_zone', 'property_type', 'zone']) else None
    
    print("Running comprehensive evaluation...")
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("rent-price-modeling")
    
    with mlflow.start_run(run_name="comprehensive_evaluation"):
        results = comprehensive_evaluation(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            model_name="Production",
            metadata_df=metadata_test[['is_premium_zone', 'property_type', 'zone']] if 'is_premium_zone' in metadata_test.columns else None,
            mlflow_log=True,
            n_cv_splits=5
        )
        
        report_path = generate_evaluation_report(results, "comprehensive_evaluation_report.json")
        mlflow.log_artifact(report_path)
        print(f"Evaluation complete! Report saved to {report_path}")

