"""
Model Training Module
Customer Churn Prediction Project
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.data_cleaning import load_data, clean_data
from src.preprocessing import prepare_data


def get_models():
    """Return dictionary of candidate classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, use_label_encoder=False,
            eval_metric="logloss", random_state=42, scale_pos_weight=3
        )
    return models


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics for a fitted pipeline."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
    }


def train_and_select(data_path: str, model_dir: str = "models"):
    """
    Full training loop:
    1. Load & clean data
    2. Prepare features
    3. Train all models
    4. Select best by ROC-AUC
    5. Save model + metadata
    """
    print("=" * 60)
    print("  Customer Churn Prediction — Model Training")
    print("=" * 60)

    # ── Load & clean ──────────────────────────────────────────────
    df_raw = load_data(data_path)
    df = clean_data(df_raw)

    # ── Prepare data ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = prepare_data(df)
    print(f"\n📊 Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"   Churn rate in train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}\n")

    # ── Train all models ──────────────────────────────────────────
    models = get_models()
    results = {}

    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = {"pipeline": pipeline, "metrics": metrics}
        print(f"  {name:<25}  Acc={metrics['Accuracy']:.4f}  "
              f"Rec={metrics['Recall']:.4f}  AUC={metrics['ROC-AUC']:.4f}")

    # ── Select best model (ROC-AUC) ───────────────────────────────
    best_name = max(results, key=lambda n: results[n]["metrics"]["ROC-AUC"])
    best_pipeline = results[best_name]["pipeline"]
    best_metrics = results[best_name]["metrics"]

    print(f"\n🏆 Best model: {best_name}")
    print(f"   Metrics: {best_metrics}")

    # ── Save artifacts ────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "churn_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\n💾 Model saved → {model_path}")

    # Build comparison table
    comparison = {
        name: res["metrics"]
        for name, res in results.items()
    }

    # Feature importance (tree-based models)
    feature_importance = {}
    try:
        clf = best_pipeline.named_steps["classifier"]
        ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        all_feature_names = num_cols + cat_feature_names
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            fi = sorted(
                zip(all_feature_names, importances),
                key=lambda x: x[1], reverse=True
            )[:15]
            feature_importance = {k: float(v) for k, v in fi}
    except Exception:
        pass

    # Confusion matrix values
    from sklearn.metrics import confusion_matrix
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()

    # ROC curve data
    from sklearn.metrics import roc_curve
    y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)

    metadata = {
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "model_comparison": comparison,
        "feature_importance": feature_importance,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {
            "fpr": fpr.tolist()[:200],
            "tpr": tpr.tolist()[:200],
        },
        "dataset_info": {
            "total_rows": len(df),
            "churn_rate": float(df["Churn"].mean()),
            "features": len(num_cols) + len(cat_cols),
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
        },
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved → {meta_path}")
    print("\n✅ Training complete!\n")

    return best_pipeline, metadata


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model_dir = os.path.join(base, "models")
    train_and_select(data_path, model_dir)
