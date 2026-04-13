# ml_wave/training.py

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from ml_wave.models import get_pipelines


def train_and_save_models(X, y, groups=None, global_mean=None, global_std=None, n_splits=5):
    """
    Train all pipelines with group-aware cross-validation.
    Saves KNN and LogReg models for cogency fusion, plus normalization stats.
    """
    pipelines = get_pipelines()
    results = {}

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print(f"Groups: {len(np.unique(groups))} unique repetitions")
    print("=" * 70)

    for name, pipeline in pipelines.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {name}")
        print("=" * 70)

        y_pred = cross_val_predict(pipeline, X, y, cv=cv, groups=groups)
        accuracy = np.mean(y_pred == y)
        results[name] = accuracy

        print(f"\nCV Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        print("\nConfusion Matrix:")
        labels = sorted(np.unique(y))
        cm = confusion_matrix(y, y_pred, labels=labels)

        print(f"{'':>12}", end="")
        for label in labels:
            print(f"{label:>12}", end="")
        print()
        for i, label in enumerate(labels):
            print(f"{label:>12}", end="")
            for j in range(len(labels)):
                print(f"{cm[i, j]:>12}", end="")
            print()

    # Summary comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:25s}: {acc:.4f}")

    # Train BOTH models needed for cogency fusion on full data
    print(f"\nTraining models for cogency fusion on full dataset...")

    knn_pipeline = get_pipelines()['lda_to_knn']
    logreg_pipeline = get_pipelines()['lda_to_logreg']

    knn_pipeline.fit(X, y)
    logreg_pipeline.fit(X, y)

    # Save bundle with both models for cogency fusion
    model_bundle = {
        "models": {
            "lda_to_knn": knn_pipeline,
            "lda_to_logreg": logreg_pipeline,
        },
        "best_model": max(results, key=results.get),
        "cv_results": results,
        "classes": list(np.unique(y)),
        "n_features": X.shape[1],
        "global_mean": global_mean,
        "global_std": global_std,
    }

    model_path = "models/gesture_model.joblib"
    joblib.dump(model_bundle, model_path)

    print(f"\nSaved: {model_path}")
    print(f"  - Models: lda_to_knn, lda_to_logreg")
    print(f"  - global_mean: {global_mean}")
    print(f"  - global_std: {global_std}")
    print(f"  - Best CV: {max(results, key=results.get)} ({max(results.values()):.4f})")

    return results
