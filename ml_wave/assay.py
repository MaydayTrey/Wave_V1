# ml_wave/assay.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.base import clone
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# ==============================================================================
# FEATURE NAME REGISTRY - CLEANED UP (removed low-discriminative timing features)
# ==============================================================================
FEATURE_NAMES = []

channels = ["CH1", "CH2", "CH3", "CH4"]

# Per-channel temporal features (6 per channel = 24 total)
temporal_feats = ["RMS", "MAV", "WL", "ZCR", "ENV_SLOPE", "ENV_TREND"]

# Per-channel frequency features (2 per channel = 8 total)
freq_feats = ["B20_50", "B50_90"]

# Spatial features (11 total) - envelope-based with normalized ratios
spatial_feats = [
    "D12", "D23", "D34", "D13", "D24",  # 5 differences
    "R12", "R34",  # 2 pairwise ratios
    "NORM_CH1", "NORM_CH2", "NORM_CH3", "NORM_CH4"  # 4 normalized ratios
]

# Cross-channel temporal features (8 total)
cross_temp_feats = [
    "SLOPE_12", "SLOPE_23", "SLOPE_34",
    "CORR_12", "CORR_23", "CORR_34",
    "CORR_13", "CORR_24",
]

# Covariance features (6 total)
cov_feats = [
    "COV_12", "COV_13", "COV_14",
    "COV_23", "COV_24", "COV_34",
]

# Build feature names list in order matching generator.py
for ch in channels:
    for f in temporal_feats:
        FEATURE_NAMES.append(f"{ch}_{f}")
    for f in freq_feats:
        FEATURE_NAMES.append(f"{ch}_{f}")

FEATURE_NAMES.extend(spatial_feats)
FEATURE_NAMES.extend(cross_temp_feats)
FEATURE_NAMES.extend(cov_feats)

# Feature count verification
EXPECTED_FEATURES = (
        4 * (6 + 2) +  # 4 channels × (6 temporal + 2 freq) = 32
        11 +  # spatial
        8 +  # cross-channel temporal
        6  # covariance
)  # Total: 57

assert len(FEATURE_NAMES) == EXPECTED_FEATURES, \
    f"Registry mismatch: {len(FEATURE_NAMES)} != {EXPECTED_FEATURES}"

print(f"[assay.py] Feature registry: {len(FEATURE_NAMES)} features")
print(f"[assay.py] Expected: {EXPECTED_FEATURES} features")

if len(FEATURE_NAMES) != EXPECTED_FEATURES:
    print(f"[assay.py] WARNING: Feature count mismatch!")

# ==============================================================================
# FEATURE GROUPS FOR TARGETED ANALYSIS - UPDATED
# ==============================================================================
FEATURE_GROUPS = {
    "temporal": [f for f in FEATURE_NAMES if any(t in f for t in ["RMS", "MAV", "WL", "ZCR"])],
    "envelope": [f for f in FEATURE_NAMES if "ENV" in f],
    "frequency": [f for f in FEATURE_NAMES if "B20" in f or "B50" in f],
    "spatial": spatial_feats,
    "covariance": cov_feats,
    "cross_channel": cross_temp_feats,
    # Updated swipe discriminators - features that EXIST and should help differentiate swipes
    "swipe_discriminators": [
        # Normalized channel ratios (should show directional activation patterns)
        "NORM_CH1", "NORM_CH2", "NORM_CH3", "NORM_CH4",
        # Envelope slopes (timing/dynamics)
        "CH1_ENV_SLOPE", "CH2_ENV_SLOPE", "CH3_ENV_SLOPE", "CH4_ENV_SLOPE",
        # Cross-channel slopes (relative activation order)
        "SLOPE_12", "SLOPE_23", "SLOPE_34",
        # Cross-channel correlations (co-activation patterns)
        "CORR_13", "CORR_24",
        # Spatial differences (amplitude balance)
        "D13", "D24",
    ]
}


def get_feature_indices(feature_names_subset):
    """Get indices for a subset of feature names."""
    return [FEATURE_NAMES.index(f) for f in feature_names_subset if f in FEATURE_NAMES]


# ==============================================================================
# CROSS VALIDATION - FIXED FOR GROUPED DATA
# ==============================================================================
def cross_validation_model(model, X, y, groups=None, n_splits=5):
    """
    Cross-validation that respects repetition boundaries.
    Returns fold scores.
    """
    if groups is not None:
        n_unique_groups = len(np.unique(groups))
        actual_splits = min(n_splits, n_unique_groups)

        if actual_splits < n_splits:
            print(f"[CV] Warning: Only {n_unique_groups} groups available, using {actual_splits} folds")

        cv = GroupKFold(n_splits=actual_splits)
        scores = cross_val_score(model, X, y, cv=cv, groups=groups)
    else:
        print("[CV] Warning: No groups provided, using StratifiedKFold (may leak data)")
        scores = cross_val_score(model, X, y, cv=n_splits)

    return scores


def cross_val_predictions(model, X, y, groups=None, n_splits=5):
    """
    Get cross-validated predictions (each sample predicted when in test fold).
    This gives honest predictions for confusion matrix / classification report.
    """
    if groups is not None:
        n_unique_groups = len(np.unique(groups))
        actual_splits = min(n_splits, n_unique_groups)
        cv = GroupKFold(n_splits=actual_splits)

        y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups, method='predict')
        y_prob = cross_val_predict(model, X, y, cv=cv, groups=groups, method='predict_proba')
    else:
        y_pred = cross_val_predict(model, X, y, cv=n_splits, method='predict')
        y_prob = cross_val_predict(model, X, y, cv=n_splits, method='predict_proba')

    return y_pred, y_prob


# ==============================================================================
# FEATURE CORRELATION MATRIX
# ==============================================================================
def feature_correlation(X):
    """Compute correlation matrix, handling feature count mismatch."""
    n_features = X.shape[1]

    if n_features != len(FEATURE_NAMES):
        print(f"[correlation] WARNING: X has {n_features} features, registry has {len(FEATURE_NAMES)}")
        print(f"[correlation] Using generic feature names")
        names = [f"F{i}" for i in range(n_features)]
    else:
        names = FEATURE_NAMES

    df = pd.DataFrame(X, columns=names)
    corr = df.corr(method='pearson')
    return corr


# ==============================================================================
# COHEN'S D EFFECT SIZE
# ==============================================================================
def cohens_d(x1, x2):
    """Compute Cohen's d effect size between two distributions."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan

    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(x1) - np.mean(x2)) / pooled_std


def effect_size_per_feature(X, y, class_a, class_b):
    """Compute Cohen's d for each feature between two classes."""
    results = {}
    mask_a = y == class_a
    mask_b = y == class_b

    n_features = X.shape[1]
    names = FEATURE_NAMES if n_features == len(FEATURE_NAMES) else [f"F{i}" for i in range(n_features)]

    for i, name in enumerate(names):
        feature_a = X[mask_a, i]
        feature_b = X[mask_b, i]
        d = cohens_d(feature_a, feature_b)
        results[name] = d

    return pd.Series(results)


# ==============================================================================
# SWIPE DISCRIMINATION ANALYSIS - FIXED
# ==============================================================================
def swipe_discrimination_report(X, y):
    """
    Detailed analysis of features that should discriminate swipe directions.
    """
    print("\n" + "=" * 60)
    print("SWIPE DISCRIMINATION ANALYSIS")
    print("=" * 60)

    swipe_pairs = [
        ("SWIPE_UP", "SWIPE_DOWN"),
        ("SWIPE_LEFT", "SWIPE_RIGHT"),
        ("SWIPE_UP", "SWIPE_RIGHT"),
        ("SWIPE_DOWN", "SWIPE_LEFT"),
    ]

    # Get indices for swipe-discriminating features
    swipe_feat_names = FEATURE_GROUPS.get("swipe_discriminators", [])

    # Filter to features that actually exist in registry
    swipe_feat_names = [f for f in swipe_feat_names if f in FEATURE_NAMES]

    if not swipe_feat_names:
        print("[swipe_analysis] No swipe discriminator features found in registry")
        return

    n_features = X.shape[1]
    if n_features != len(FEATURE_NAMES):
        print(f"[swipe_analysis] Feature count mismatch ({n_features} vs {len(FEATURE_NAMES)}), skipping")
        return

    swipe_indices = get_feature_indices(swipe_feat_names)

    print(f"[swipe_analysis] Analyzing {len(swipe_feat_names)} discriminator features:")
    print(f"  {swipe_feat_names[:5]}..." if len(swipe_feat_names) > 5 else f"  {swipe_feat_names}")

    for class_a, class_b in swipe_pairs:
        # Check if classes exist in y (handle both string and array)
        unique_classes = np.unique(y)
        if class_a not in unique_classes or class_b not in unique_classes:
            continue

        print(f"\n--- {class_a} vs {class_b} ---")

        mask_a = y == class_a
        mask_b = y == class_b

        print(f"  Samples: {mask_a.sum()} vs {mask_b.sum()}")

        # Effect sizes for swipe features
        effect_sizes = []
        for idx, name in zip(swipe_indices, swipe_feat_names):
            d = cohens_d(X[mask_a, idx], X[mask_b, idx])
            effect_sizes.append((name, d))

            # Also show means
            mean_a = X[mask_a, idx].mean()
            mean_b = X[mask_b, idx].mean()
            print(f"  {name:20s}: {class_a[:6]}={mean_a:+.4f}, {class_b[:6]}={mean_b:+.4f}, d={d:+.3f}")

        # Highlight best discriminators
        effect_sizes.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
        print(f"\n  Best discriminators:")
        for name, d in effect_sizes[:5]:
            if np.isnan(d):
                continue
            magnitude = "STRONG" if abs(d) > 0.8 else "MEDIUM" if abs(d) > 0.5 else "WEAK"
            print(f"    {name}: d={d:+.3f} ({magnitude})")


def plot_swipe_feature_distributions(X, y, figsize=(14, 10)):
    """
    Visualize key swipe-discriminating features across gesture classes.
    """
    n_features = X.shape[1]
    if n_features != len(FEATURE_NAMES):
        print("[plot_swipe] Feature count mismatch, skipping")
        return

    # Updated to use features that exist
    swipe_features = [
        "NORM_CH1", "NORM_CH2", "NORM_CH3", "NORM_CH4",
        "D13", "D24",
    ]

    # Filter to features that exist
    available = [f for f in swipe_features if f in FEATURE_NAMES]

    if not available:
        print("[plot_swipe] No swipe features available")
        return

    n_plots = len(available)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["Gesture"] = y

    # Filter to swipe gestures only
    swipe_gestures = ["SWIPE_UP", "SWIPE_DOWN", "SWIPE_LEFT", "SWIPE_RIGHT"]
    df_swipe = df[df["Gesture"].isin(swipe_gestures)]

    for i, feat in enumerate(available):
        ax = axes[i]
        sns.boxplot(
            data=df_swipe,
            x="Gesture",
            y=feat,
            hue="Gesture",
            palette="Set2",
            ax=ax,
            legend=False
        )
        ax.set_title(feat)
        ax.tick_params(axis='x', rotation=45)

    # Hide unused axes
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Swipe Discrimination Features", fontsize=14, fontweight='bold')
    plt.tight_layout()


def plot_normalized_ratio_scatter(X, y, figsize=(12, 5)):
    """
    Scatter plot of normalized channel ratios - should show directional separation.
    Replaces plot_envelope_timing_scatter since timing features were removed.
    """
    n_features = X.shape[1]
    if n_features != len(FEATURE_NAMES):
        print("[plot_ratio] Feature count mismatch, skipping")
        return

    # Check for required features
    required = ["NORM_CH1", "NORM_CH2", "NORM_CH3", "NORM_CH4"]
    missing = [f for f in required if f not in FEATURE_NAMES]
    if missing:
        print(f"[plot_ratio] Missing features: {missing}")
        return

    idx_1 = FEATURE_NAMES.index("NORM_CH1")
    idx_2 = FEATURE_NAMES.index("NORM_CH2")
    idx_3 = FEATURE_NAMES.index("NORM_CH3")
    idx_4 = FEATURE_NAMES.index("NORM_CH4")

    df = pd.DataFrame({
        "NORM_CH1 (Radial)": X[:, idx_1],
        "NORM_CH3 (Ulnar)": X[:, idx_3],
        "NORM_CH2 (Dorsal)": X[:, idx_2],
        "NORM_CH4 (Ventral)": X[:, idx_4],
        "Gesture": y
    })

    # Filter to swipes
    swipe_gestures = ["SWIPE_UP", "SWIPE_DOWN", "SWIPE_LEFT", "SWIPE_RIGHT"]
    df_swipe = df[df["Gesture"].isin(swipe_gestures)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: CH1 vs CH3 (Radial-Ulnar axis) - should separate LEFT/RIGHT
    sns.scatterplot(
        data=df_swipe,
        x="NORM_CH1 (Radial)",
        y="NORM_CH3 (Ulnar)",
        hue="Gesture",
        palette="Set1",
        s=60,
        alpha=0.7,
        ax=axes[0]
    )
    axes[0].set_title("Radial vs Ulnar (LEFT/RIGHT?)")
    axes[0].axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0.25, color='gray', linestyle='--', alpha=0.5)

    # Right: CH2 vs CH4 (Dorsal-Ventral axis) - should separate UP/DOWN
    sns.scatterplot(
        data=df_swipe,
        x="NORM_CH2 (Dorsal)",
        y="NORM_CH4 (Ventral)",
        hue="Gesture",
        palette="Set1",
        s=60,
        alpha=0.7,
        ax=axes[1]
    )
    axes[1].set_title("Dorsal vs Ventral (UP/DOWN?)")
    axes[1].axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.25, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()


# Keep old function name as alias for compatibility
def plot_envelope_timing_scatter(X, y, figsize=(12, 5)):
    """Deprecated: redirects to plot_normalized_ratio_scatter"""
    plot_normalized_ratio_scatter(X, y, figsize)


# ==============================================================================
# PEARSON CORRELATION
# ==============================================================================
def pearson(feature1, feature2):
    """Direct Pearson correlation between two feature vectors."""
    return pearsonr(feature1, feature2)


# ==============================================================================
# VISUALS
# ==============================================================================
def plot_correlation_heatmap(corr, figsize=(14, 12)):
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        vmin=-1, vmax=1
    )
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)
    plt.tight_layout()


def feature_boxplots(X, y, features_to_plot=None):
    """Plot boxplots for selected features."""
    n_features = X.shape[1]
    names = FEATURE_NAMES if n_features == len(FEATURE_NAMES) else [f"F{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=names)
    df["Gesture"] = y

    if features_to_plot is None:
        features_to_plot = [f for f in names if "RMS" in f][:4]

    for feature in features_to_plot:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="Gesture",
            y=feature,
            hue="Gesture",
            palette="Set2",
            legend=False
        )
        plt.title(f'{feature} Distribution by Gesture')
        plt.xticks(rotation=45)
        plt.tight_layout()


def pca_visuals(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Gesture": y
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Gesture",
        palette="Set1",
        s=60,
        alpha=0.7
    )
    plt.title(f'PCA Visualization (explained var: {pca.explained_variance_ratio_.sum():.2%})')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


def pca_swipe_only(X, y):
    """PCA visualization for swipe gestures only."""
    swipe_gestures = ["SWIPE_UP", "SWIPE_DOWN", "SWIPE_LEFT", "SWIPE_RIGHT"]
    mask = np.isin(y, swipe_gestures)

    if mask.sum() == 0:
        print("[pca_swipe] No swipe gestures found")
        return

    X_swipe = X[mask]
    y_swipe = y[mask]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_swipe)

    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Gesture": y_swipe
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Gesture",
        palette={"SWIPE_UP": "blue", "SWIPE_DOWN": "red",
                 "SWIPE_LEFT": "green", "SWIPE_RIGHT": "orange"},
        s=80,
        alpha=0.7
    )
    plt.title(f'PCA: Swipe Gestures Only ({pca.explained_variance_ratio_.sum():.1%} var)')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend()
    plt.tight_layout()


# ==============================================================================
# FEATURE IMPORTANCE (for pipeline with LDA)
# ==============================================================================
def lda_feature_importance(pipeline, feature_names=None):
    """
    Extract and visualize LDA coefficients as feature importance.
    Works for pipelines with 'lda' step.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    try:
        lda = pipeline.named_steps.get('lda')
        if lda is None:
            print("[lda_importance] No LDA step found in pipeline")
            return None

        # LDA coefficients: shape (n_classes-1, n_features) after scaling
        # Need to account for scaler if present
        scaler = pipeline.named_steps.get('scaler')

        coef = lda.coef_  # Shape: (n_classes, n_features) or (n_classes-1, n_features)

        # Average absolute coefficient across discriminant functions
        importance = np.mean(np.abs(coef), axis=0)

        if len(importance) != len(feature_names):
            print(f"[lda_importance] Coefficient count ({len(importance)}) != feature count ({len(feature_names)})")
            feature_names = [f"F{i}" for i in range(len(importance))]

        # Create DataFrame and sort
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        print("\n===== LDA Feature Importance (Top 15) =====")
        print(df.head(15).to_string(index=False))

        return df

    except Exception as e:
        print(f"[lda_importance] Error: {e}")
        return None


# ==============================================================================
# ML ANALYSIS ENGINE - UPDATED
# ==============================================================================
class MLAnalysis:
    """
    Comprehensive ML analysis with proper grouped cross-validation.

    All metrics (confusion matrix, classification report, ROC-AUC) are computed
    using cross-validated predictions to prevent overfitting artifacts.
    """

    def __init__(self, model, X, y, groups=None, analysis='Basic', n_splits=5):
        self.model = model
        self.X = X
        self.y = y
        self.groups = groups
        self.analysis = analysis
        self.n_splits = n_splits

        self.results = {}
        self._run_analysis()

    def _run_analysis(self):
        # Verify feature count
        n_features = self.X.shape[1]
        print(f"\n[MLAnalysis] Input: {self.X.shape[0]} samples × {n_features} features")
        print(f"[MLAnalysis] Feature registry: {len(FEATURE_NAMES)} names")

        if n_features != len(FEATURE_NAMES):
            print(f"[MLAnalysis] WARNING: Feature count mismatch!")

        # Cross Validation Scores
        print("\n===== Cross Validation (GroupKFold) =====")
        if self.groups is None:
            print("WARNING: No groups provided - CV scores may be inflated!")

        cv_scores = cross_validation_model(
            clone(self.model),
            self.X, self.y, self.groups, self.n_splits
        )
        self.results['cv_scores'] = cv_scores
        print(f"Fold scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
        print(f"Std CV accuracy:  {cv_scores.std():.4f}")

        # Get cross-validated predictions
        print("\n[Computing cross-validated predictions...]")
        self.y_pred, self.y_prob = cross_val_predictions(
            clone(self.model),
            self.X, self.y, self.groups, self.n_splits
        )

        # Confusion Matrix
        print("\n===== Confusion Matrix (CV Predictions) =====")
        cm = confusion_matrix(self.y, self.y_pred)
        self.results['confusion_matrix'] = cm
        print(cm)

        # ROC-AUC
        print("\n===== ROC-AUC Score (CV Predictions) =====")
        try:
            roc = roc_auc_score(self.y, self.y_prob, multi_class='ovr')
            self.results['roc_auc'] = roc
            print(f"ROC-AUC (OvR): {roc:.4f}")
        except Exception as e:
            print(f"ROC-AUC failed: {e}")
            self.results['roc_auc'] = None

        # Classification Report
        print("\n===== Classification Report (CV Predictions) =====")
        cr = classification_report(self.y, self.y_pred)
        self.results['classification_report'] = cr
        print(cr)

        # Verification
        cv_acc = np.mean(self.y_pred == self.y)
        print(f"\n[Verification] CV prediction accuracy: {cv_acc:.4f}")

        # Feature Correlation
        print("\n===== Feature Correlation =====")
        corr = feature_correlation(self.X)
        self.results['correlation'] = corr
        print("(See heatmap visualization)")

        # Swipe discrimination analysis (always run for swipe debugging)
        swipe_discrimination_report(self.X, self.y)

        # Effect Size (Full analysis)
        # Effect Size (Full analysis)
        if self.analysis == "Full":
            unique_labels = set(np.unique(self.y))

            if "REST" in unique_labels and "FIST" in unique_labels:
                print("\n===== Effect Size: REST vs FIST =====")
                espf = effect_size_per_feature(self.X, self.y, "REST", "FIST")
                self.results['effect_size_rest_fist'] = espf
                print(espf.sort_values(key=abs, ascending=False).head(10))
            else:
                print("\n[Skipping REST vs FIST effect size - REST not in training data]")

            if "SWIPE_UP" in unique_labels and "SWIPE_DOWN" in unique_labels:
                print("\n===== Effect Size: SWIPE_UP vs SWIPE_DOWN =====")
                espf_swipe = effect_size_per_feature(self.X, self.y, "SWIPE_UP", "SWIPE_DOWN")
                self.results['effect_size_up_down'] = espf_swipe
                print(espf_swipe.sort_values(key=abs, ascending=False).head(10))

        # LDA feature importance
        try:
            importance_df = lda_feature_importance(self.model)
            if importance_df is not None:
                self.results['feature_importance'] = importance_df
        except Exception as e:
            print(f"[MLAnalysis] Feature importance failed: {e}")

        self._plot_all()

    def _plot_all(self):
        """Generate all visualizations."""
        pca_visuals(self.X, self.y)
        pca_swipe_only(self.X, self.y)
        plot_correlation_heatmap(self.results['correlation'])
        feature_boxplots(self.X, self.y)

        # Swipe-specific plots
        plot_swipe_feature_distributions(self.X, self.y)
        plot_normalized_ratio_scatter(self.X, self.y)

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            self.results['confusion_matrix'],
            display_labels=np.unique(self.y)
        )
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix (Cross-Validated)')
        plt.tight_layout()

    def show(self):
        """Display all plots."""
        plt.show()

    def save_report(self, path="analysis_report.txt"):
        """Save text report to file."""
        with open(path, 'w') as f:
            f.write("ML Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Features: {self.X.shape[1]}\n")
            f.write(f"Samples: {self.X.shape[0]}\n\n")

            f.write("Cross Validation (GroupKFold):\n")
            f.write(f"  Scores: {self.results['cv_scores']}\n")
            f.write(f"  Mean: {self.results['cv_scores'].mean():.4f}\n")
            f.write(f"  Std:  {self.results['cv_scores'].std():.4f}\n\n")

            f.write("ROC-AUC (CV): ")
            f.write(f"{self.results['roc_auc']:.4f}\n\n" if self.results['roc_auc'] else "N/A\n\n")

            f.write("Classification Report (CV):\n")
            f.write(self.results['classification_report'])

            # Add swipe-specific notes
            f.write("\n\nSwipe Discrimination Notes:\n")
            f.write("Check plots for NORM_CH* and SLOPE* feature separation.\n")

        print(f"Report saved to {path}")


# ==============================================================================
# QUICK DIAGNOSTIC FUNCTION
# ==============================================================================
def quick_swipe_diagnostic(X, y):
    """
    Run just the swipe discrimination analysis without full ML pipeline.
    Useful for quick feature debugging.
    """
    print("=" * 60)
    print("QUICK SWIPE DIAGNOSTIC")
    print("=" * 60)

    swipe_discrimination_report(X, y)
    plot_swipe_feature_distributions(X, y)
    plot_normalized_ratio_scatter(X, y)
    pca_swipe_only(X, y)

    plt.show()
