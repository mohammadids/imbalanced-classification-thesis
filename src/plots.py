from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


CLASSIFIER_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

TREATMENT_DISPLAY = {
    "original": "Original",
    "random_oversampling": "Random Oversampling",
    "random_undersampling": "Random Undersampling",
    "smote": "SMOTE",
    "class_weighting": "Class Weighting",
}

DATASET_DISPLAY = {
    "satimage": "Satimage",
    "ozone_level": "Ozone Level",
    "abalone_19": "Abalone-19",
}

METRIC_DISPLAY = {
    "cv_average_precision_mean": "Mean CV Average Precision",
    "cv_f1_mean": "Mean CV F1",
    "cv_recall_mean": "Mean CV Minority Recall",
    "cv_balanced_accuracy_mean": "Mean CV Balanced Accuracy",
    "cv_geometric_mean_mean": "Mean CV Geometric Mean",
    "average_precision": "Test Average Precision",
    "f1": "Test F1",
    "recall": "Test Minority Recall",
    "balanced_accuracy": "Test Balanced Accuracy",
    "geometric_mean": "Test Geometric Mean",
}

METRIC_STEMS = {
    "cv_average_precision_mean": "average_precision",
    "cv_f1_mean": "f1",
    "cv_recall_mean": "recall",
    "cv_balanced_accuracy_mean": "balanced_accuracy",
    "cv_geometric_mean_mean": "geometric_mean",
}


def display_classifier(value: str) -> str:
    return CLASSIFIER_DISPLAY.get(value, value)


def display_treatment(value: str) -> str:
    return TREATMENT_DISPLAY.get(value, value)


def display_config(classifier: str, treatment: str) -> str:
    return f"{display_classifier(classifier)} + {display_treatment(treatment)}"


def display_dataset(value: str) -> str:
    return DATASET_DISPLAY.get(value, value)


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _ordered_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    order = ["satimage", "ozone_level", "abalone_19"]
    return df.set_index("dataset").loc[order].reset_index()


def plot_class_distribution(dataset_characteristics: pd.DataFrame, output_dir: Path) -> list[Path]:
    df = _ordered_dataset_frame(dataset_characteristics)
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, df["majority_class_count"], width, label="Majority class", color="#4c78a8")
    ax.bar(x + width / 2, df["minority_class_count"], width, label="Minority class", color="#f58518")
    ax.set_yscale("log")
    ax.set_title("Class Distribution by Dataset")
    ax.set_ylabel("Sample count (log scale)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x, df["dataset"])
    ax.legend()
    for idx, row in df.iterrows():
        ax.text(idx - width / 2, row["majority_class_count"], f"{int(row['majority_class_count'])}", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, row["minority_class_count"], f"{int(row['minority_class_count'])}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, "dataset_class_distribution")


def plot_minority_percentage(dataset_characteristics: pd.DataFrame, output_dir: Path) -> list[Path]:
    df = _ordered_dataset_frame(dataset_characteristics)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(df["dataset"], df["minority_class_percentage"], color="#54a24b")
    ax.set_title("Minority-Class Percentage by Dataset")
    ax.set_ylabel("Minority class (%)")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, max(df["minority_class_percentage"]) * 1.25)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, output_dir, "dataset_minority_percentage")


def plot_imbalance_ratio(dataset_characteristics: pd.DataFrame, output_dir: Path) -> list[Path]:
    df = _ordered_dataset_frame(dataset_characteristics)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(df["dataset"], df["imbalance_ratio"], color="#b279a2")
    ax.set_yscale("log")
    ax.set_title("Imbalance Ratio by Dataset")
    ax.set_ylabel("Majority-to-minority ratio (log scale)")
    ax.set_xlabel("Dataset")
    ax.bar_label(bars, labels=[f"{value:.2f}:1" for value in df["imbalance_ratio"]], padding=3, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, output_dir, "dataset_imbalance_ratio")


def highest_variance_features(X: pd.DataFrame, max_features: int = 4) -> list[str]:
    return X.var(numeric_only=True).sort_values(ascending=False).head(max_features).index.tolist()


def plot_feature_distributions(bundles: dict, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, dataset in zip(axes, ["satimage", "ozone_level", "abalone_19"]):
        bundle = bundles[dataset]
        features = highest_variance_features(bundle.X, max_features=4)
        for feature in features:
            values = bundle.X[feature]
            standardized = (values - values.mean()) / values.std(ddof=0)
            ax.hist(standardized, bins=30, alpha=0.42, density=True, label=feature)
        ax.set_title(f"{dataset}: Highest-Variance Feature Distributions")
        ax.set_xlabel("Standardized value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("Descriptive Feature Distributions (No Feature Selection Performed)", fontsize=13)
    return save_figure(fig, output_dir, "dataset_feature_distributions")


def plot_correlation_overview(bundles: dict, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, dataset in zip(axes, ["satimage", "ozone_level", "abalone_19"]):
        bundle = bundles[dataset]
        features = highest_variance_features(bundle.X, max_features=10)
        corr = bundle.X[features].corr()
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{dataset}: Correlation Overview")
        ax.set_xticks(range(len(features)), features, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(features)), features, fontsize=7)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Pearson correlation")
    fig.suptitle("Correlation Overview for Deterministically Selected High-Variance Features", fontsize=13)
    return save_figure(fig, output_dir, "dataset_correlation_overview")


def plot_cv_heatmap(cv_results: pd.DataFrame, dataset: str, metric: str, output_dir: Path) -> list[Path]:
    subset = cv_results.loc[cv_results["dataset"] == dataset].copy()
    subset["Classifier"] = subset["classifier"].map(display_classifier)
    subset["Treatment"] = subset["treatment"].map(display_treatment)
    row_order = [display_classifier(name) for name in ("logistic_regression", "random_forest", "xgboost")]
    col_order = [display_treatment(name) for name in ("original", "random_oversampling", "random_undersampling", "smote", "class_weighting")]
    pivot = subset.pivot(index="Classifier", columns="Treatment", values=metric).loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{dataset}: {METRIC_DISPLAY[metric]} (Cross-Validation)")
    ax.set_xlabel("Imbalance treatment")
    ax.set_ylabel("Classifier")
    ax.set_xticks(range(len(col_order)), col_order, rotation=35, ha="right")
    ax.set_yticks(range(len(row_order)), row_order)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label=METRIC_DISPLAY[metric])
    fig.tight_layout()
    stem = f"cv_heatmap_{dataset}_{METRIC_STEMS[metric]}"
    return save_figure(fig, output_dir, stem)


def plot_confusion_matrix(row: pd.Series, output_dir: Path) -> list[Path]:
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0) * 100
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    configuration = display_config(row["classifier"], row["treatment"]).replace(" + ", " with ")
    ax.set_title(f"{display_dataset(row['dataset'])}\n{configuration}", fontsize=12, pad=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["Majority (0)", "Minority (1)"])
    ax.set_yticks([0, 1], ["Majority (0)", "Minority (1)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matrix[i, j])}\n{percentages[i, j]:.1f}%", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, label="Count", pad=0.08, fraction=0.046)
    return save_figure(fig, output_dir, f"selected_{row['dataset']}_confusion_matrix")


def plot_precision_recall_curve(predictions: pd.DataFrame, label: str, title: str, output_dir: Path, stem: str) -> list[Path]:
    prevalence = float(predictions["y_true"].mean())
    fig, ax = plt.subplots(figsize=(6.4, 5))
    PrecisionRecallDisplay.from_predictions(predictions["y_true"], predictions["y_score"], name=label, ax=ax)
    ax.axhline(prevalence, color="#777777", linestyle="--", label=f"No-skill baseline ({prevalence:.3f})")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def plot_roc_curve(predictions: pd.DataFrame, label: str, title: str, output_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    RocCurveDisplay.from_predictions(predictions["y_true"], predictions["y_score"], name=label, ax=ax)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="No-skill baseline")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def plot_curve_comparison(prediction_frames: list[tuple[pd.DataFrame, str]], curve_type: str, title: str, output_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for predictions, label in prediction_frames:
        if curve_type == "precision_recall":
            PrecisionRecallDisplay.from_predictions(predictions["y_true"], predictions["y_score"], name=label, ax=ax)
        elif curve_type == "roc":
            RocCurveDisplay.from_predictions(predictions["y_true"], predictions["y_score"], name=label, ax=ax)
        else:
            raise ValueError(f"Unsupported curve type {curve_type!r}")
    if curve_type == "precision_recall":
        prevalence = float(prediction_frames[0][0]["y_true"].mean())
        ax.axhline(prevalence, color="#777777", linestyle="--", label=f"No-skill baseline ({prevalence:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    else:
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="No-skill baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def plot_selected_model_performance(best: pd.DataFrame, output_dir: Path) -> list[Path]:
    metrics = ["average_precision", "f1", "recall", "balanced_accuracy", "geometric_mean"]
    labels = [METRIC_DISPLAY[metric].replace("Test ", "") for metric in metrics]
    x = np.arange(len(best))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, best[metric], width, label=labels[idx])
    ax.set_title("CV-Selected Configuration Test Performance")
    ax.set_ylabel("Test metric value")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x, best["dataset"])
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, "selected_model_test_performance")


def plot_original_vs_selected(original_vs_selected: pd.DataFrame, output_dir: Path) -> list[Path]:
    metrics = ["test_average_precision", "test_f1", "test_recall", "test_balanced_accuracy", "test_geometric_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, ["satimage", "ozone_level", "abalone_19"]):
        subset = original_vs_selected.loc[original_vs_selected["dataset"] == dataset]
        x = np.arange(len(metrics))
        width = 0.36
        original = subset.loc[subset["comparison"] == "Original same classifier"].iloc[0]
        selected = subset.loc[subset["comparison"] == "CV-selected"].iloc[0]
        ax.bar(x - width / 2, [original[m] for m in metrics], width, label="Original")
        ax.bar(x + width / 2, [selected[m] for m in metrics], width, label="CV-selected")
        ax.set_title(dataset)
        ax.set_xticks(x, ["AP", "F1", "Recall", "Bal. Acc.", "G-Mean"], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test metric value")
        ax.legend(fontsize=8)
    fig.suptitle("Original Versus CV-Selected Configuration")
    return save_figure(fig, output_dir, "original_vs_selected")


def plot_overall_mean_rank(overall: pd.DataFrame, output_dir: Path) -> list[Path]:
    df = overall.copy()
    df["configuration"] = df.apply(lambda row: display_config(row["classifier"], row["treatment"]), axis=1)
    df = df.sort_values("mean_within_dataset_rank", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["configuration"], df["mean_within_dataset_rank"], color="#4c78a8")
    ax.invert_yaxis()
    ax.set_title("Overall Classifier-Treatment Ranking by Mean Within-Dataset Rank")
    ax.set_xlabel("Mean within-dataset rank (lower is better)")
    ax.set_ylabel("Configuration")
    fig.tight_layout()
    return save_figure(fig, output_dir, "overall_mean_rank")


def plot_mean_cv_performance(summary: pd.DataFrame, group_column: str, output_dir: Path, stem: str, title: str) -> list[Path]:
    metrics = [
        "cv_average_precision_mean",
        "cv_f1_mean",
        "cv_recall_mean",
        "cv_balanced_accuracy_mean",
        "cv_geometric_mean_mean",
    ]
    labels = ["AP", "F1", "Recall", "Bal. Acc.", "G-Mean"]
    df = summary.copy()
    if group_column == "classifier":
        df[group_column] = df[group_column].map(display_classifier)
    elif group_column == "treatment":
        df[group_column] = df[group_column].map(display_treatment)
    x = np.arange(len(df))
    width = 0.14
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, df[metric], width, label=labels[idx])
    ax.set_title(title)
    ax.set_ylabel("Mean cross-validation metric")
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_xticks(x, df[group_column], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)
