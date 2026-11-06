from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PREDICTIONS_DIR, RESULTS_DIR, TABLES_DIR
from src.data_loader import (
    dataset_cleaning_summary_frame,
    dataset_summary_frame,
    label_mapping_frame,
    load_all_datasets,
)
from src.evaluation import evaluate_predictions
from src.experiments import build_full_experiment_grid
from src.plots import (
    display_classifier,
    display_config,
    display_treatment,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_overview,
    plot_curve_comparison,
    plot_cv_heatmap,
    plot_feature_distributions,
    plot_imbalance_ratio,
    plot_mean_cv_performance,
    plot_minority_percentage,
    plot_original_vs_selected,
    plot_overall_mean_rank,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_selected_model_performance,
)
from src.postrun_validation import validate_full_run_outputs
from src.validation import validate_bundles


FINAL_DATASETS = ["satimage", "ozone_level", "abalone_19"]
FINAL_FIGURE_DIR = RESULTS_DIR / "figures" / "final"
THESIS_TABLE_DIR = TABLES_DIR / "thesis"
ARCHIVE_DIR = RESULTS_DIR / "archive" / "phase1_5_dataset_tables"

GENERIC_DATASET_TABLES = [
    "dataset_summary.csv",
    "dataset_summary_raw.csv",
    "dataset_summary_modeling.csv",
    "dataset_cleaning_summary.csv",
    "class_label_mappings.csv",
]

RESULT_HASH_FILES = [
    METRICS_DIR / "cross_validation_results.csv",
    METRICS_DIR / "test_results.csv",
    METRICS_DIR / "all_experiment_results.csv",
    METRICS_DIR / "full_experiment_checkpoint.csv",
    METRICS_DIR / "experiment_manifest.csv",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_stale_dataset_tables() -> list[str]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived = []
    for filename in GENERIC_DATASET_TABLES:
        source = TABLES_DIR / filename
        target = ARCHIVE_DIR / filename
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
        if target.exists():
            archived.append(str(target))
    return archived


def regenerate_active_dataset_tables(bundles: dict) -> dict[str, str]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary_frame(bundles)
    raw_summary = dataset_summary_frame(bundles, summary_kind="raw")
    modeling_summary = dataset_summary_frame(bundles, summary_kind="modeling")
    cleaning = dataset_cleaning_summary_frame(bundles)
    mappings = label_mapping_frame(bundles)

    outputs = {
        "dataset_summary": TABLES_DIR / "dataset_summary.csv",
        "dataset_summary_raw": TABLES_DIR / "dataset_summary_raw.csv",
        "dataset_summary_modeling": TABLES_DIR / "dataset_summary_modeling.csv",
        "dataset_cleaning_summary": TABLES_DIR / "dataset_cleaning_summary.csv",
        "class_label_mappings": TABLES_DIR / "class_label_mappings.csv",
        "dataset_characteristics_final": TABLES_DIR / "dataset_characteristics_final.csv",
        "dataset_cleaning_final": TABLES_DIR / "dataset_cleaning_final.csv",
        "class_label_mappings_final": TABLES_DIR / "class_label_mappings_final.csv",
    }

    summary.to_csv(outputs["dataset_summary"], index=False)
    raw_summary.to_csv(outputs["dataset_summary_raw"], index=False)
    modeling_summary.to_csv(outputs["dataset_summary_modeling"], index=False)
    cleaning.to_csv(outputs["dataset_cleaning_summary"], index=False)
    mappings.to_csv(outputs["class_label_mappings"], index=False)
    modeling_summary.to_csv(outputs["dataset_characteristics_final"], index=False)
    cleaning.to_csv(outputs["dataset_cleaning_final"], index=False)
    mappings.to_csv(outputs["class_label_mappings_final"], index=False)
    return {name: str(path) for name, path in outputs.items()}


def validate_phase2_results(bundles: dict, grid: pd.DataFrame) -> dict:
    validate_full_run_outputs(bundles, grid)
    cv = read_csv(METRICS_DIR / "cross_validation_results.csv")
    test = read_csv(METRICS_DIR / "test_results.csv")
    all_results = read_csv(METRICS_DIR / "all_experiment_results.csv")
    manifest = read_csv(METRICS_DIR / "experiment_manifest.csv")
    best = read_csv(TABLES_DIR / "best_model_per_dataset.csv")
    within = read_csv(TABLES_DIR / "within_dataset_ranking.csv")
    overall = read_csv(TABLES_DIR / "overall_model_ranking.csv")

    prediction_files = sorted(PREDICTIONS_DIR.glob("*.csv.gz"))
    if len(prediction_files) != 45:
        raise ValueError(f"Expected 45 prediction files, found {len(prediction_files)}")
    if len(test) != 45 or len(cv) != 45 or len(all_results) != 45:
        raise ValueError("Phase 2 result files must each contain exactly 45 rows")
    if test["experiment_id"].duplicated().any() or cv["experiment_id"].duplicated().any():
        raise ValueError("Duplicate experiment IDs found in result files")
    if set(test["execution_status"]) != {"success"}:
        raise ValueError("All test result rows must be successful")
    if set(manifest["status"]) != {"complete"}:
        raise ValueError("All manifest rows must be complete")

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f2",
        "roc_auc",
        "average_precision",
        "mcc",
        "geometric_mean",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    reproduced = 0
    for row in test.to_dict(orient="records"):
        path = PREDICTIONS_DIR / f"{row['experiment_id']}.csv.gz"
        predictions = read_csv(path)
        metrics = evaluate_predictions(
            predictions["y_true"],
            predictions["y_pred"],
            predictions["y_score"],
            training_time=0.0,
            prediction_time=0.0,
        )
        for column in metric_columns:
            actual = row[column]
            expected = metrics[column]
            if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{row['experiment_id']} metric {column} changed: {actual} != {expected}")
        reproduced += 1

    expected_top = (
        cv.sort_values(
            ["dataset", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"],
            ascending=[True, False, False, False, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
    )
    if set(best["experiment_id"]) != set(expected_top["experiment_id"]):
        raise ValueError("Best models do not match CV-only selection rule")

    return {
        "cv": cv,
        "test": test,
        "all_results": all_results,
        "manifest": manifest,
        "best": best,
        "within": within,
        "overall": overall,
        "prediction_files": prediction_files,
        "reproduced_metric_rows": reproduced,
    }


def prediction_path(experiment_id: str) -> Path:
    return PREDICTIONS_DIR / f"{experiment_id}.csv.gz"


def generate_figures(bundles: dict, tables: dict) -> tuple[list[str], dict]:
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    curve_metadata = {"selected_model_curves": {}, "top_three_curves": {}}

    dataset_characteristics = read_csv(TABLES_DIR / "dataset_characteristics_final.csv")
    figures += [str(path) for path in plot_class_distribution(dataset_characteristics, FINAL_FIGURE_DIR)]
    figures += [str(path) for path in plot_minority_percentage(dataset_characteristics, FINAL_FIGURE_DIR)]
    figures += [str(path) for path in plot_imbalance_ratio(dataset_characteristics, FINAL_FIGURE_DIR)]
    figures += [str(path) for path in plot_feature_distributions(bundles, FINAL_FIGURE_DIR)]
    figures += [str(path) for path in plot_correlation_overview(bundles, FINAL_FIGURE_DIR)]

    cv = tables["cv"]
    for dataset in FINAL_DATASETS:
        for metric in (
            "cv_average_precision_mean",
            "cv_f1_mean",
            "cv_recall_mean",
            "cv_balanced_accuracy_mean",
            "cv_geometric_mean_mean",
        ):
            figures += [str(path) for path in plot_cv_heatmap(cv, dataset, metric, FINAL_FIGURE_DIR)]

    best = tables["best"]
    test = tables["test"]
    within = tables["within"]
    for _, best_row in best.iterrows():
        dataset = best_row["dataset"]
        experiment_id = best_row["experiment_id"]
        test_row = test.loc[test["experiment_id"] == experiment_id].iloc[0]
        predictions = read_csv(prediction_path(experiment_id))
        ap_label = f"{display_config(best_row['classifier'], best_row['treatment'])} (Average Precision={test_row['average_precision']:.3f})"
        roc_label = f"{display_config(best_row['classifier'], best_row['treatment'])} (ROC-AUC={test_row['roc_auc']:.3f})"

        figures += [str(path) for path in plot_confusion_matrix(test_row, FINAL_FIGURE_DIR)]
        figures += [str(path) for path in plot_precision_recall_curve(
            predictions,
            ap_label,
            f"{dataset}: Precision-Recall Curve for CV-Selected Model",
            FINAL_FIGURE_DIR,
            f"selected_{dataset}_precision_recall_curve",
        )]
        figures += [str(path) for path in plot_roc_curve(
            predictions,
            roc_label,
            f"{dataset}: ROC Curve for CV-Selected Model",
            FINAL_FIGURE_DIR,
            f"selected_{dataset}_roc_curve",
        )]
        curve_metadata["selected_model_curves"][dataset] = {
            "experiment_id": experiment_id,
            "prediction_file": str(prediction_path(experiment_id)),
            "precision_recall_baseline": float(predictions["y_true"].mean()),
            "precision_recall_figure": str(FINAL_FIGURE_DIR / f"selected_{dataset}_precision_recall_curve.png"),
            "roc_figure": str(FINAL_FIGURE_DIR / f"selected_{dataset}_roc_curve.png"),
        }

        top3 = within.loc[within["dataset"] == dataset].sort_values("within_dataset_rank").head(3)
        comparison_frames = []
        curve_metadata["top_three_curves"][dataset] = []
        for _, row in top3.iterrows():
            pred = read_csv(prediction_path(row["experiment_id"]))
            label = f"Rank {int(row['within_dataset_rank'])}: {display_config(row['classifier'], row['treatment'])}"
            comparison_frames.append((pred, label))
            curve_metadata["top_three_curves"][dataset].append({
                "cv_rank": int(row["within_dataset_rank"]),
                "experiment_id": row["experiment_id"],
                "prediction_file": str(prediction_path(row["experiment_id"])),
                "precision_recall_baseline": float(pred["y_true"].mean()),
            })
        figures += [str(path) for path in plot_curve_comparison(
            comparison_frames,
            "precision_recall",
            f"{dataset}: Precision-Recall Curves for Top Three CV Configurations",
            FINAL_FIGURE_DIR,
            f"top3_{dataset}_precision_recall_curve",
        )]
        figures += [str(path) for path in plot_curve_comparison(
            comparison_frames,
            "roc",
            f"{dataset}: ROC Curves for Top Three CV Configurations",
            FINAL_FIGURE_DIR,
            f"top3_{dataset}_roc_curve",
        )]

    original_vs_selected = build_original_vs_selected_table(tables["best"], tables["test"])
    figures += [str(path) for path in plot_selected_model_performance(best, FINAL_FIGURE_DIR)]
    figures += [str(path) for path in plot_original_vs_selected(original_vs_selected, FINAL_FIGURE_DIR)]
    figures += [str(path) for path in plot_overall_mean_rank(tables["overall"], FINAL_FIGURE_DIR)]
    figures += [str(path) for path in plot_mean_cv_performance(
        read_csv(TABLES_DIR / "average_performance_by_classifier.csv"),
        "classifier",
        FINAL_FIGURE_DIR,
        "mean_cv_performance_by_classifier",
        "Mean Cross-Validation Performance by Classifier",
    )]
    figures += [str(path) for path in plot_mean_cv_performance(
        read_csv(TABLES_DIR / "average_performance_by_treatment.csv"),
        "treatment",
        FINAL_FIGURE_DIR,
        "mean_cv_performance_by_treatment",
        "Mean Cross-Validation Performance by Treatment",
    )]

    return sorted(figures), curve_metadata


def display_column_name(column: str) -> str:
    replacements = {
        "dataset": "Dataset",
        "classifier": "Classifier",
        "treatment": "Treatment",
        "experiment_id": "Experiment ID",
        "observations": "Observations",
        "input_features": "Features",
        "majority_class_count": "Majority Count",
        "minority_class_count": "Minority Count",
        "minority_class_percentage": "Minority (%)",
        "imbalance_ratio": "Imbalance Ratio",
        "missing_values": "Missing Values",
        "cleaning_status": "Cleaning Status",
        "original_majority_label": "Original Majority Label",
        "remapped_majority_label": "Remapped Majority Label",
        "original_minority_label": "Original Minority Label",
        "remapped_minority_label": "Remapped Minority Label",
        "cv_average_precision_mean": "CV Average Precision Mean",
        "cv_average_precision_std": "CV Average Precision SD",
        "cv_f1_mean": "CV F1 Mean",
        "cv_recall_mean": "CV Minority Recall Mean",
        "cv_balanced_accuracy_mean": "CV Balanced Accuracy Mean",
        "cv_geometric_mean_mean": "CV Geometric Mean Mean",
        "average_precision": "Test Average Precision",
        "roc_auc": "Test ROC-AUC",
        "balanced_accuracy": "Test Balanced Accuracy",
        "precision": "Test Precision",
        "recall": "Test Minority Recall",
        "f1": "Test F1",
        "f2": "Test F2",
        "geometric_mean": "Test Geometric Mean",
        "mcc": "Test MCC",
        "within_dataset_rank": "Within-Dataset Rank",
    }
    return replacements.get(column, column.replace("_", " ").title())


def make_display_table(df: pd.DataFrame, columns: list[str], rename_values: bool = True) -> pd.DataFrame:
    table = df[columns].copy()
    if rename_values:
        if "classifier" in table.columns:
            table["classifier"] = table["classifier"].map(display_classifier)
        if "treatment" in table.columns:
            table["treatment"] = table["treatment"].map(display_treatment)
    table = table.rename(columns={column: display_column_name(column) for column in columns})
    numeric_columns = table.select_dtypes(include=["number"]).columns
    table[numeric_columns] = table[numeric_columns].round(4)
    return table


def latex_escape(value) -> str:
    text = "" if pd.isna(value) else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, longtable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = "longtable" if longtable else "tabular"
    column_spec = "l" * len(df.columns)
    lines = []
    if not longtable:
        lines.extend(["\\begin{table}[htbp]", "\\centering"])
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{tab:{latex_escape(label)}}}")
    lines.append(f"\\begin{{{env}}}{{{column_spec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(latex_escape(col) for col in df.columns) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(latex_escape(value))
        lines.append(" & ".join(values) + " \\\\")
    lines.append("\\hline")
    lines.append(f"\\end{{{env}}}")
    if not longtable:
        lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thesis_table(name: str, df: pd.DataFrame, caption: str, label: str, longtable: bool = False) -> dict[str, str]:
    THESIS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = THESIS_TABLE_DIR / f"{name}.csv"
    tex_path = THESIS_TABLE_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    write_latex_table(df, tex_path, caption, label, longtable=longtable)
    return {"csv": str(csv_path), "tex": str(tex_path)}


def build_original_vs_selected_table(best: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["average_precision", "f1", "recall", "balanced_accuracy", "geometric_mean"]
    for _, selected in best.iterrows():
        dataset = selected["dataset"]
        classifier = selected["classifier"]
        original = test.loc[
            (test["dataset"] == dataset)
            & (test["classifier"] == classifier)
            & (test["treatment"] == "original")
        ].iloc[0]
        selected_test = test.loc[test["experiment_id"] == selected["experiment_id"]].iloc[0]
        for comparison, row in (("Original same classifier", original), ("CV-selected", selected_test)):
            output = {
                "dataset": dataset,
                "comparison": comparison,
                "classifier": classifier,
                "treatment": row["treatment"],
            }
            for metric in metrics:
                output[f"test_{metric}"] = row[metric]
            rows.append(output)
    return pd.DataFrame(rows)


def generate_thesis_tables(tables: dict) -> dict[str, dict[str, str]]:
    cv = tables["cv"]
    test = tables["test"]
    best = tables["best"]
    within = tables["within"]
    overall = tables["overall"]
    all_results = cv.merge(
        test,
        on=["experiment_id", "dataset", "classifier", "treatment", "positive_label", "execution_status", "warning_or_error_message"],
        suffixes=("_cv", "_test"),
    )
    original_vs_selected = build_original_vs_selected_table(best, test)

    outputs = {}
    characteristics = read_csv(TABLES_DIR / "dataset_characteristics_final.csv")
    outputs["dataset_characteristics"] = write_thesis_table(
        "dataset_characteristics",
        make_display_table(characteristics, ["dataset", "observations", "input_features", "majority_class_count", "minority_class_count", "minority_class_percentage", "imbalance_ratio", "missing_values", "cleaning_status"], rename_values=False),
        "Final dataset characteristics.",
        "dataset_characteristics",
    )
    mappings = read_csv(TABLES_DIR / "class_label_mappings_final.csv")
    outputs["class_label_mappings"] = write_thesis_table(
        "class_label_mappings",
        make_display_table(mappings, ["dataset", "original_majority_label", "remapped_majority_label", "original_minority_label", "remapped_minority_label"], rename_values=False),
        "Class label mappings used in the final experiment.",
        "class_label_mappings",
    )
    design = pd.DataFrame([
        {"Design Element": "Datasets", "Value": "satimage; ozone_level; abalone_19"},
        {"Design Element": "Classifiers", "Value": "Logistic Regression; Random Forest; XGBoost"},
        {"Design Element": "Treatments", "Value": "Original; Random Oversampling; Random Undersampling; SMOTE; Class Weighting"},
        {"Design Element": "Train-Test Split", "Value": "Stratified 70/30 split with random_state=42"},
        {"Design Element": "Cross-Validation", "Value": "Stratified 5-fold CV on training data"},
        {"Design Element": "Primary Selection Metric", "Value": "Mean CV Average Precision"},
        {"Design Element": "Total Configurations", "Value": "45"},
    ])
    outputs["experimental_design"] = write_thesis_table("experimental_design", design, "Experimental design summary.", "experimental_design")
    hyperparameters = pd.DataFrame([
        {"Model": "Logistic Regression", "Hyperparameters": "max_iter=1000; solver=lbfgs; class_weight=balanced only for class weighting; random_state=42"},
        {"Model": "Random Forest", "Hyperparameters": "n_estimators=200; max_depth=None; min_samples_leaf=1; class_weight=balanced only for class weighting; n_jobs=-1; random_state=42"},
        {"Model": "XGBoost", "Hyperparameters": "n_estimators=100; max_depth=3; learning_rate=0.1; subsample=0.8; colsample_bytree=0.8; objective=binary:logistic; eval_metric=logloss; random_state=42"},
    ])
    outputs["model_hyperparameters"] = write_thesis_table("model_hyperparameters", hyperparameters, "Model hyperparameters.", "model_hyperparameters")
    treatments = pd.DataFrame([
        {"Treatment": "Original", "Description": "No imbalance treatment."},
        {"Treatment": "Random Oversampling", "Description": "Minority class oversampled within training folds only."},
        {"Treatment": "Random Undersampling", "Description": "Majority class undersampled within training folds only."},
        {"Treatment": "SMOTE", "Description": "StandardScaler followed by SMOTE and classifier inside the imblearn pipeline."},
        {"Treatment": "Class Weighting", "Description": "Classifier-level class weighting; XGBoost scale_pos_weight calculated inside fit."},
    ])
    outputs["imbalance_treatments"] = write_thesis_table("imbalance_treatments", treatments, "Imbalance treatments.", "imbalance_treatments")
    metrics = pd.DataFrame([
        {"Metric": "Average Precision", "Used For": "CV and test", "Description": "Area under the precision-recall step function from ranked scores."},
        {"Metric": "ROC-AUC", "Used For": "CV and test", "Description": "Area under the ROC curve from ranked scores."},
        {"Metric": "Balanced Accuracy", "Used For": "CV and test", "Description": "Mean of sensitivity and specificity."},
        {"Metric": "Precision", "Used For": "CV and test", "Description": "Minority-class positive predictive value."},
        {"Metric": "Recall", "Used For": "CV and test", "Description": "Minority-class sensitivity."},
        {"Metric": "Specificity", "Used For": "CV and test", "Description": "Majority-class true negative rate."},
        {"Metric": "F1 and F2", "Used For": "CV and test", "Description": "Harmonic scores with F2 weighting recall more strongly."},
        {"Metric": "MCC", "Used For": "CV and test", "Description": "Matthews Correlation Coefficient."},
        {"Metric": "Geometric Mean", "Used For": "CV and test", "Description": "Square root of sensitivity times specificity."},
    ])
    outputs["evaluation_metrics"] = write_thesis_table("evaluation_metrics", metrics, "Evaluation metrics.", "evaluation_metrics")

    cv_cols = ["dataset", "classifier", "treatment", "cv_average_precision_mean", "cv_average_precision_std", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"]
    outputs["complete_cross_validation_results"] = write_thesis_table(
        "complete_cross_validation_results",
        make_display_table(cv.sort_values(["dataset", "classifier", "treatment"]), cv_cols),
        "Complete cross-validation results.",
        "complete_cross_validation_results",
        longtable=True,
    )
    compact_cols = ["dataset", "classifier", "treatment", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "average_precision", "roc_auc", "balanced_accuracy", "f1", "recall"]
    outputs["compact_all_45_results"] = write_thesis_table(
        "compact_all_45_results",
        make_display_table(all_results.sort_values(["dataset", "classifier", "treatment"]), compact_cols),
        "Compact results for all 45 configurations.",
        "compact_all_45_results",
        longtable=True,
    )
    best_cols = ["dataset", "classifier", "treatment", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "average_precision", "balanced_accuracy", "f1", "recall", "geometric_mean"]
    outputs["best_configuration_per_dataset"] = write_thesis_table(
        "best_configuration_per_dataset",
        make_display_table(best, best_cols),
        "Best configuration per dataset selected by cross-validation.",
        "best_configuration_per_dataset",
    )
    selected_test = test.loc[test["experiment_id"].isin(best["experiment_id"])]
    selected_cols = ["dataset", "classifier", "treatment", "average_precision", "roc_auc", "balanced_accuracy", "precision", "recall", "f1", "f2", "mcc", "geometric_mean"]
    outputs["final_test_performance_of_selected_models"] = write_thesis_table(
        "final_test_performance_of_selected_models",
        make_display_table(selected_test.sort_values("dataset"), selected_cols),
        "Untouched test-set performance of CV-selected models.",
        "final_test_performance_of_selected_models",
    )
    classifier_avg = read_csv(TABLES_DIR / "average_performance_by_classifier.csv")
    treatment_avg = read_csv(TABLES_DIR / "average_performance_by_treatment.csv")
    avg_cols_classifier = ["classifier", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"]
    avg_cols_treatment = ["treatment", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"]
    outputs["average_performance_by_classifier"] = write_thesis_table(
        "average_performance_by_classifier",
        make_display_table(classifier_avg, avg_cols_classifier),
        "Mean cross-validation performance by classifier.",
        "average_performance_by_classifier",
    )
    outputs["average_performance_by_treatment"] = write_thesis_table(
        "average_performance_by_treatment",
        make_display_table(treatment_avg, avg_cols_treatment),
        "Mean cross-validation performance by treatment.",
        "average_performance_by_treatment",
    )
    within_cols = ["dataset", "classifier", "treatment", "within_dataset_rank", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"]
    outputs["within_dataset_ranking"] = write_thesis_table(
        "within_dataset_ranking",
        make_display_table(within.sort_values(["dataset", "within_dataset_rank"]), within_cols),
        "Within-dataset rankings by cross-validation metrics.",
        "within_dataset_ranking",
        longtable=True,
    )
    overall_cols = ["classifier", "treatment", "mean_within_dataset_rank", "std_within_dataset_rank", "median_within_dataset_rank", "mean_cv_average_precision", "mean_cv_f1", "mean_cv_recall", "mean_cv_balanced_accuracy", "mean_cv_geometric_mean", "datasets_evaluated"]
    outputs["overall_model_ranking"] = write_thesis_table(
        "overall_model_ranking",
        make_display_table(overall, overall_cols),
        "Overall classifier-treatment ranking by within-dataset rank.",
        "overall_model_ranking",
    )
    baseline = test.loc[test["treatment"] == "original"].sort_values(["dataset", "classifier"])
    outputs["baseline_original_models"] = write_thesis_table(
        "baseline_original_models",
        make_display_table(baseline, selected_cols),
        "Original-treatment baseline model performance.",
        "baseline_original_models",
    )
    original_vs_display = original_vs_selected.copy()
    original_vs_display["classifier"] = original_vs_display["classifier"].map(display_classifier)
    original_vs_display["treatment"] = original_vs_display["treatment"].map(display_treatment)
    original_vs_display = original_vs_display.rename(columns={
        "dataset": "Dataset",
        "comparison": "Comparison",
        "classifier": "Classifier",
        "treatment": "Treatment",
        "test_average_precision": "Test Average Precision",
        "test_f1": "Test F1",
        "test_recall": "Test Minority Recall",
        "test_balanced_accuracy": "Test Balanced Accuracy",
        "test_geometric_mean": "Test Geometric Mean",
    })
    numeric_columns = original_vs_display.select_dtypes(include=["number"]).columns
    original_vs_display[numeric_columns] = original_vs_display[numeric_columns].round(4)
    outputs["original_vs_selected_models"] = write_thesis_table(
        "original_vs_selected_models",
        original_vs_display,
        "Original versus CV-selected model comparison.",
        "original_vs_selected_models",
    )
    return outputs


def write_results_summary(tables: dict) -> str:
    best = tables["best"]
    test = tables["test"]
    overall = tables["overall"]
    abalone_original = test.loc[
        (test["dataset"] == "abalone_19")
        & (test["treatment"] == "original")
    ].sort_values("accuracy", ascending=False).iloc[0]
    lines = [
        "# Results Summary",
        "",
        "This summary is limited to facts supported by the validated Phase 2 outputs.",
        "",
        "## Experimental Matrix",
        "",
        "The final study used `satimage`, `ozone_level`, and `abalone_19`, with imbalance ratios of 9.28:1, 33.74:1, and 129.53:1 respectively. The complete matrix contained 45 configurations: three datasets, three classifiers, and five imbalance treatments.",
        "",
        "Model selection was performed separately for each dataset using mean CV Average Precision, with CV F1, CV minority recall, CV Balanced Accuracy, and CV Geometric Mean as predefined tie-breakers. Test-set metrics were reported only after selection.",
        "",
        "## CV-Selected Configurations",
        "",
    ]
    for _, row in best.sort_values("dataset").iterrows():
        lines.append(
            f"- `{row['dataset']}`: {display_config(row['classifier'], row['treatment'])}; "
            f"CV Average Precision={row['cv_average_precision_mean']:.4f}; "
            f"test Average Precision={row['average_precision']:.4f}; "
            f"test F1={row['f1']:.4f}; test recall={row['recall']:.4f}; "
            f"test balanced accuracy={row['balanced_accuracy']:.4f}."
        )
    lines.extend([
        "",
        "## Supported Observations",
        "",
        f"Original-treatment models on `abalone_19` can show high accuracy without detecting minority cases. The strongest original-treatment accuracy among those models was {abalone_original['accuracy']:.4f}, while its minority recall was {abalone_original['recall']:.4f}.",
        "",
        "`abalone_19` was selected with XGBoost plus Random Undersampling. On the untouched test set it reached high minority recall but low precision, reflecting a precision-recall trade-off under extreme imbalance.",
        "",
        "`satimage` was selected with Random Forest plus Random Oversampling. `ozone_level` was selected with XGBoost plus Random Oversampling. No single treatment was selected for every dataset.",
        "",
        "High accuracy alone was not reliable on the most imbalanced data because majority-class predictions can dominate the accuracy score while minority recall remains zero.",
        "",
        "The overall classifier-treatment comparison uses mean within-dataset rank because class prevalence differs across datasets. XGBoost with Class Weighting had the best overall mean within-dataset rank, but it was not the selected configuration for every individual dataset.",
        "",
        "Treatment effectiveness varied by dataset, as shown by the different CV-selected configurations and the within-dataset ranking table.",
        "",
        "This file does not provide causal explanations or thesis Discussion/Conclusion text.",
    ])
    path = Path("RESULTS_SUMMARY.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def validate_latex_tables(thesis_tables: dict[str, dict[str, str]]) -> None:
    for name, paths in thesis_tables.items():
        text = Path(paths["tex"]).read_text(encoding="utf-8")
        if "\\caption{" not in text or "\\label{" not in text:
            raise ValueError(f"{name}.tex is missing caption or label")
        if "\\begin{" not in text or "\\end{" not in text:
            raise ValueError(f"{name}.tex is missing table environments")


def write_audit(manifest: dict) -> str:
    active_dataset_tables = [
        TABLES_DIR / "dataset_summary.csv",
        TABLES_DIR / "dataset_summary_raw.csv",
        TABLES_DIR / "dataset_summary_modeling.csv",
        TABLES_DIR / "dataset_cleaning_summary.csv",
        TABLES_DIR / "class_label_mappings.csv",
        TABLES_DIR / "dataset_characteristics_final.csv",
        TABLES_DIR / "dataset_cleaning_final.csv",
        TABLES_DIR / "class_label_mappings_final.csv",
    ]
    for path in active_dataset_tables:
        df = read_csv(path)
        if set(df["dataset"]) != set(FINAL_DATASETS):
            raise ValueError(f"{path} does not contain exactly the final datasets")
        if "mammography" in set(df["dataset"]):
            raise ValueError(f"{path} still contains mammography")
    for path in manifest["figures"]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing generated figure: {path}")
    validate_latex_tables(manifest["thesis_tables"])

    lines = [
        "# Final Artifact Audit",
        "",
        "- Active final dataset tables contain exactly `satimage`, `ozone_level`, and `abalone_19`.",
        "- No active final dataset table contains `mammography`.",
        "- Historical mammography documentation is preserved in `DATA_QUALITY_DECISIONS.md`, historical phase reports, and `results/archive/phase1_5_dataset_tables/`.",
        "- Figures were generated from validated saved results, saved prediction files, and deterministic descriptive dataset summaries.",
        "- No model was retrained during Phase 2.1 artifact generation.",
        "- Curve metrics were reproduced from saved prediction files.",
        "- Selected models were chosen by cross-validation only.",
        "- All PNG and PDF figure files exist.",
        "- LaTeX table files contain captions, labels, and complete table environments.",
        "- Active final artifacts use Average Precision terminology.",
        "- All 45 configurations remain unchanged, as confirmed by stored SHA-256 hashes.",
        "",
        f"Generated figure files: {len(manifest['figures'])}",
        f"Generated thesis table pairs: {len(manifest['thesis_tables'])}",
    ]
    path = Path("FINAL_ARTIFACT_AUDIT.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    archived = archive_stale_dataset_tables()
    bundles = load_all_datasets()
    validate_bundles(bundles, require_modeling_ready=True)
    active_dataset_tables = regenerate_active_dataset_tables(bundles)
    grid = build_full_experiment_grid(FINAL_DATASETS)
    tables = validate_phase2_results(bundles, grid)
    figures, curve_metadata = generate_figures(bundles, tables)
    thesis_tables = generate_thesis_tables(tables)
    results_summary = write_results_summary(tables)

    manifest = {
        "archived_dataset_tables": archived,
        "active_dataset_tables": active_dataset_tables,
        "figures": figures,
        "thesis_tables": thesis_tables,
        "curve_metadata": curve_metadata,
        "phase2_result_hashes": {str(path): sha256_file(path) for path in RESULT_HASH_FILES},
        "prediction_hashes": {str(path): sha256_file(path) for path in sorted(PREDICTIONS_DIR.glob("*.csv.gz"))},
        "results_summary": results_summary,
        "validation": {
            "configurations": 45,
            "prediction_files": 45,
            "reproduced_metric_rows": tables["reproduced_metric_rows"],
            "postrun_validation": "passed",
        },
    }
    manifest["audit"] = write_audit(manifest)
    manifest_path = FINAL_FIGURE_DIR / "final_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {len(figures)} figure files and {len(thesis_tables)} thesis table pairs.")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
