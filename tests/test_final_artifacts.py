import hashlib
import json
from pathlib import Path

import pandas as pd


FINAL_DATASETS = {"satimage", "ozone_level", "abalone_19"}
MANIFEST_PATH = Path("results/figures/final/final_artifact_manifest.json")
THESIS_TABLES = {
    "dataset_characteristics",
    "class_label_mappings",
    "experimental_design",
    "model_hyperparameters",
    "imbalance_treatments",
    "evaluation_metrics",
    "complete_cross_validation_results",
    "compact_all_45_results",
    "best_configuration_per_dataset",
    "final_test_performance_of_selected_models",
    "average_performance_by_classifier",
    "average_performance_by_treatment",
    "within_dataset_ranking",
    "overall_model_ranking",
    "baseline_original_models",
    "original_vs_selected_models",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_active_dataset_tables_contain_only_final_datasets():
    paths = [
        Path("results/tables/dataset_summary.csv"),
        Path("results/tables/dataset_summary_raw.csv"),
        Path("results/tables/dataset_summary_modeling.csv"),
        Path("results/tables/dataset_cleaning_summary.csv"),
        Path("results/tables/class_label_mappings.csv"),
        Path("results/tables/dataset_characteristics_final.csv"),
        Path("results/tables/dataset_cleaning_final.csv"),
        Path("results/tables/class_label_mappings_final.csv"),
    ]
    for path in paths:
        df = pd.read_csv(path)
        assert len(df) == 3
        assert set(df["dataset"]) == FINAL_DATASETS
        assert "mammography" not in set(df["dataset"])


def test_historical_mammography_dataset_tables_are_archived():
    archive = Path("results/archive/phase1_5_dataset_tables")
    archived_tables = sorted(path.name for path in archive.glob("*.csv"))
    assert archived_tables == [
        "class_label_mappings.csv",
        "dataset_cleaning_summary.csv",
        "dataset_summary.csv",
        "dataset_summary_modeling.csv",
        "dataset_summary_raw.csv",
    ]
    assert any("mammography" in set(pd.read_csv(path)["dataset"]) for path in archive.glob("*.csv"))


def test_all_required_final_figures_exist_in_png_and_pdf():
    manifest = artifact_manifest()
    figure_paths = {Path(path).resolve() for path in manifest["figures"]}
    assert len(figure_paths) == 80
    expected_stems = {
        "dataset_class_distribution",
        "dataset_minority_percentage",
        "dataset_imbalance_ratio",
        "dataset_feature_distributions",
        "dataset_correlation_overview",
        "selected_model_test_performance",
        "original_vs_selected",
        "overall_mean_rank",
        "mean_cv_performance_by_classifier",
        "mean_cv_performance_by_treatment",
    }
    for dataset in FINAL_DATASETS:
        for metric in ("average_precision", "f1", "recall", "balanced_accuracy", "geometric_mean"):
            expected_stems.add(f"cv_heatmap_{dataset}_{metric}")
        expected_stems.update({
            f"selected_{dataset}_confusion_matrix",
            f"selected_{dataset}_precision_recall_curve",
            f"selected_{dataset}_roc_curve",
            f"top3_{dataset}_precision_recall_curve",
            f"top3_{dataset}_roc_curve",
        })
    for stem in expected_stems:
        for suffix in ("png", "pdf"):
            path = Path("results/figures/final") / f"{stem}.{suffix}"
            assert path.resolve() in figure_paths
            assert path.exists()
            assert path.stat().st_size > 0


def test_selected_model_curves_use_cv_selected_prediction_files():
    manifest = artifact_manifest()
    best = pd.read_csv("results/tables/best_model_per_dataset.csv")
    expected = dict(zip(best["dataset"], best["experiment_id"]))
    for dataset, metadata in manifest["curve_metadata"]["selected_model_curves"].items():
        experiment_id = metadata["experiment_id"]
        assert experiment_id == expected[dataset]
        assert Path(metadata["prediction_file"]).name == f"{experiment_id}.csv.gz"


def test_precision_recall_baselines_equal_test_minority_prevalence():
    manifest = artifact_manifest()
    for metadata in manifest["curve_metadata"]["selected_model_curves"].values():
        predictions = pd.read_csv(metadata["prediction_file"])
        assert metadata["precision_recall_baseline"] == predictions["y_true"].mean()
    for top_three in manifest["curve_metadata"]["top_three_curves"].values():
        for metadata in top_three:
            predictions = pd.read_csv(metadata["prediction_file"])
            assert metadata["precision_recall_baseline"] == predictions["y_true"].mean()


def test_all_thesis_tables_have_csv_and_latex_files():
    manifest = artifact_manifest()
    assert set(manifest["thesis_tables"]) == THESIS_TABLES
    for name, paths in manifest["thesis_tables"].items():
        csv_path = Path(paths["csv"])
        tex_path = Path(paths["tex"])
        assert csv_path.exists()
        assert tex_path.exists()
        assert len(pd.read_csv(csv_path)) > 0
        text = tex_path.read_text(encoding="utf-8")
        assert "\\caption{" in text
        assert "\\label{" in text
        assert "\\begin{" in text
        assert "\\end{" in text


def test_active_final_terminology_uses_average_precision():
    paths = [
        Path("generate_final_artifacts.py"),
        Path("src/plots.py"),
        Path("RESULTS_SUMMARY.md"),
        Path("FINAL_ARTIFACT_AUDIT.md"),
        Path("PHASE2_REPORT.md"),
        Path("results/figures/final/final_artifact_manifest.json"),
    ]
    paths.extend(Path("results/tables/thesis").glob("*.csv"))
    paths.extend(Path("results/tables/thesis").glob("*.tex"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "pr_auc" not in text
        assert "PR-AUC" not in text
        assert "PR AUC" not in text


def test_metric_files_and_predictions_were_not_changed_by_artifact_generation():
    manifest = artifact_manifest()
    for path, expected_hash in manifest["phase2_result_hashes"].items():
        assert sha256_file(Path(path)) == expected_hash
    for path, expected_hash in manifest["prediction_hashes"].items():
        assert sha256_file(Path(path)) == expected_hash


def test_all_45_result_table_still_has_unique_configurations():
    source = pd.read_csv("results/metrics/all_experiment_results.csv")
    thesis = pd.read_csv("results/tables/thesis/compact_all_45_results.csv")
    assert len(source) == 45
    assert source["experiment_id"].is_unique
    assert len(thesis) == 45
