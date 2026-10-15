"""Fast offline unit tests for K-fold CV isolation and evaluation metrics."""

from sklearn.model_selection import StratifiedKFold

from animoser.evaluate import compute_task_metrics, evaluate_predictions
from animoser.schema import PRIMARY_EMOTION_CLASSES, TASK_CLASSES


def test_stratified_kfold_isolation_and_coverage() -> None:
    """Verify 5-fold CV guarantees zero leakage and exact 1:1 OOF coverage."""
    num_samples = 304
    k_folds = 5
    # Simulate primary emotion labels matching the 11 classes
    labels = [i % len(PRIMARY_EMOTION_CLASSES) for i in range(num_samples)]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    val_indices_collected = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(range(num_samples), labels)
    ):
        train_set = set(train_idx)
        val_set = set(val_idx)

        # Zero leakage between train and val
        assert len(train_set.intersection(val_set)) == 0, (
            f"Leakage detected in fold {fold_idx}"
        )
        assert len(train_set) + len(val_set) == num_samples

        val_indices_collected.extend(val_idx)

    # Every sample must be in validation exactly once
    assert len(val_indices_collected) == num_samples
    assert sorted(val_indices_collected) == list(range(num_samples))


def test_compute_task_metrics_zero_support() -> None:
    """Verify metrics computation handles zero-support classes safely."""
    classes = ["ClassA", "ClassB", "ClassC"]
    y_true = [0, 0, 1]  # ClassC has 0 support
    y_pred = [0, 1, 1]

    metrics = compute_task_metrics(y_true, y_pred, classes)
    assert metrics["accuracy"] == 2 / 3
    assert "ClassC" in metrics["unsupported_classes"]
    assert metrics["per_class"]["ClassC"]["support"] == 0
    assert metrics["per_class"]["ClassC"]["f1"] == 0.0
    assert metrics["macro_f1"] >= 0.0


def test_evaluate_predictions_all_tasks() -> None:
    """Verify evaluate_predictions computes metrics for all 6 tasks."""
    records = [
        {
            "sample_id": f"s{i}",
            "ground_truth": dict.fromkeys(TASK_CLASSES, 0),
            "predicted": dict.fromkeys(TASK_CLASSES, 0),
        }
        for i in range(10)
    ]

    results = evaluate_predictions(records)
    assert len(results) == 6
    for task in TASK_CLASSES:
        assert task in results
        assert results[task]["accuracy"] == 1.0
        assert results[task]["total_samples"] == 10
