"""Evaluation metrics computation and reporting for multi-task audio classification."""

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from animoser.schema import TASK_CLASSES

LOW_SUPPORT_THRESHOLD = 5


def compute_task_metrics(
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
) -> dict[str, Any]:
    """Compute task accuracy, macro/weighted F1, and per-class metrics safely."""
    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "unsupported_classes": list(classes),
            "total_samples": 0,
        }

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    labels_all = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        zero_division=0,
    )

    per_class: dict[str, dict[str, Any]] = {}
    unsupported: list[str] = []

    for idx, name in enumerate(classes):
        sup = int(support[idx])
        if sup == 0:
            unsupported.append(name)
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": sup,
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "unsupported_classes": unsupported,
        "total_samples": len(y_true),
    }


def evaluate_predictions(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute evaluation metrics for all 6 tasks from prediction records."""
    results: dict[str, Any] = {}

    for task, classes in TASK_CLASSES.items():
        y_true: list[int] = []
        y_pred: list[int] = []

        for r in records:
            true_idx = r.get("ground_truth", {}).get(task)
            pred_idx = r.get("predicted", {}).get(task)

            # Skip masked/missing labels (-100 or None)
            if true_idx is not None and true_idx >= 0 and pred_idx is not None:
                y_true.append(true_idx)
                y_pred.append(pred_idx)

        results[task] = compute_task_metrics(y_true, y_pred, classes)

    return results


def format_evaluation_report(results: dict[str, Any]) -> str:
    """Format markdown evaluation summary table."""
    lines: list[str] = [
        "# ViP-VL Multi-Task Audio Fine-Tuning — OOF Evaluation Report",
        "",
        "| Task | Accuracy | Macro F1 | Weighted F1 | Support | Unsupported Classes |",
        "|:---|:---:|:---:|:---:|:---:|:---|",
    ]

    for task, m in results.items():
        unsupported = (
            ", ".join(m["unsupported_classes"]) if m["unsupported_classes"] else "None"
        )
        row = (
            f"| **{task}** | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['weighted_f1']:.4f} | {m['total_samples']} | {unsupported} |"
        )
        lines.append(row)

    lines.append("")
    lines.append("## Per-Class Breakdown")
    for task, m in results.items():
        lines.append(f"\n### {task}")
        lines.append("| Class | Precision | Recall | F1-Score | Support | Status |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---|")
        for cls_name, stats in m["per_class"].items():
            sup = stats["support"]
            if sup == 0:
                status = "⚠️ Zero Support"
            elif sup < LOW_SUPPORT_THRESHOLD:
                status = "⚡ Low Support (<5)"
            else:
                status = "✅ Normal"

            row = (
                f"| {cls_name} | {stats['precision']:.4f} | "
                f"{stats['recall']:.4f} | {stats['f1']:.4f} | "
                f"{sup} | {status} |"
            )
            lines.append(row)

    return "\n".join(lines)


def save_evaluation_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save metrics to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
