"""
confusion_matrix.py — Per-class confusion matrix at IoU 0.5.

Produces the confusion matrix reported in
docs/evaluation_and_results.md#confusion-matrix-final-pipeline-test-set,
including the explicit "missed (background)" column — a prediction system
that only reports a square class x class matrix silently hides how many
ground-truth objects were never detected at all, which is precisely the
failure mode (violation -> compliant confusion aside) most relevant to
this system's safety purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from evaluate_map import PerImageDetectionResult, _box_iou

logger = logging.getLogger("confusion_matrix")

CLASS_SCHEMA = [
    "person_ppe_compliant",
    "person_ppe_violation",
    "forklift",
    "fixed_machinery",
    "restricted_zone_marker",
]


def build_confusion_matrix(results: list[PerImageDetectionResult],
                            iou_thresh: float = 0.5, conf_thresh: float = 0.5) -> np.ndarray:
    """
    Returns an (n_classes, n_classes + 1) matrix. Row = ground-truth class,
    column = predicted class (last column = missed / background, i.e. no
    prediction matched this ground-truth box above iou_thresh).
    """
    n_classes = len(CLASS_SCHEMA)
    matrix = np.zeros((n_classes, n_classes + 1), dtype=int)

    for r in results:
        kept = r.pred_scores >= conf_thresh
        pred_boxes, pred_classes = r.pred_boxes[kept], r.pred_classes[kept]
        matched_preds = set()

        for gi, (gt_box, gt_cls) in enumerate(zip(r.gt_boxes, r.gt_classes)):
            best_iou, best_pred_idx, best_pred_cls = 0.0, -1, None

            for pi, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
                if pi in matched_preds:
                    continue
                iou = _box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_pred_idx, best_pred_cls = iou, pi, pred_cls

            if best_iou >= iou_thresh:
                matrix[int(gt_cls), int(best_pred_cls)] += 1
                matched_preds.add(best_pred_idx)
            else:
                matrix[int(gt_cls), n_classes] += 1  # Missed / background column

    return matrix


def format_matrix_markdown(matrix: np.ndarray) -> str:
    """Renders the matrix as a markdown table matching the format used in
    docs/evaluation_and_results.md, for regenerating that table directly
    from a fresh evaluation run rather than hand-editing it."""
    headers = [f"pred: {c}" for c in CLASS_SCHEMA] + ["missed (bg)"]
    lines = ["| | " + " | ".join(headers) + " |",
             "|---|" + "|".join(["---"] * len(headers)) + "|"]

    for i, class_name in enumerate(CLASS_SCHEMA):
        row = matrix[i].tolist()
        lines.append(f"| **{class_name}** | " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(lines)


def highest_stakes_errors(matrix: np.ndarray) -> list[tuple[str, str, int]]:
    """
    Flags the confusion cells most relevant to this system's safety
    purpose: any ground-truth person_ppe_violation predicted as
    person_ppe_compliant (a missed safety violation — the highest-stakes
    error this system can make), returned sorted by count descending.
    """
    violation_idx = CLASS_SCHEMA.index("person_ppe_violation")
    compliant_idx = CLASS_SCHEMA.index("person_ppe_compliant")

    count = int(matrix[violation_idx, compliant_idx])
    return [("person_ppe_violation", "person_ppe_compliant", count)] if count > 0 else []


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raise NotImplementedError(
        "Wire up PerImageDetectionResult loading, then call build_confusion_matrix(results)"
    )


if __name__ == "__main__":
    main()
