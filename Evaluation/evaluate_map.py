"""
evaluate_map.py — mAP/precision/recall evaluation with bootstrap CIs.

Computes GluonCV's VOCMApMetric at IoU 0.5 across three training seeds,
reporting the mean and a 95% bootstrap confidence interval (1,000
resamples) rather than a single run's point estimate. See
docs/methodology.md#statistical-methodology for why this matters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

logger = logging.getLogger("evaluate_map")

CLASS_SCHEMA = [
    "person_ppe_compliant",
    "person_ppe_violation",
    "forklift",
    "fixed_machinery",
    "restricted_zone_marker",
]

N_BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95


@dataclass
class PerImageDetectionResult:
    """One test-set image's ground truth and predictions, retained per-image
    so bootstrap resampling can operate over images (the correct resampling
    unit — resampling over individual detections would break the natural
    grouping of multiple objects per image)."""
    image_id: str
    gt_boxes: np.ndarray      # (N, 4)
    gt_classes: np.ndarray    # (N,)
    pred_boxes: np.ndarray    # (M, 4)
    pred_classes: np.ndarray  # (M,)
    pred_scores: np.ndarray   # (M,)


def compute_map_for_subset(results: list[PerImageDetectionResult], iou_thresh: float = 0.5) -> dict:
    """Computes mAP, per-class AP, precision, and recall for a given subset
    of per-image results (used both for the full test set and for each
    bootstrap resample)."""
    metric = VOC07MApMetric(iou_thresh=iou_thresh, class_names=CLASS_SCHEMA)

    for r in results:
        metric.update(
            [r.pred_boxes], [r.pred_classes], [r.pred_scores],
            [r.gt_boxes], [r.gt_classes],
        )

    names, aps = metric.get(return_each=True)
    per_class_ap = dict(zip(CLASS_SCHEMA, aps[:-1]))  # last entry is mAP
    mean_ap = aps[-1]

    precision, recall = _compute_precision_recall(results, iou_thresh)

    return {
        "mAP": mean_ap,
        "per_class_ap": per_class_ap,
        "precision": precision,
        "recall": recall,
    }


def _compute_precision_recall(results: list[PerImageDetectionResult],
                               iou_thresh: float, conf_thresh: float = 0.5) -> tuple[float, float]:
    """Aggregate precision/recall across all classes at a fixed confidence
    threshold — the operating point this dissertation reports throughout,
    distinct from mAP's threshold-integrated definition."""
    tp, fp, fn = 0, 0, 0

    for r in results:
        kept = r.pred_scores >= conf_thresh
        pred_boxes, pred_classes = r.pred_boxes[kept], r.pred_classes[kept]

        matched_gt = set()
        for pb, pc in zip(pred_boxes, pred_classes):
            best_iou, best_idx = 0.0, -1
            for gi, (gb, gc) in enumerate(zip(r.gt_boxes, r.gt_classes)):
                if gi in matched_gt or gc != pc:
                    continue
                iou = _box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, gi

            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += len(r.gt_boxes) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def bootstrap_confidence_interval(results: list[PerImageDetectionResult],
                                   metric_key: str, seed: int = 13) -> tuple[float, float, float]:
    """
    Resamples the test-set images (with replacement) N_BOOTSTRAP_RESAMPLES
    times, recomputing the requested metric each time, and returns
    (point_estimate, ci_lower, ci_upper) at the 95% confidence level.
    """
    rng = np.random.RandomState(seed)
    n = len(results)

    point_estimate = compute_map_for_subset(results)[metric_key]

    bootstrap_values = []
    for _ in range(N_BOOTSTRAP_RESAMPLES):
        resample_idx = rng.randint(0, n, size=n)
        resample = [results[i] for i in resample_idx]
        bootstrap_values.append(compute_map_for_subset(resample)[metric_key])

    alpha = 1 - CONFIDENCE_LEVEL
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper


def evaluate_multi_seed(seed_results: dict[int, list[PerImageDetectionResult]]) -> dict:
    """
    Aggregates across the three training seeds (docs/methodology.md's
    three-seed protocol), reporting the mean metric across seeds alongside
    a bootstrap CI computed from the seed with the median mAP (a
    representative single seed's CI, not conflated with cross-seed variance).
    """
    seed_maps = {seed: compute_map_for_subset(results)["mAP"] for seed, results in seed_results.items()}
    mean_map = float(np.mean(list(seed_maps.values())))
    std_map = float(np.std(list(seed_maps.values())))

    median_seed = sorted(seed_maps.items(), key=lambda x: x[1])[len(seed_maps) // 2][0]
    point, ci_lo, ci_hi = bootstrap_confidence_interval(seed_results[median_seed], "mAP")

    logger.info("mAP across seeds %s: mean=%.4f std=%.4f (representative-seed 95%% CI: [%.4f, %.4f])",
                list(seed_maps.keys()), mean_map, std_map, ci_lo, ci_hi)

    return {
        "per_seed_map": seed_maps,
        "mean_map": mean_map,
        "std_map": std_map,
        "bootstrap_ci_95": [ci_lo, ci_hi],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True,
                         help="Directory of per-seed prediction JSON files (seed13.json, seed47.json, seed89.json)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Loading logic omitted for brevity — deserialises PerImageDetectionResult
    # objects from the prediction JSON files written during test-set inference.
    raise NotImplementedError(
        "Wire up PerImageDetectionResult loading from your inference output format, "
        "then call evaluate_multi_seed(seed_results)"
    )


if __name__ == "__main__":
    main()
