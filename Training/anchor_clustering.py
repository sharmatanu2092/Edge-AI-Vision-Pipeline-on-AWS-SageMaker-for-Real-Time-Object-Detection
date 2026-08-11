"""
anchor_clustering.py — k-means anchor re-clustering on the industrial dataset.

Re-derives YOLOv3's 9 anchor boxes (3 per detection scale) from the actual
training set's bounding-box width/height distribution, using IoU-distance
k-means rather than Euclidean distance — Euclidean distance in (w,h) space
doesn't correlate well with the IoU a candidate anchor will actually
achieve against ground-truth boxes. See
docs/model_architecture.md#anchor-re-clustering for the full rationale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("anchor_clustering")

N_ANCHORS = 9          # 3 anchors x 3 detection scales (YOLOv3 design)
N_SCALES = 3
INPUT_SIZE = 416        # Normalisation reference — matches training input resolution
MAX_ITER = 300
CONVERGENCE_TOL = 1e-6


def iou_distance(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Computes 1 - IoU between each box and each anchor, assuming both are
    centred at the origin (only width/height matter for anchor shape
    matching — position is irrelevant to anchor clustering).

    boxes:   (N, 2) array of [w, h]
    anchors: (K, 2) array of [w, h]
    returns: (N, K) distance matrix
    """
    box_area = boxes[:, 0] * boxes[:, 1]
    anchor_area = anchors[:, 0] * anchors[:, 1]

    inter_w = np.minimum(boxes[:, 0:1], anchors[:, 0])
    inter_h = np.minimum(boxes[:, 1:2], anchors[:, 1])
    inter_area = inter_w * inter_h

    union_area = box_area[:, None] + anchor_area[None, :] - inter_area
    iou = inter_area / np.maximum(union_area, 1e-9)

    return 1.0 - iou


def kmeans_iou(boxes: np.ndarray, k: int, seed: int = 13) -> np.ndarray:
    """
    k-means clustering of bounding boxes using IoU distance. Centroids are
    updated as the median (w, h) of assigned boxes — median rather than
    mean, since box dimensions are not well-approximated by a Gaussian and
    median is more robust to the outlier boxes present in an occlusion-
    heavy class like fixed_machinery.
    """
    rng = np.random.RandomState(seed)
    n = boxes.shape[0]

    # k-means++ style initialisation: pick first centroid randomly, then
    # each subsequent centroid weighted by squared distance from nearest
    # existing centroid — avoids poor convergence from unlucky random init.
    centroids = [boxes[rng.randint(n)]]
    for _ in range(k - 1):
        dists = iou_distance(boxes, np.array(centroids)).min(axis=1)
        probs = dists ** 2
        probs /= probs.sum()
        next_idx = rng.choice(n, p=probs)
        centroids.append(boxes[next_idx])
    centroids = np.array(centroids)

    prev_assignment = np.zeros(n, dtype=int) - 1
    for iteration in range(MAX_ITER):
        distances = iou_distance(boxes, centroids)
        assignment = distances.argmin(axis=1)

        if np.array_equal(assignment, prev_assignment):
            logger.info("Converged after %d iterations", iteration)
            break

        for cluster_idx in range(k):
            members = boxes[assignment == cluster_idx]
            if len(members) > 0:
                centroids[cluster_idx] = np.median(members, axis=0)

        prev_assignment = assignment

    return centroids


def mean_iou_score(boxes: np.ndarray, anchors: np.ndarray) -> float:
    """Average best-anchor IoU across all boxes — the metric used to
    validate that re-clustered anchors actually improve on COCO defaults."""
    distances = iou_distance(boxes, anchors)
    best_iou = 1.0 - distances.min(axis=1)
    return float(best_iou.mean())


def load_box_dimensions(record_stats_path: Path) -> np.ndarray:
    """
    Loads normalised [width, height] pairs for every annotated box in the
    training set, scaled to the 416x416 input resolution. Expects a JSON
    file of the form {"boxes": [[w_norm, h_norm], ...]} produced during
    annotation_conversion.py's training-set pass.
    """
    with open(record_stats_path) as f:
        data = json.load(f)
    boxes = np.array(data["boxes"], dtype=np.float32) * INPUT_SIZE
    return boxes


def cluster_anchors(record_stats_path: Path, output_path: Path, seed: int = 13) -> dict:
    boxes = load_box_dimensions(record_stats_path)
    logger.info("Clustering anchors from %d bounding boxes", len(boxes))

    anchors = kmeans_iou(boxes, N_ANCHORS, seed=seed)

    # Sort anchors by area, ascending — required so the smallest 3 map to
    # the finest detection scale (52x52), matching YOLOv3's convention.
    areas = anchors[:, 0] * anchors[:, 1]
    anchors = anchors[np.argsort(areas)]

    achieved_iou = mean_iou_score(boxes, anchors)

    # COCO defaults, for comparison — this is the baseline the ablation
    # study measures the re-clustered anchors against.
    coco_default_anchors = np.array([
        [10, 13], [16, 30], [33, 23],
        [30, 61], [62, 45], [59, 119],
        [116, 90], [156, 198], [373, 326],
    ], dtype=np.float32)
    coco_baseline_iou = mean_iou_score(boxes, coco_default_anchors)

    result = {
        "anchors": anchors.tolist(),
        "mean_best_anchor_iou": achieved_iou,
        "coco_default_mean_best_anchor_iou": coco_baseline_iou,
        "improvement": achieved_iou - coco_baseline_iou,
        "n_boxes_clustered": len(boxes),
        "seed": seed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Re-clustered anchors: mean best-anchor IoU %.4f (COCO defaults: %.4f, +%.4f)",
                achieved_iou, coco_baseline_iou, result["improvement"])
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cluster_anchors(args.record_stats, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
