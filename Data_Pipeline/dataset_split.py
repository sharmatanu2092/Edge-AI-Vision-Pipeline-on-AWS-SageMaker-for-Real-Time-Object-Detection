"""
dataset_split.py — Stratified 70/15/15 train/val/test split.

A naive random split risks leaving too few restricted_zone_marker instances
(4.2% of all labelled instances) in the test set to evaluate its AP
reliably. This module stratifies on per-image class co-occurrence pattern
and asserts a minimum per-class test-set instance count before accepting
a split — see docs/methodology.md#split-strategy.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("dataset_split")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Minimum instances of the rarest class (restricted_zone_marker) required
# in each split before a candidate split is accepted. Chosen so per-class
# AP on the test set has a reasonable sample size to be meaningful.
MIN_MINORITY_CLASS_TEST_INSTANCES = 60

RARE_CLASS = "restricted_zone_marker"


@dataclass
class SplitResult:
    train_files: list[str]
    val_files: list[str]
    test_files: list[str]
    class_counts_per_split: dict[str, dict[str, int]]


def load_image_class_manifest(manifest_path: Path) -> dict[str, list[str]]:
    """
    Expects a JSON manifest: {"image_filename": ["class_a", "class_b", ...]}
    mapping each image to the set of classes present in it (built during
    annotation_conversion.py's parse pass).
    """
    with open(manifest_path) as f:
        return json.load(f)


def stratified_split(manifest: dict[str, list[str]], seed: int = 13) -> SplitResult:
    """
    Groups images by their class co-occurrence signature (the sorted tuple
    of classes present), then splits each group proportionally — this
    keeps rare-class-containing images distributed across all three splits
    in roughly the target ratio, rather than risking them clustering into
    just one split under a purely random assignment.
    """
    rng = random.Random(seed)

    groups: dict[tuple, list[str]] = defaultdict(list)
    for filename, classes in manifest.items():
        signature = tuple(sorted(set(classes)))
        groups[signature].append(filename)

    train, val, test = [], [], []
    for signature, files in groups.items():
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        train.extend(files[:n_train])
        val.extend(files[n_train:n_train + n_val])
        test.extend(files[n_train + n_val:])

    result = SplitResult(
        train_files=train, val_files=val, test_files=test,
        class_counts_per_split=_count_classes_per_split(manifest, train, val, test),
    )

    _validate_split(result)
    return result


def _count_classes_per_split(manifest, train, val, test) -> dict[str, dict[str, int]]:
    counts = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}
    for split_name, files in [("train", train), ("val", val), ("test", test)]:
        for f in files:
            for cls in manifest[f]:
                counts[split_name][cls] += 1
    return {k: dict(v) for k, v in counts.items()}


def _validate_split(result: SplitResult) -> None:
    """
    Asserts the rare class has enough test-set instances before accepting
    the split. Raises rather than silently proceeding with a split that
    would make the rare class's test-set AP unreliable — this check exists
    specifically because of the 4.2% instance-share imbalance documented
    in the main README's class distribution table.
    """
    test_counts = result.class_counts_per_split["test"]
    rare_count = test_counts.get(RARE_CLASS, 0)

    if rare_count < MIN_MINORITY_CLASS_TEST_INSTANCES:
        raise ValueError(
            f"Split rejected: only {rare_count} '{RARE_CLASS}' instances in test set "
            f"(minimum required: {MIN_MINORITY_CLASS_TEST_INSTANCES}). "
            f"Try a different seed or re-check manifest class coverage."
        )

    logger.info("Split accepted: %d train / %d val / %d test images",
                len(result.train_files), len(result.val_files), len(result.test_files))
    logger.info("  '%s' test-set instances: %d (threshold: %d)",
                RARE_CLASS, rare_count, MIN_MINORITY_CLASS_TEST_INSTANCES)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = load_image_class_manifest(args.manifest)
    result = stratified_split(manifest, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, files in [("train", result.train_files), ("val", result.val_files),
                              ("test", result.test_files)]:
        with open(args.output_dir / f"{split_name}.txt", "w") as f:
            f.write("\n".join(files))

    with open(args.output_dir / "split_class_distribution.json", "w") as f:
        json.dump(result.class_counts_per_split, f, indent=2)


if __name__ == "__main__":
    main()
