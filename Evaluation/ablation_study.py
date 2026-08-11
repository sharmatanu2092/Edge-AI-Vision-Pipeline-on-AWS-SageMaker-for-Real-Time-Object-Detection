"""
ablation_study.py — Controlled ablation experiment runner.

Orchestrates the four experimental conditions defined in
docs/methodology.md#ablation-study-design, each adding exactly one change
on top of the previous condition, all under the same three-seed protocol
as the baseline. Produces the ablation table in
docs/evaluation_and_results.md#ablation-study.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from evaluate_map import evaluate_multi_seed

logger = logging.getLogger("ablation_study")


@dataclass
class AblationCondition:
    name: str
    description: str
    use_reclustered_anchors: bool
    use_industrial_augmentation: bool
    use_hpo_tuned_hyperparameters: bool
    results: dict = field(default_factory=dict)


ABLATION_CONDITIONS = [
    AblationCondition(
        name="baseline",
        description="Naive transfer learning — COCO anchors, standard flip/crop only, fixed LR",
        use_reclustered_anchors=False,
        use_industrial_augmentation=False,
        use_hpo_tuned_hyperparameters=False,
    ),
    AblationCondition(
        name="anchors",
        description="Baseline + k-means anchor re-clustering",
        use_reclustered_anchors=True,
        use_industrial_augmentation=False,
        use_hpo_tuned_hyperparameters=False,
    ),
    AblationCondition(
        name="anchors-augmentation",
        description="+ industrial augmentation (mosaic, HSV, motion blur, glare, cutout)",
        use_reclustered_anchors=True,
        use_industrial_augmentation=True,
        use_hpo_tuned_hyperparameters=False,
    ),
    AblationCondition(
        name="final",
        description="+ Bayesian hyperparameter tuning (full pipeline)",
        use_reclustered_anchors=True,
        use_industrial_augmentation=True,
        use_hpo_tuned_hyperparameters=True,
    ),
]


def run_ablation_study(training_launcher, evaluation_fn) -> list[AblationCondition]:
    """
    Runs all four conditions in sequence. `training_launcher` and
    `evaluation_fn` are injected (rather than hardcoded to a specific
    SageMaker call) so this module can be unit-tested against a mock
    trainer without submitting real SageMaker jobs — see
    training/sagemaker_estimator_launch.py for the real launcher this
    would be wired to in production use.
    """
    for condition in ABLATION_CONDITIONS:
        logger.info("Running ablation condition: %s (%s)", condition.name, condition.description)

        job_names = training_launcher(
            ablation_condition=condition.name,
            use_reclustered_anchors=condition.use_reclustered_anchors,
            use_industrial_augmentation=condition.use_industrial_augmentation,
            use_hpo_tuned_hyperparameters=condition.use_hpo_tuned_hyperparameters,
        )

        seed_results = evaluation_fn(job_names)
        condition.results = evaluate_multi_seed(seed_results)

        logger.info("Condition '%s': mAP=%.4f (std=%.4f)",
                    condition.name, condition.results["mean_map"], condition.results["std_map"])

    _log_marginal_contributions()
    return ABLATION_CONDITIONS


def _log_marginal_contributions() -> None:
    """
    Computes and logs each condition's marginal mAP contribution over the
    previous condition — this is what answers RQ3 (docs/methodology.md):
    which single design decision contributes most to the overall
    improvement, isolated from the cumulative endpoint figure alone.
    """
    for i in range(1, len(ABLATION_CONDITIONS)):
        prev, curr = ABLATION_CONDITIONS[i - 1], ABLATION_CONDITIONS[i]
        if not prev.results or not curr.results:
            continue
        delta = curr.results["mean_map"] - prev.results["mean_map"]
        logger.info("Marginal contribution of '%s' over '%s': %+.4f mAP",
                    curr.name, prev.name, delta)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the ablation condition plan without launching training jobs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        for c in ABLATION_CONDITIONS:
            logger.info("[%s] %s", c.name, c.description)
        return

    raise NotImplementedError(
        "Wire up training_launcher to training/sagemaker_estimator_launch.py "
        "and evaluation_fn to evaluate_map.py's PerImageDetectionResult loading"
    )


if __name__ == "__main__":
    main()
