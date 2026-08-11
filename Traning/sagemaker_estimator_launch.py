"""
sagemaker_estimator_launch.py — Training job submission with experiment
tracking and debugging hooks.

Submits the three-seed training protocol (docs/methodology.md#statistical-
methodology) for a given ablation condition, tracked under SageMaker
Experiments so all runs — not just the aggregate mean reported in the
results doc — are retained for audit.
"""

from __future__ import annotations

import logging

import sagemaker
from sagemaker.debugger import DebuggerHookConfig, Rule, rule_configs
from sagemaker.estimator import Estimator
from sagemaker.experiments.run import Run

logger = logging.getLogger("estimator_launch")

# The three seeds used throughout this dissertation's reported results —
# fixed here, not selected after seeing outcomes. See
# docs/evaluation_and_results.md#reproducibility-notes.
SEEDS = [13, 47, 89]

# SageMaker Debugger rules: catch training instability (loss divergence,
# vanishing gradients) automatically rather than relying on a human
# noticing a bad loss curve after the fact.
DEBUGGER_RULES = [
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
]


def launch_seeded_runs(experiment_name: str, ablation_condition: str,
                        role: str, image_uri: str, output_path: str,
                        train_s3_uri: str, val_s3_uri: str, test_s3_uri: str,
                        anchors_s3_uri: str, hyperparameters: dict) -> list[str]:
    """
    Launches one training job per seed, all tagged under the same
    SageMaker Experiments experiment/trial-component group so the
    3-seed mean and per-seed variance reported in evaluation_and_results.md
    are traceable back to the actual jobs that produced them.
    """
    job_names = []

    for seed in SEEDS:
        with Run(experiment_name=experiment_name,
                  run_name=f"{ablation_condition}-seed{seed}") as run:

            run.log_parameters({
                "ablation_condition": ablation_condition,
                "seed": seed,
                **hyperparameters,
            })

            estimator = Estimator(
                image_uri=image_uri,
                role=role,
                instance_count=1,
                instance_type="ml.p3.8xlarge",
                output_path=output_path,
                entry_point="train_yolo3_sagemaker.py",
                source_dir="training/",
                hyperparameters={**hyperparameters, "seed": seed},
                debugger_hook_config=DebuggerHookConfig(
                    s3_output_path=f"{output_path}/debug/{ablation_condition}-seed{seed}",
                ),
                rules=DEBUGGER_RULES,
                max_run=48 * 3600,
            )

            job_name = f"yolo3-industrial-{ablation_condition}-seed{seed}"
            estimator.fit(
                {"train": train_s3_uri, "val": val_s3_uri, "anchors": anchors_s3_uri},
                job_name=job_name,
                wait=False,   # Launch all three seeds' jobs concurrently
            )
            job_names.append(job_name)

            logger.info("Launched %s (experiment=%s)", job_name, experiment_name)

    return job_names


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--ablation-condition", required=True,
                         choices=["baseline", "anchors", "anchors-augmentation", "final"])
    parser.add_argument("--role", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--train-s3-uri", required=True)
    parser.add_argument("--val-s3-uri", required=True)
    parser.add_argument("--test-s3-uri", required=True,
                         help="Retained for post-hoc evaluation only — never passed to the estimator")
    parser.add_argument("--anchors-s3-uri", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    launch_seeded_runs(
        args.experiment_name, args.ablation_condition, args.role, args.image_uri,
        args.output_path, args.train_s3_uri, args.val_s3_uri, args.test_s3_uri,
        args.anchors_s3_uri, hyperparameters={},
    )


if __name__ == "__main__":
    main()
