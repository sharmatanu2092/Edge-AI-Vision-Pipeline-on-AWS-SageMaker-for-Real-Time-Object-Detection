"""
hyperparameter_tuning.py — SageMaker Automatic Model Tuning job definition.

Bayesian search (40 trials) over the hyperparameters identified as most
consequential during pilot experimentation: learning rate schedule
shape, IoU threshold, and NMS threshold. Runs on top of the anchor-
reclustered, augmentation-enabled training configuration (ablation
condition [3] in docs/evaluation_and_results.md) to produce condition [4].

Critically: the test split's file paths are never included in any channel
this HPO job can access — see docs/methodology.md#threats-to-validity for
why this is enforced structurally, not just by convention.
"""

from __future__ import annotations

import logging

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)

logger = logging.getLogger("hyperparameter_tuning")

MAX_JOBS = 40
MAX_PARALLEL_JOBS = 4   # Balances search speed against ml.p3.8xlarge cost

OBJECTIVE_METRIC_NAME = "validation:mAP"
OBJECTIVE_TYPE = "Maximize"

# Regex used to parse the mAP value SageMaker Training reports to CloudWatch
# from train_yolo3_sagemaker.py's stdout logging line format.
METRIC_DEFINITIONS = [
    {"Name": OBJECTIVE_METRIC_NAME, "Regex": r"mAP@0\.5=([0-9\.]+)"},
]

HYPERPARAMETER_RANGES = {
    "backbone-lr": ContinuousParameter(1e-5, 5e-4, scaling_type="Logarithmic"),
    "head-lr": ContinuousParameter(1e-4, 5e-3, scaling_type="Logarithmic"),
    "weight-decay": ContinuousParameter(1e-5, 1e-3, scaling_type="Logarithmic"),
    "iou-thresh": ContinuousParameter(0.3, 0.6),
    "nms-thresh": ContinuousParameter(0.35, 0.65),
    "warmup-epochs": IntegerParameter(2, 8),
}


def build_estimator(role: str, image_uri: str, output_path: str) -> Estimator:
    """
    Custom container extending the MXNet Deep Learning Container with
    GluonCV installed — see infra/docker/Dockerfile.inference for the
    equivalent inference-side container; the training container follows
    the same base-image pattern with training-specific dependencies.
    """
    return Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.p3.8xlarge",   # 4x V100 — matches training regime table
        output_path=output_path,
        entry_point="train_yolo3_sagemaker.py",
        source_dir="training/",
        hyperparameters={
            "batch-size": 32,
            "anchors-path": "/opt/ml/input/data/anchors/reclustered_anchors.json",
        },
        max_run=48 * 3600,   # 180 epochs on 4x V100 comfortably fits within 48h
    )


def launch_tuning_job(role: str, image_uri: str, output_path: str,
                       train_s3_uri: str, val_s3_uri: str, anchors_s3_uri: str) -> HyperparameterTuner:
    estimator = build_estimator(role, image_uri, output_path)

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=OBJECTIVE_METRIC_NAME,
        objective_type=OBJECTIVE_TYPE,
        hyperparameter_ranges=HYPERPARAMETER_RANGES,
        metric_definitions=METRIC_DEFINITIONS,
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
        strategy="Bayesian",
    )

    # NOTE: only "train" and "val" channels are provided. The test split's
    # S3 URI is deliberately never passed to this job — enforcing the
    # train/val/test discipline structurally rather than by convention,
    # per docs/methodology.md#threats-to-validity.
    tuner.fit({
        "train": train_s3_uri,
        "val": val_s3_uri,
        "anchors": anchors_s3_uri,
    })

    logger.info("Launched HPO job: %d max trials, %d parallel, objective=%s",
                MAX_JOBS, MAX_PARALLEL_JOBS, OBJECTIVE_METRIC_NAME)
    return tuner


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, help="SageMaker execution IAM role ARN")
    parser.add_argument("--image-uri", required=True, help="Custom MXNet+GluonCV training container URI")
    parser.add_argument("--output-path", required=True, help="S3 URI for model artifacts")
    parser.add_argument("--train-s3-uri", required=True)
    parser.add_argument("--val-s3-uri", required=True)
    parser.add_argument("--anchors-s3-uri", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    launch_tuning_job(
        args.role, args.image_uri, args.output_path,
        args.train_s3_uri, args.val_s3_uri, args.anchors_s3_uri,
    )


if __name__ == "__main__":
    main()
