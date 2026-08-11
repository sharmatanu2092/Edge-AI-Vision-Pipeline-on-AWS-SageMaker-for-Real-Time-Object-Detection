"""
inference_endpoint.py — Real-time SageMaker endpoint with autoscaling.

Deploys the Neo-compiled, INT8-quantised model behind a SageMaker
real-time endpoint on ml.g4dn.xlarge, with a target-tracking autoscaling
policy keyed on invocations-per-instance, scaling out before predicted
latency would exceed a 40ms SLO. Serves the live-camera-feed inference
path — see docs/optimization_and_deployment.md#real-time-cloud-endpoint.
"""

from __future__ import annotations

import logging

import boto3
import sagemaker
from sagemaker.model import Model

logger = logging.getLogger("inference_endpoint")

ENDPOINT_INSTANCE_TYPE = "ml.g4dn.xlarge"
LATENCY_SLO_MS = 40

# Target invocations-per-instance chosen empirically from the benchmark
# results in docs/evaluation_and_results.md (batch-8 compiled+quantised
# achieves ~36 FPS per instance) with a safety margin so autoscaling
# triggers before the SLO is actually breached, not after.
TARGET_INVOCATIONS_PER_INSTANCE = 28  # ~78% of measured max throughput

MIN_CAPACITY = 2   # No single point of failure for a live safety-alert path
MAX_CAPACITY = 10


def deploy_endpoint(compiled_model_s3_uri: str, role: str, endpoint_name: str,
                     sagemaker_session=None) -> str:
    model = Model(
        model_data=compiled_model_s3_uri,
        role=role,
        sagemaker_session=sagemaker_session,
        # Neo-compiled models require the corresponding Neo inference
        # container, matched to the compilation target framework/version.
        image_uri=sagemaker.image_uris.retrieve(
            framework="neo-mxnet", region=boto3.Session().region_name,
            version="1.9.1", instance_type=ENDPOINT_INSTANCE_TYPE,
        ),
    )

    predictor = model.deploy(
        initial_instance_count=MIN_CAPACITY,
        instance_type=ENDPOINT_INSTANCE_TYPE,
        endpoint_name=endpoint_name,
    )

    _configure_autoscaling(endpoint_name)
    logger.info("Deployed real-time endpoint '%s' on %s (min=%d, max=%d capacity)",
                endpoint_name, ENDPOINT_INSTANCE_TYPE, MIN_CAPACITY, MAX_CAPACITY)
    return predictor.endpoint_name


def _configure_autoscaling(endpoint_name: str) -> None:
    """
    Configures target-tracking autoscaling on SageMakerVariantInvocations
    PerInstance — the standard SageMaker autoscaling metric for real-time
    endpoints, tuned to the empirical throughput ceiling measured in
    optimization/benchmark_latency.py rather than a default/guessed value.
    """
    client = boto3.client("application-autoscaling")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=MIN_CAPACITY,
        MaxCapacity=MAX_CAPACITY,
    )

    client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-target-tracking",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": TARGET_INVOCATIONS_PER_INSTANCE,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 300,   # Avoid flapping on transient dips in camera traffic
            "ScaleOutCooldown": 60,   # Scale out quickly — latency SLO breach is the risk
        },
    )
    logger.info("Autoscaling configured: target=%d invocations/instance/min, cooldowns 60s/300s",
                TARGET_INVOCATIONS_PER_INSTANCE)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-model-s3-uri", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--endpoint-name", default="yolo3-industrial-realtime")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    deploy_endpoint(args.compiled_model_s3_uri, args.role, args.endpoint_name)


if __name__ == "__main__":
    main()
