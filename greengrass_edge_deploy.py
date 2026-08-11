"""
greengrass_edge_deploy.py — Edge deployment via AWS IoT Greengrass V2.

AWS SageMaker Edge Manager — the service originally intended for exactly
this deployment pattern — was discontinued on 26 April 2024. This module
implements AWS's current recommended replacement: packaging the
Neo-compiled model + DLR (Deep Learning Runtime) as a custom Greengrass V2
component, deployed to an NVIDIA Jetson Xavier NX Greengrass core device.
See docs/optimization_and_deployment.md#why-sagemaker-edge-manager-is-not-used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import boto3

logger = logging.getLogger("greengrass_edge_deploy")

COMPONENT_NAME = "com.industrialvision.yolo3detector"
COMPONENT_VERSION = "1.0.0"
TARGET_DEVICE_TYPE = "jetson_xavier"


def build_component_recipe(neo_compiled_model_s3_uri: str, dlr_version: str = "1.10.0") -> dict:
    """
    Builds the Greengrass V2 component recipe (the deployment manifest
    Greengrass uses to install and run the component on a core device).
    Bundles the Neo-compiled model artifact with a DLR runtime install
    step and a lifecycle script that starts the inference process.
    """
    return {
        "RecipeFormatVersion": "2020-01-25",
        "ComponentName": COMPONENT_NAME,
        "ComponentVersion": COMPONENT_VERSION,
        "ComponentDescription": (
            "YOLOv3 industrial object detector — Neo-compiled for Jetson Xavier NX, "
            "run via DLR. Replaces the SageMaker Edge Manager deployment pattern "
            "discontinued 26 April 2024."
        ),
        "ComponentPublisher": "LJMU-MSc-Dissertation",
        "ComponentConfiguration": {
            "DefaultConfiguration": {
                "modelS3Uri": neo_compiled_model_s3_uri,
                "inferenceIntervalMs": 100,   # ~10 inferences/sec, within the
                                               # Xavier NX's measured throughput headroom
            }
        },
        "Manifests": [
            {
                "Platform": {"os": "linux", "architecture": "aarch64"},
                "Lifecycle": {
                    "Install": f"pip3 install dlr=={dlr_version}",
                    "Run": (
                        "python3 -u {artifacts:path}/run_inference.py "
                        "--model-path {artifacts:path}/compiled_model "
                        "--interval-ms {configuration:/inferenceIntervalMs}"
                    ),
                },
                "Artifacts": [
                    {
                        "Uri": neo_compiled_model_s3_uri,
                        "Unarchive": "ZIP",
                    },
                    {
                        "Uri": "s3://industrial-vision-edge-artifacts/run_inference.py",
                    },
                ],
            }
        ],
    }


def publish_component(recipe: dict, region: str = "eu-west-2") -> str:
    """Publishes the component recipe to AWS IoT Greengrass V2's component
    registry, making it available to deploy to registered core devices."""
    client = boto3.client("greengrassv2", region_name=region)

    response = client.create_component_version(
        inlineRecipe=json.dumps(recipe).encode("utf-8"),
    )
    arn = response["componentVersionArn"]
    logger.info("Published Greengrass component: %s", arn)
    return arn


def deploy_to_core_devices(component_arn: str, target_arn: str, region: str = "eu-west-2") -> str:
    """
    Creates a Greengrass V2 deployment targeting the specified thing group
    (the registered fleet of Jetson Xavier NX edge devices).
    """
    client = boto3.client("greengrassv2", region_name=region)

    response = client.create_deployment(
        targetArn=target_arn,
        deploymentName=f"{COMPONENT_NAME}-deployment",
        components={
            COMPONENT_NAME: {"componentVersion": COMPONENT_VERSION},
        },
    )
    deployment_id = response["deploymentId"]
    logger.info("Created Greengrass deployment %s targeting %s", deployment_id, target_arn)
    return deployment_id


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neo-compiled-model-s3-uri", required=True)
    parser.add_argument("--target-thing-group-arn", required=True,
                         help="ARN of the Greengrass thing group representing the Jetson Xavier NX fleet")
    parser.add_argument("--region", default="eu-west-2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    recipe = build_component_recipe(args.neo_compiled_model_s3_uri)
    component_arn = publish_component(recipe, region=args.region)
    deploy_to_core_devices(component_arn, args.target_thing_group_arn, region=args.region)


if __name__ == "__main__":
    main()
