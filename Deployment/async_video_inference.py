"""
async_video_inference.py — SageMaker Asynchronous Inference for bulk
footage review.

Not every use case is live: reviewing a week of recorded footage after an
incident report doesn't need sub-50ms latency, and forcing it through the
real-time endpoint would be both slower under load and needlessly
expensive to keep provisioned for worst-case bulk throughput. See
docs/optimization_and_deployment.md#asynchronous-inference-for-bulk-footage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
import sagemaker
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.model import Model

logger = logging.getLogger("async_video_inference")

ASYNC_INSTANCE_TYPE = "ml.g4dn.xlarge"
MAX_CONCURRENT_INVOCATIONS = 4
S3_OUTPUT_PREFIX = "s3://industrial-vision-async-results/"
S3_FAILURE_PREFIX = "s3://industrial-vision-async-failures/"


def deploy_async_endpoint(compiled_model_s3_uri: str, role: str, endpoint_name: str,
                           sns_success_topic: str | None = None,
                           sns_failure_topic: str | None = None) -> str:
    """
    Deploys the same compiled model as the real-time endpoint, but behind
    an asynchronous inference configuration — requests are queued to S3
    rather than requiring dedicated always-on capacity sized for
    worst-case bulk load.
    """
    model = Model(
        model_data=compiled_model_s3_uri,
        role=role,
        image_uri=sagemaker.image_uris.retrieve(
            framework="neo-mxnet", region=boto3.Session().region_name,
            version="1.9.1", instance_type=ASYNC_INSTANCE_TYPE,
        ),
    )

    async_config = AsyncInferenceConfig(
        output_path=S3_OUTPUT_PREFIX,
        failure_path=S3_FAILURE_PREFIX,
        max_concurrent_invocations_per_instance=MAX_CONCURRENT_INVOCATIONS,
        notification_config={
            "SuccessTopic": sns_success_topic,
            "ErrorTopic": sns_failure_topic,
        } if sns_success_topic else None,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=ASYNC_INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        async_inference_config=async_config,
    )

    logger.info("Deployed async inference endpoint '%s'", endpoint_name)
    return predictor.endpoint_name


def submit_video_for_bulk_analysis(endpoint_name: str, video_s3_uri: str,
                                    frames_per_batch: int = 32) -> list[str]:
    """
    Splits a recorded video into frame batches (reusing the extraction
    logic conceptually from data_pipeline/extract_frames.py, but at full
    frame rate rather than the 1fps dataset-building sample rate) and
    submits each batch as a separate async invocation, returning the
    output S3 locations to poll.
    """
    runtime = boto3.client("sagemaker-runtime")
    output_locations = []

    # In production: frames are extracted via OpenCV at native frame rate,
    # batched, and each batch's payload uploaded to S3 before invocation —
    # async inference requires the input to already be in S3, unlike the
    # real-time endpoint's inline payload.
    batch_s3_uris = _prepare_video_batches(video_s3_uri, frames_per_batch)

    for batch_uri in batch_s3_uris:
        response = runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=batch_uri,
            ContentType="application/x-recordio",
        )
        output_locations.append(response["OutputLocation"])

    logger.info("Submitted %d batches from %s for async processing", len(batch_s3_uris), video_s3_uri)
    return output_locations


def _prepare_video_batches(video_s3_uri: str, frames_per_batch: int) -> list[str]:
    raise NotImplementedError(
        "Downloads video from S3, extracts frames at native frame rate, "
        "batches into RecordIO shards, re-uploads to S3, returns batch URIs"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-model-s3-uri", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--endpoint-name", default="yolo3-industrial-async")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    deploy_async_endpoint(args.compiled_model_s3_uri, args.role, args.endpoint_name)


if __name__ == "__main__":
    main()
