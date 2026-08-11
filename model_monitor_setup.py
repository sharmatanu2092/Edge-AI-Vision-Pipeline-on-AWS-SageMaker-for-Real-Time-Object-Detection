"""
model_monitor_setup.py — SageMaker Model Monitor drift detection.

Configures a coarse, statistics-based drift signal against the real-time
endpoint — comparing incoming frame colour-histogram distribution and
average detected-object scale against a baseline captured from the
training set. Deliberately not a full retraining trigger: its purpose is
to demonstrate the monitoring hook and generate drift evidence motivating
the continual-adaptation research direction in
docs/phd_research_trajectory.md, not to claim a solved online-adaptation
system. See docs/optimization_and_deployment.md#drift-monitoring.
"""

from __future__ import annotations

import logging

from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

logger = logging.getLogger("model_monitor_setup")

DATA_CAPTURE_SAMPLING_PERCENTAGE = 20   # Capture 1 in 5 requests — full
                                        # capture would be unnecessary
                                        # storage/cost for a coarse drift
                                        # signal, and 20% gives ample
                                        # sample size given endpoint volume
MONITORING_SCHEDULE_CRON = "cron(0 * ? * * *)"   # Hourly


def configure_data_capture(s3_capture_path: str) -> DataCaptureConfig:
    return DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=DATA_CAPTURE_SAMPLING_PERCENTAGE,
        destination_s3_uri=s3_capture_path,
    )


def establish_baseline(monitor: DefaultModelMonitor, training_frame_stats_s3_uri: str,
                        baseline_output_s3_path: str) -> None:
    """
    Establishes the drift-detection baseline from the training set's frame
    statistics (colour histogram distribution, average detected-object
    scale per class) — computed once, offline, from the same training
    data used to fit the model.
    """
    monitor.suggest_baseline(
        baseline_dataset=training_frame_stats_s3_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_output_s3_path,
    )
    logger.info("Baseline established from %s", training_frame_stats_s3_uri)


def schedule_monitoring(monitor: DefaultModelMonitor, endpoint_name: str,
                         baseline_statistics_uri: str, baseline_constraints_uri: str,
                         report_s3_path: str) -> str:
    """
    Schedules hourly drift-checking runs against the live endpoint's
    captured inference data, comparing against the established baseline.
    An hourly cadence was chosen as a pragmatic default for a dissertation-
    scope deployment — a genuine production system would likely tune this
    based on observed drift rate, which is itself part of what the
    continual-adaptation PhD research direction would investigate.
    """
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{endpoint_name}-drift-schedule",
        endpoint_input=endpoint_name,
        statistics=baseline_statistics_uri,
        constraints=baseline_constraints_uri,
        schedule_cron_expression=MONITORING_SCHEDULE_CRON,
        output_s3_uri=report_s3_path,
    )
    logger.info("Monitoring schedule created for endpoint '%s' (cron: %s)",
                endpoint_name, MONITORING_SCHEDULE_CRON)
    return f"{endpoint_name}-drift-schedule"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--training-frame-stats-s3-uri", required=True)
    parser.add_argument("--baseline-output-s3-path", required=True)
    parser.add_argument("--report-s3-path", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    monitor = DefaultModelMonitor(role=args.role, instance_count=1, instance_type="ml.m5.xlarge")
    establish_baseline(monitor, args.training_frame_stats_s3_uri, args.baseline_output_s3_path)
    schedule_monitoring(
        monitor, args.endpoint_name,
        baseline_statistics_uri=f"{args.baseline_output_s3_path}/statistics.json",
        baseline_constraints_uri=f"{args.baseline_output_s3_path}/constraints.json",
        report_s3_path=args.report_s3_path,
    )


if __name__ == "__main__":
    main()
