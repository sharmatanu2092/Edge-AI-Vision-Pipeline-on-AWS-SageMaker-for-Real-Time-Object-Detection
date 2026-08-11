# Optimisation & Deployment

## Compilation: AWS SageMaker Neo

The trained MXNet/GluonCV model is exported to ONNX (`mxnet.contrib.onnx` export, since GluonCV's `export_block` produces a symbol/params pair that MXNet's ONNX exporter consumes directly) and compiled via **AWS SageMaker Neo** for two distinct targets:

| Target | Purpose |
|---|---|
| `ml_g4dn` (NVIDIA T4) | Cloud real-time inference endpoint |
| `jetson_xavier` (NVIDIA Xavier NX) | Edge deployment |

Neo's compiler applies target-specific graph optimisations (operator fusion, layout optimisation for the target's tensor cores) that are not available from the generic MXNet inference path — this is the primary source of the latency improvement, independent of quantisation. `optimization/compile_neo.py` submits both compilation jobs and validates output accuracy parity (compiled model predictions compared against the uncompiled model on a 200-image validation subset, requiring bounding-box IoU agreement above 0.98 before accepting a compiled artifact).

## Quantisation: Post-Training INT8

Following compilation, `optimization/quantize_int8.py` applies post-training INT8 quantisation, calibrated on a 200-frame representative subset of the training set (chosen to span all five classes and both indoor/loading-bay lighting conditions present in the source data). Quantisation-aware training was considered and rejected for this dissertation's scope — post-training quantisation achieved an acceptable accuracy cost (see the ablation table) within a much smaller implementation and compute budget, which was the appropriate trade-off given the project timeline, though quantisation-aware training remains a natural next step (noted in [`phd_research_trajectory.md`](phd_research_trajectory.md)).

## Why SageMaker Edge Manager Is Not Used

An earlier design draft of this pipeline targeted AWS SageMaker Edge Manager for the edge deployment path, which was purpose-built for exactly this (managing Neo-compiled model packages on fleets of edge devices). **AWS discontinued SageMaker Edge Manager on 26 April 2024.** The edge deployment path in this project instead uses:

```
SageMaker Neo (jetson_xavier target)
        │
        ▼
   Compiled artifact + DLR (Deep Learning Runtime) — AWS's Neo-compatible runtime
        │
        ▼
   AWS IoT Greengrass V2 component (aws.greengrass.SageMakerEdgeManager's
   successor pattern: package the Neo output + DLR runtime as a custom
   Greengrass component, deployed and version-managed via Greengrass V2's
   own deployment mechanism)
        │
        ▼
   NVIDIA Jetson Xavier NX (Greengrass core device)
```

This is the deployment pattern AWS itself now recommends in place of Edge Manager, and `deployment/greengrass_edge_deploy.py` implements it directly rather than working around a deprecated service.

## Real-Time Cloud Endpoint

`deployment/inference_endpoint.py` deploys the compiled, quantised model behind a SageMaker real-time endpoint on `ml.g4dn.xlarge`, with a target-tracking autoscaling policy keyed on `SageMakerVariantInvocationsPerInstance`, scaling out when per-instance invocation rate would push predicted latency above a 40ms SLO. This endpoint serves live camera-feed inference requests where round-trip latency is user-facing (the safety-alert path).

## Asynchronous Inference for Bulk Footage

Not every use case is live: reviewing a week of recorded footage after an incident report doesn't need sub-50ms latency, and forcing it through the real-time endpoint would be both slower under load and needlessly expensive. `deployment/async_video_inference.py` instead uses **SageMaker Asynchronous Inference**, queuing frame batches from an uploaded video file and processing them at the endpoint's sustainable throughput rather than requiring dedicated always-on capacity sized for worst-case bulk load.

## Drift Monitoring

`deployment/model_monitor_setup.py` configures **SageMaker Model Monitor** against the real-time endpoint, comparing incoming frame statistics (colour histogram distribution, average detected-object scale) against a baseline captured from the training set. This is deliberately a coarse, statistics-based drift signal rather than a full retraining trigger — its purpose in this dissertation is to demonstrate the monitoring hook exists and to generate the kind of drift evidence that would justify (and is discussed further as) the continual-adaptation research direction in [`phd_research_trajectory.md`](phd_research_trajectory.md), not to claim a solved online-adaptation system.

## Latency Benchmarking Methodology

`optimization/benchmark_latency.py` measures per-frame latency across batch sizes {1, 2, 4, 8, 16} and both the compiled and uncompiled model, on a fixed 500-frame benchmark set, reporting p50/p95/p99 latency and effective FPS. The final reported "40% latency reduction" figure compares uncompiled FP32 batch-1 latency (the baseline a naive deployment would use) against compiled + INT8-quantised batch-8 latency (the configuration the real-time endpoint actually runs), since dynamic batching at the endpoint means batch-8 is genuinely representative of production load, not an artificially favourable comparison point chosen after the fact.
