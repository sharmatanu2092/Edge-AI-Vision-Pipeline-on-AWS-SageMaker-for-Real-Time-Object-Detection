<div align="center">

# Edge AI Vision Pipeline on AWS SageMaker for Real-Time Object Detection

**An MSc dissertation project: an end-to-end YOLOv3 object detection pipeline — from data governance through training, evaluation, and edge-optimised deployment — applied to industrial safety and asset monitoring.**

[![Institution](https://img.shields.io/badge/Institution-Liverpool%20John%20Moores%20University-a78bfa?style=flat-square&labelColor=16141f)](.)
[![Programme](https://img.shields.io/badge/Programme-MSc%20Embedded%20Systems%20%26%20IC%20Design-a78bfa?style=flat-square&labelColor=16141f)](.)
[![mAP](https://img.shields.io/badge/mAP%400.5-81.0%25-34d399?style=flat-square&labelColor=16141f)](.)
[![Precision](https://img.shields.io/badge/Precision-%2B15%25%20rel.-34d399?style=flat-square&labelColor=16141f)](.)
[![Latency](https://img.shields.io/badge/Inference%20Latency-%E2%88%9240%25-2dd4bf?style=flat-square&labelColor=16141f)](.)
[![License](https://img.shields.io/badge/License-MIT-948da8?style=flat-square&labelColor=16141f)](LICENSE)

[Dashboard](dashboard/index.html) · [Methodology](docs/methodology.md) · [Model Architecture](docs/model_architecture.md) · [Optimisation & Deployment](docs/optimization_and_deployment.md) · [Results](docs/evaluation_and_results.md) · [Ethics](docs/ethics_and_data_governance.md) · [PhD Research Trajectory](docs/phd_research_trajectory.md)

</div>

---

## Abstract

Manual CCTV monitoring for industrial safety compliance — personal protective equipment (PPE) checks, restricted-zone intrusion, vehicle-pedestrian conflict — does not scale, and its detection latency is bounded by human attention span, not sensor capability. This dissertation designs, trains, evaluates, and deploys a real-time object detection pipeline that automates this monitoring task end-to-end: a YOLOv3 (Darknet-53) detector fine-tuned via GluonCV/Apache MXNet on a curated industrial imagery dataset, trained and tuned on AWS SageMaker, and optimised for deployment at both cloud and edge endpoints.

Against a naive transfer-learning baseline, the final pipeline improves detection **precision by 15.0% (relative)** and reduces **inference latency by 40%**, through a combination of anchor re-clustering, domain-specific augmentation, Bayesian hyperparameter optimisation, and post-training INT8 quantisation via SageMaker Neo. The project is scoped and documented as a piece of applied ML infrastructure research, not a model-only exercise: data governance and ethics, experimental rigour (multi-seed evaluation, bootstrapped confidence intervals, ablation analysis), and production deployment concerns (autoscaling, drift monitoring, edge compilation) are treated as first-class parts of the contribution.

---

## Institution & Context

| | |
|---|---|
| **Institution** | Liverpool John Moores University |
| **Department** | Electronic & Electrical Engineering Division, School of Engineering, Faculty of Engineering and Technology |
| **Programme** | MSc Embedded Systems and IC Design |
| **Project type** | Dissertation — independent research project |
| **Author** | Tanu Sharma |

**A note on programme fit.** Embedded Systems and IC Design is a hardware-first programme — microcontroller architecture, digital design, SoC design, real-time device management. This dissertation was deliberately scoped so the ML model itself is only half the contribution: the other half is treating deployment as a hardware-software co-design problem — compiling for a specific target (SageMaker Neo), quantising for a specific accelerator, and shipping to a real edge device (Jetson Xavier NX) via Greengrass — which is the part of this project that draws most directly on the programme's core subject matter, not an add-on to it.

---

## The Problem

Industrial sites (warehouses, factory floors, construction yards) rely on CCTV for safety compliance, but the monitoring bottleneck is human: a security operator watching 12+ camera feeds cannot reliably catch a missing hard hat or a pedestrian in a forklift's blind spot in real time. The cost of a miss is not abstract — it shows up in HSE incident reports and insurance claims. This project asks whether a real-time detector can close that gap, and — just as importantly for a dissertation — whether it can do so with methodological rigour sufficient to trust the resulting numbers, not just report them.

---

## What Was Built

1. **A governed data pipeline** — video-to-frame extraction, face/plate anonymisation *before* any frame is persisted, and CVAT-based annotation into a unified 5-class schema, assembled from public safety-detection datasets and hand-annotated Creative Commons footage.
2. **A trained detector** — YOLOv3/Darknet-53, transfer-learned via GluonCV on Apache MXNet, with anchor boxes re-clustered on the target domain rather than reused from COCO.
3. **A tuned detector** — SageMaker Automatic Model Tuning (Bayesian search, 40 trials) over learning rate schedule, IoU threshold, and NMS threshold; domain-specific augmentation (motion blur, synthetic glare, occlusion cutout) targeting failure modes observed in a pilot evaluation.
4. **A rigorously evaluated detector** — mAP@0.5, per-class AP, precision/recall, confusion matrices, all reported as a mean ± 95% bootstrap CI across three training seeds, plus a controlled ablation study isolating the contribution of each design decision.
5. **An optimised, deployed detector** — SageMaker Neo compilation, INT8 post-training quantisation, a real-time autoscaled SageMaker endpoint for live feeds, an asynchronous inference path for bulk footage review, and an edge deployment path to an NVIDIA Jetson Xavier NX via AWS IoT Greengrass V2.
6. **A monitored detector** — SageMaker Model Monitor watching for input distribution drift, directly motivating the research extensions below.

---

## Headline Results

| Metric | Baseline (naive transfer learning) | Final pipeline | Change |
|---|---|---|---|
| mAP@0.5 | 71.2% ± 0.6 | **81.0% ± 0.5** | +9.8 pts (+13.8% relative) |
| Precision @0.5 IoU | 76.4% ± 0.7 | **87.9% ± 0.6** | +11.5 pts (**+15.0% relative**) |
| Recall @0.5 IoU | 69.8% ± 0.8 | **79.3% ± 0.7** | +9.5 pts |
| Inference latency (p50, per frame) | 46.3 ms | **27.8 ms** | **−40.0%** |
| Effective throughput | 21.6 FPS | **36.0 FPS** | +66.7% |

All figures are the mean over three independent training seeds on a held-out test set of 1,440 images, with uncertainty reported as a 95% bootstrap confidence interval (1,000 resamples). Full methodology and the incremental ablation breakdown behind these numbers: [`docs/evaluation_and_results.md`](docs/evaluation_and_results.md).

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              DATA GOVERNANCE LAYER                          │
│  Source video (CC-licensed / public safety datasets)                        │
│    → OpenCV frame extraction → face/plate anonymisation (BEFORE persist)     │
│    → CVAT annotation (5-class schema) → S3 (annotated, anonymised only)      │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE  (data_pipeline/)                     │
│  Stratified 70/15/15 split → k-means anchor re-clustering (k=9)              │
│  Industrial augmentation: mosaic, HSV jitter, motion blur, synthetic glare,   │
│  occlusion cutout  →  GluonCV RecordIO shards                                │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              TRAINING  (training/)  —  AWS SageMaker Training Job            │
│  YOLOv3 / Darknet-53 (GluonCV model zoo, COCO-pretrained)                     │
│  Instance: ml.p3.8xlarge (4× V100) · script-mode custom MXNet container       │
│  SageMaker Automatic Model Tuning — Bayesian search, 40 trials                │
│  SageMaker Experiments (run tracking) + SageMaker Debugger (loss divergence)  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION  (evaluation/)                                 │
│  mAP@0.5 / per-class AP / PR curves / confusion matrix                       │
│  3-seed mean ± bootstrap 95% CI · controlled ablation study                  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                   OPTIMISATION  (optimization/)                             │
│  SageMaker Neo compilation (ml_g4dn + jetson_xavier targets)                  │
│  MXNet → ONNX export · INT8 post-training quantisation (200-frame calibration)│
│  Latency/throughput benchmarking across batch sizes and instance types        │
└───────────────────┬───────────────────────────────────┬──────────────────────┘
                    ▼                                   ▼
┌────────────────────────────────┐      ┌────────────────────────────────────┐
│   CLOUD DEPLOYMENT                │      │   EDGE DEPLOYMENT                    │
│   Real-time SageMaker endpoint     │      │   AWS IoT Greengrass V2 component     │
│   (autoscaled, ml.g4dn.xlarge)     │      │   → Neo-compiled + DLR runtime         │
│   Async Inference (bulk footage)   │      │   → NVIDIA Jetson Xavier NX             │
│   SageMaker Model Monitor (drift)  │      │                                        │
└────────────────────────────────┘      └────────────────────────────────────┘
```

---

## Application Domain: Industrial Safety & Asset Monitoring

Five detection classes were chosen to cover the most commonly cited HSE-reportable incident categories in warehouse/factory environments:

| Class | Description | Share of labelled instances |
|---|---|---|
| `person_ppe_compliant` | Worker wearing hard hat + hi-vis vest | 38.4% |
| `person_ppe_violation` | Worker missing hard hat and/or hi-vis vest | 21.7% |
| `forklift` | Forklift / industrial vehicle | 19.6% |
| `fixed_machinery` | Static machinery (occlusion-heavy class) | 16.1% |
| `restricted_zone_marker` | Floor marking denoting a restricted/vehicle zone | 4.2% |

The class imbalance (`restricted_zone_marker` at 4.2% of instances) is deliberate — it reflects a real deployment condition and motivated the hard-negative mining strategy discussed in [`docs/model_architecture.md`](docs/model_architecture.md). Detections of `person_ppe_violation` inside a `forklift`-adjacent or `restricted_zone_marker`-bounded region trigger the rule-based safety alert this system is ultimately built to support.

---

## Why These Design Decisions

**Why re-cluster anchor boxes instead of reusing COCO's defaults?**
COCO's anchor priors are shaped by natural-image object statistics — people, vehicles, and everyday objects at typical photographic scales. Industrial CCTV geometry is different: fixed elevated cameras produce a narrower range of object aspect ratios (workers and forklifts photographed from a consistent downward angle) and a different scale distribution (long focal-length coverage of a warehouse floor). Running k-means (k=9) directly on the training set's bounding-box width/height distribution before training began was the single highest-leverage change in the ablation study — see [`docs/evaluation_and_results.md`](docs/evaluation_and_results.md#ablation-study).

**Why GluonCV/MXNet rather than a PyTorch-based YOLO implementation?**
At the time this project began, GluonCV's model zoo provided a well-documented, SageMaker-integrated YOLOv3 reference implementation with first-class MXNet training container support, making it the pragmatic choice for a dissertation timeline. Apache MXNet was formally retired by the Apache Software Foundation in September 2023 (Attic transfer completed February 2024) — a fact this project treats as a data point, not something to obscure. See the **Framework Longevity** note in [`docs/model_architecture.md`](docs/model_architecture.md) for the reflection this prompted about research infrastructure sustainability, which now directly feeds the PhD research trajectory below.

**Why both a real-time cloud endpoint and an edge deployment path?**
A live safety-alert system cannot tolerate the round-trip latency and bandwidth cost of streaming every camera's full video feed to the cloud continuously. The architecture supports both: SageMaker Neo compiles the same trained model for either an EC2 inference instance (cloud, high throughput, centralised monitoring) or a Jetson Xavier NX at the edge (local, low-latency, no continuous video egress). AWS SageMaker Edge Manager — the AWS-native tool originally intended for exactly this — was discontinued on 26 April 2024; the edge deployment path here instead uses SageMaker Neo output plus the DLR runtime deployed as an AWS IoT Greengrass V2 component, which is AWS's current recommended replacement path.

**Why INT8 quantisation rather than a smaller architecture (e.g. YOLOv3-tiny)?**
YOLOv3-tiny was evaluated early as an alternative and rejected: its reduced backbone capacity cost far more accuracy (mAP@0.5 dropped to 58.3% in a pilot run) than the latency it saved. Post-training INT8 quantisation of the full Darknet-53 backbone recovered nearly all of the latency budget YOLOv3-tiny would have offered, at a fraction of the accuracy cost (see the ablation table) — the right lever to pull was numerical precision, not architectural capacity.

---

## Repository Structure

```
edge-ai-vision-pipeline/
│
├── data_pipeline/
│   ├── extract_frames.py             # OpenCV video → frame extraction
│   ├── annotation_conversion.py      # CVAT/Pascal VOC → GluonCV RecordIO
│   ├── augmentation.py               # Industrial-specific augmentation pipeline
│   └── dataset_split.py              # Stratified 70/15/15 split
│
├── training/
│   ├── anchor_clustering.py          # k-means anchor re-clustering (k=9)
│   ├── train_yolo3_sagemaker.py      # SageMaker script-mode entry point (GluonCV)
│   ├── hyperparameter_tuning.py      # SageMaker Automatic Model Tuning job def
│   └── sagemaker_estimator_launch.py # Training job submission
│
├── evaluation/
│   ├── evaluate_map.py               # mAP / PR curve computation (GluonCV metrics)
│   ├── confusion_matrix.py           # Per-class confusion matrix @ IoU 0.5
│   └── ablation_study.py             # Controlled ablation experiment runner
│
├── optimization/
│   ├── compile_neo.py                # SageMaker Neo compilation (cloud + edge targets)
│   ├── quantize_int8.py              # Post-training INT8 quantisation + calibration
│   └── benchmark_latency.py          # Latency/throughput benchmarking harness
│
├── deployment/
│   ├── inference_endpoint.py         # Real-time SageMaker endpoint + autoscaling
│   ├── async_video_inference.py      # SageMaker Async Inference for bulk footage
│   ├── greengrass_edge_deploy.py     # AWS IoT Greengrass V2 edge component deploy
│   └── model_monitor_setup.py        # SageMaker Model Monitor — drift detection
│
├── infra/
│   ├── cloudformation_stack.yaml     # S3, IAM, endpoint config as IaC
│   └── docker/Dockerfile.inference   # Custom inference container
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── dashboard/
│   └── index.html                    # Research analytics dashboard
│
├── docs/
│   ├── methodology.md                # Research questions, dataset, experimental design
│   ├── model_architecture.md         # YOLOv3 architecture, anchors, training regime
│   ├── optimization_and_deployment.md
│   ├── evaluation_and_results.md     # Full results, ablation, statistical methodology
│   ├── ethics_and_data_governance.md # GDPR, anonymisation, ethics review process
│   └── phd_research_trajectory.md    # Open research questions this project surfaces
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Stack

<div align="center">

| Layer | Technology |
|---|---|
| Detection model | YOLOv3 (Darknet-53 backbone) |
| Deep learning framework | Apache MXNet · GluonCV model zoo |
| Training infrastructure | AWS SageMaker Training Jobs · SageMaker Automatic Model Tuning |
| Experiment tracking | SageMaker Experiments · SageMaker Debugger · TensorBoard |
| Data tooling | OpenCV · CVAT · Pandas · Roboflow (dataset curation) |
| Model compilation | AWS SageMaker Neo · ONNX (interchange) |
| Quantisation | Post-training INT8, calibration-based |
| Cloud inference | SageMaker real-time endpoint (autoscaled) · SageMaker Async Inference |
| Edge inference | AWS IoT Greengrass V2 · DLR runtime · NVIDIA Jetson Xavier NX |
| Monitoring | SageMaker Model Monitor · Amazon CloudWatch |
| Infra as code | AWS CloudFormation · Docker |
| Storage / trigger | Amazon S3 · AWS Lambda |
| Language | Python 3.9 |

</div>

---

## PhD Relevance

This dissertation was deliberately scoped so its limitations point toward genuine open research questions rather than simply toward "more data" or "more compute." Full discussion, including how each links to an active research area with citable prior work: [`docs/phd_research_trajectory.md`](docs/phd_research_trajectory.md). In brief:

1. **Domain shift & continual adaptation** for safety-critical detectors deployed across sites with varying camera geometry and lighting
2. **Edge–cloud split inference architectures** for latency/bandwidth/energy trade-offs in continuous industrial monitoring
3. **Uncertainty quantification and calibration** for object detectors used in automated compliance/alerting systems, where a miscalibrated confidence score has a real safety cost
4. **Energy-aware model compression** ("Green AI") for 24/7 multi-camera inference at industrial scale
5. **Privacy-preserving federated training** across multiple industrial sites that cannot centralise CCTV-derived data
6. **Explainability for automated safety-compliance decisions**, relevant to regulatory auditability

---

## Ethics & Data Governance

All footage used was either public-domain/Creative-Commons licensed or drawn from established public safety-detection research datasets; no proprietary site footage was used. Face and licence-plate regions are anonymised **before** any frame is persisted to storage, not as a post-hoc step. The project's data handling was designed against LJMU's postgraduate research ethics review process. Full details: [`docs/ethics_and_data_governance.md`](docs/ethics_and_data_governance.md).

---

## License

MIT — see [LICENSE](LICENSE)

## Author

**Tanu Sharma** · [github.com/sharmatanu2092](https://github.com/sharmatanu2092)

*MSc Embedded Systems and IC Design dissertation, Liverpool John Moores University.*
