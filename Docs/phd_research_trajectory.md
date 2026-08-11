# PhD Research Trajectory

A dissertation's limitations, honestly reported, are usually a better source of research questions than its results. This document sets out six directions this project's design and findings point toward, each linked to an active research area with a track record of prior published work — the intent is that this could plausibly seed a PhD research proposal, not just gesture at "future work" generically.

---

## 1. Domain Shift & Continual Adaptation for Safety-Critical Detectors

**What this project found:** the dataset's daytime-only, single-camera-geometry composition (flagged in [`methodology.md`](methodology.md#known-dataset-limitations)) and the SageMaker Model Monitor drift-detection hook (see [`optimization_and_deployment.md`](optimization_and_deployment.md#drift-monitoring)) both point at the same underlying problem: a detector trained on one site's visual distribution will degrade at a second site with different camera elevation, lighting, or PPE styling, and there is currently no mechanism in this pipeline to adapt to that shift without a full retraining cycle.

**Research direction:** online/continual domain adaptation for object detectors in safety-critical deployments, specifically addressing catastrophic forgetting — a detector that adapts to a new site's visual distribution must not silently lose accuracy on classes or conditions well-represented in the original training distribution. This connects to established work on continual learning and test-time adaptation, applied here to a domain (industrial safety) where a forgetting-induced regression has a direct safety cost, not just an accuracy-metric cost.

---

## 2. Edge–Cloud Split Inference Architectures

**What this project found:** the deployment architecture treats cloud and edge as two separate, complete deployment targets (see [`optimization_and_deployment.md`](optimization_and_deployment.md)) — the full model runs either entirely on the SageMaker endpoint or entirely on the Jetson Xavier NX. Neither option is universally optimal: full edge deployment saves bandwidth but is constrained by the Xavier's compute budget; full cloud deployment has more compute headroom but incurs continuous video-egress cost and network-dependent latency.

**Research direction:** split (collaborative) inference architectures, where early backbone layers execute on the edge device and later layers execute in the cloud, with the partition point chosen adaptively based on current network conditions and device thermal/power state. This is an active research area ("split computing") with clear applicability to exactly this project's deployment constraints.

---

## 3. Uncertainty Quantification & Calibration for Automated Compliance Systems

**What this project found:** the confusion matrix in [`evaluation_and_results.md`](evaluation_and_results.md#confusion-matrix-final-pipeline-test-set) shows 58 `person_ppe_violation` instances misclassified as `person_ppe_compliant` — the single highest-stakes error mode in the whole system, since it represents a missed safety violation. YOLOv3's confidence scores are not calibrated probabilities; a "0.9 confidence" detection and a genuinely-90%-likely-correct detection are not the same thing without explicit calibration.

**Research direction:** conformal prediction or Bayesian approaches to object detection confidence, specifically calibrated for the asymmetric cost structure of a safety-compliance system (a missed violation is more costly than a false alarm). This would let the system route low-confidence borderline detections to human review rather than acting on an uncalibrated score, directly addressing the highest-stakes error mode this dissertation's own evaluation surfaced.

---

## 4. Energy-Aware Model Compression ("Green AI") for Continuous Industrial Monitoring

**What this project found:** the latency/accuracy trade-off curve in the optimisation ablation (INT8 quantisation costing 0.9 mAP points for a substantial latency gain) was evaluated purely on the latency and accuracy axes. A real multi-camera, 24/7 deployment has a third axis this project did not measure: energy consumption, which compounds across dozens of continuously running edge devices.

**Research direction:** systematically characterising the accuracy–latency–energy Pareto frontier for industrial object detection at the edge, extending this project's quantisation and compilation ablation methodology (already reproducible and documented here) with power measurement instrumentation on the Jetson Xavier NX. This connects to the broader Green AI research agenda, applied to a concrete deployment context with a plausible cost justification for the added measurement effort.

---

## 5. Privacy-Preserving Federated Training Across Industrial Sites

**What this project found:** the ethics and data governance approach in [`ethics_and_data_governance.md`](ethics_and_data_governance.md) deliberately avoided using proprietary site-specific footage, which kept this dissertation feasible but also limits how representative the trained model is of any single real deployment site. A genuine multi-site industrial safety system faces exactly this tension at scale: each site has valuable, representative training data, but centralising CCTV-derived imagery across multiple organisations raises the same data protection concerns this project sidestepped by using only public data.

**Research direction:** federated learning approaches allowing a shared detector to improve from multiple sites' data without any site's imagery leaving its own infrastructure, with attention to the specific privacy-utility trade-offs relevant to imagery (rather than the tabular/text data federated learning research more commonly targets).

---

## 6. Explainability for Automated Safety-Compliance Decisions

**What this project found:** the system's ultimate purpose — triggering a safety alert when `person_ppe_violation` is detected inside a `restricted_zone_marker`-bounded region — is a rule layered on top of black-box detector output. If this system generates an alert (or fails to), and that alert becomes part of an incident investigation or HSE compliance record, "the model said so" is not an adequate audit trail.

**Research direction:** explainability methods for object detectors (saliency-based approaches such as Grad-CAM extended to detection heads, or attention-based architectures with inherently more interpretable intermediate representations) evaluated specifically against the auditability requirements of a regulatory/compliance context, not only against qualitative "does this look reasonable" inspection — a distinction the XAI literature does not always make explicit but which matters directly for this system's real-world deployment justification.

---

## Why These Six, and Not Others

Each direction above is derived directly from a specific, documented finding or design decision elsewhere in this repository — the intent is that a reader can trace every research question back to the evidence that motivated it, rather than treating this section as a generic "further work" list detached from what the dissertation actually did. This traceability is itself intended as evidence of the kind of research maturity a PhD application should demonstrate: knowing precisely which limitation of your own work is worth pursuing next, and why.
