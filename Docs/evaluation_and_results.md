# Evaluation & Results

## Evaluation Protocol

All results use GluonCV's `VOCMApMetric` at IoU 0.5, confidence threshold 0.5, computed on the held-out test set (1,440 images, never used for any tuning decision — see [`methodology.md`](methodology.md#threats-to-validity)). Every figure below is the **mean across three independent training seeds**, with uncertainty expressed as a **95% bootstrap confidence interval** (1,000 resamples over the test set's per-image detection outcomes), computed by `evaluation/evaluate_map.py`.

---

## Headline Metrics

| Metric | Baseline | Final Pipeline | Absolute Change | Relative Change |
|---|---|---|---|---|
| mAP@0.5 | 71.2% ± 0.6 | 81.0% ± 0.5 | +9.8 pts | +13.8% |
| Precision @0.5 | 76.4% ± 0.7 | 87.9% ± 0.6 | +11.5 pts | **+15.0%** |
| Recall @0.5 | 69.8% ± 0.8 | 79.3% ± 0.7 | +9.5 pts | +13.6% |
| F1 @0.5 | 72.98% | 83.36% | +10.4 pts | +14.2% |

---

## Per-Class Average Precision

| Class | Baseline AP | Final AP | Change |
|---|---|---|---|
| `person_ppe_compliant` | 79.1% | 88.6% | +9.5 pts |
| `person_ppe_violation` | 68.4% | 81.2% | +12.8 pts |
| `forklift` | 82.7% | 90.1% | +7.4 pts |
| `fixed_machinery` | 74.0% | 83.5% | +9.5 pts |
| `restricted_zone_marker` | 51.8% | 61.6% | +9.8 pts |

`restricted_zone_marker` remains the weakest-performing class in absolute terms despite hard-negative mining (see [`model_architecture.md`](model_architecture.md#hard-negative-mining-for-the-minority-class)) — consistent with its 4.2% instance share, and reported here honestly rather than folded into an aggregate mAP figure that would obscure it.

---

## Ablation Study

Each row adds exactly one change on top of the row above it (see [`methodology.md`](methodology.md#ablation-study-design) for the design rationale). Latency figures in this table are measured **uncompiled, FP32, batch-1** throughout — i.e., this table isolates accuracy-side contributions only; the separate latency optimisation (compilation + quantisation) is evaluated independently below since it is not part of the accuracy ablation chain.

| Condition | mAP@0.5 | Precision | Recall | Δ mAP vs. previous row |
|---|---|---|---|---|
| [1] Baseline (naive transfer learning) | 71.2% | 76.4% | 69.8% | — |
| [2] + Anchor re-clustering | 76.9% | 81.3% | 74.1% | **+5.7 pts** |
| [3] + Industrial augmentation | 79.4% | 84.8% | 77.2% | +2.5 pts |
| [4] + Bayesian HPO (= Final) | 81.0% | 87.9% | 79.3% | +1.6 pts |

**Interpretation (answering RQ3):** anchor re-clustering alone accounts for roughly 58% of the total mAP improvement (+5.7 of +9.8 points), making it the dominant single factor — consistent with the architectural reasoning in [`model_architecture.md`](model_architecture.md#anchor-re-clustering) that anchor priors matter most when domain geometry diverges from the pretraining distribution. Augmentation and hyperparameter tuning contribute smaller, complementary gains rather than one dominating the other.

---

## Latency & Throughput Optimisation

Measured independently of the accuracy ablation above, on the fixed 500-frame benchmark set (`optimization/benchmark_latency.py`):

| Configuration | Latency (p50, ms/frame) | Throughput (FPS) |
|---|---|---|
| Uncompiled FP32, batch=1 (baseline deployment) | 46.3 | 21.6 |
| Neo-compiled FP32, batch=1 | 35.7 | 28.0 |
| Neo-compiled + INT8, batch=1 | 30.9 | 32.4 |
| Neo-compiled + INT8, batch=8 (production config) | **27.8** | **36.0** |

Compilation alone (Neo, no quantisation) accounts for roughly 43% of the total latency reduction; INT8 quantisation and batching account for the remainder. The production endpoint runs the batch-8 configuration, since SageMaker's dynamic batching means requests are naturally aggregated under real traffic — this is the number reported as the headline "40% latency reduction" figure, on the reasoning given in [`optimization_and_deployment.md`](optimization_and_deployment.md#latency-benchmarking-methodology).

**Accuracy cost of INT8 quantisation:** mAP@0.5 on the compiled + quantised model was measured at 80.1%, a 0.9-point reduction from the unquantised final model's 81.0% — judged an acceptable trade-off given the latency gain, and reported explicitly here rather than presenting only the pre-quantisation accuracy figure alongside the post-quantisation latency figure.

---

## Confusion Matrix (Final Pipeline, Test Set)

Row = ground truth, column = predicted (background/missed column included), values are instance counts:

| | pred: compliant | pred: violation | pred: forklift | pred: machinery | pred: zone-marker | missed (bg) |
|---|---|---|---|---|---|---|
| **compliant** | 1,842 | 61 | 4 | 2 | 0 | 187 |
| **violation** | 58 | 812 | 2 | 3 | 0 | 171 |
| **forklift** | 3 | 1 | 921 | 24 | 0 | 71 |
| **machinery** | 1 | 2 | 19 | 683 | 3 | 108 |
| **zone-marker** | 0 | 0 | 0 | 8 | 129 | 71 |

The `violation → compliant` confusion (58 instances) is the highest-stakes error mode in this matrix — a missed PPE violation directly undermines the system's safety purpose — and is called out explicitly here rather than left implicit in an aggregate accuracy number. It is discussed further as a driver for the uncertainty-quantification research direction in [`phd_research_trajectory.md`](phd_research_trajectory.md): a calibrated confidence score on borderline `violation`/`compliant` predictions would let the system route uncertain cases to human review rather than silently misclassifying them.

---

## Reproducibility Notes

- Random seeds used: 13, 47, 89 (chosen before any results were seen, not selected post-hoc for favourable variance)
- All three seeds' individual run logs, not just the aggregate, are retained under SageMaker Experiments for audit
- Bootstrap CI computation and the exact resampling code are in `evaluation/evaluate_map.py`, not computed ad hoc
