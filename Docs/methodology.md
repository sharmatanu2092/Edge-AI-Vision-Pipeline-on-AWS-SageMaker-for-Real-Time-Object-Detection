# Methodology

This document sets out the research questions, dataset design, and experimental methodology behind the dissertation, in the form expected of a supervised MSc research project rather than a model-training writeup.

---

## Research Questions

**RQ1.** To what extent can domain-specific adaptations (anchor re-clustering, targeted augmentation, hyperparameter optimisation) improve a transfer-learned YOLOv3 detector's precision on industrial CCTV imagery, relative to a naive transfer-learning baseline?

**RQ2.** What is the accuracy–latency trade-off achievable through post-training model compression (INT8 quantisation, hardware-targeted compilation) for real-time inference, and does this trade-off differ meaningfully between a cloud GPU inference target and an edge accelerator target?

**RQ3.** Which individual design decision contributes most to the overall improvement in RQ1 — i.e., is the gain attributable to a single dominant factor or distributed across several complementary changes?

These questions were chosen to keep the project answerable within a single dissertation cycle while still requiring genuine experimental design (not just applying a tutorial pipeline to a new dataset) — RQ3 in particular requires a controlled ablation study, not just a before/after comparison.

---

## Dataset Design

### Composition

The dataset was assembled from three sources rather than adopted wholesale from a single existing benchmark, to ensure coverage of the specific industrial-CCTV visual conditions (elevated fixed cameras, warehouse lighting, PPE variety) this project targets:

| Source | Contribution | Notes |
|---|---|---|
| SHWD (Safety Helmet Wearing Dataset) | Helmet/no-helmet worker imagery | Public dataset, widely used in PPE-detection literature |
| Curated Roboflow Universe public safety datasets | Additional PPE + forklift/vehicle imagery | Filtered for elevated-camera perspective consistent with the target deployment scenario |
| Custom-annotated frames | Restricted-zone marker class, fixed-machinery occlusion cases | Extracted from Creative-Commons-licensed factory-floor video, annotated in CVAT |

Total: **9,600 images**, unified into the 5-class schema described in the main README, annotated in Pascal VOC format and converted to GluonCV RecordIO shards (`data_pipeline/annotation_conversion.py`).

### Split Strategy

A **stratified** 70/15/15 split (6,720 / 1,440 / 1,440 images) was used rather than a naive random split, stratifying on class co-occurrence pattern per image — because `restricted_zone_marker` appears in only 4.2% of instances, a naive random split risked a test set with too few examples of this class to evaluate its AP reliably. `data_pipeline/dataset_split.py` implements the stratification and asserts a minimum per-class test-set instance count before accepting a split.

### Known Dataset Limitations

- All source video is daytime-lit; low-light/nighttime industrial monitoring is not represented and is flagged explicitly as **out of scope**, not silently generalised over.
- Camera elevation angle varies less than a fully deployed multi-site system would see in practice — a limitation directly motivating the domain-shift research direction in [`phd_research_trajectory.md`](phd_research_trajectory.md).
- The dataset over-represents forklifts relative to other industrial vehicle types (cranes, pallet jacks) present in real warehouses, since source material availability skewed toward forklifts.

---

## Experimental Design

### Baseline Definition

The baseline against which all improvements are measured is a **naive transfer-learning** fine-tune: COCO-pretrained YOLOv3/Darknet-53 from the GluonCV model zoo, fine-tuned on the industrial dataset using the model zoo's default anchors, default augmentation (standard flip/crop only), and a single fixed learning rate — i.e., the pipeline a practitioner would get by following the standard GluonCV YOLOv3 fine-tuning tutorial without further adaptation. This baseline exists specifically so RQ1's "improvement" has a well-defined, reproducible reference point rather than being measured against an arbitrary earlier checkpoint.

### Controlled Variables

Across the baseline and all subsequent experimental conditions, the following were held constant to isolate the effect of each individual change:

- Training/validation/test split (identical across all runs)
- Backbone architecture and pretrained initialisation (Darknet-53, COCO weights)
- Total training epoch budget (180 epochs)
- Evaluation protocol (IoU 0.5, confidence threshold 0.5, GluonCV `VOCMApMetric`)

### Statistical Methodology

Every reported metric is the mean over **three independent training runs with different random seeds** (affecting weight initialisation of the detection head and data loader shuffling), not a single run's result. Uncertainty is reported as a **95% confidence interval from bootstrap resampling** (1,000 resamples) over the test set's per-image detections, computed in `evaluation/evaluate_map.py`. This was a deliberate methodological choice: a single-run mAP figure cannot distinguish a genuine improvement from training variance, and dissertations are exactly the context where that distinction should be demonstrated, not assumed away.

### Ablation Study Design

To answer RQ3, four experimental conditions were run in sequence, each adding exactly one change on top of the previous condition:

```
[1] Baseline (naive transfer learning)
      ↓ + anchor re-clustering (k-means, k=9, on training set boxes)
[2] Baseline + anchors
      ↓ + industrial augmentation (mosaic, HSV jitter, motion blur, glare, cutout)
[3] Baseline + anchors + augmentation
      ↓ + Bayesian hyperparameter tuning (LR schedule, IoU/NMS thresholds, 40 trials)
[4] Full pipeline (= "final" throughout this dissertation)
```

Each condition was run with the same three-seed protocol as the baseline. This design directly answers RQ3 by isolating each intervention's marginal contribution rather than only reporting the cumulative endpoint — see the ablation table in [`evaluation_and_results.md`](evaluation_and_results.md#ablation-study).

---

## Threats to Validity

**Internal validity.** Hyperparameter tuning (condition [4]) was performed using the validation split only; the test split was never used for any tuning decision, and is touched exactly once per seed, at final evaluation. This was enforced programmatically in `training/hyperparameter_tuning.py` by withholding test-set file paths from the SageMaker HPO job's data channel configuration entirely, not merely by convention.

**External validity.** The dataset's daytime-only, forklift-dominant composition (noted above) limits how far these results generalise to night-shift monitoring or sites dominated by different vehicle types. This is treated as a scoped limitation, not extrapolated past.

**Construct validity.** mAP@0.5 and precision/recall at a single confidence threshold are standard but incomplete proxies for "useful in a live safety-alert system" — a system's practical value also depends on alert latency and false-alarm rate under continuous operation, which this dissertation addresses partially (via the latency benchmarking in `optimization/benchmark_latency.py`) but does not fully resolve. This gap motivates the uncertainty-quantification research direction in [`phd_research_trajectory.md`](phd_research_trajectory.md).
