# Model Architecture

## YOLOv3 / Darknet-53 Overview

The detector is YOLOv3 with a Darknet-53 backbone, used via GluonCV's `yolo3_darknet53_coco` model zoo entry, COCO-pretrained and fine-tuned on the industrial dataset. YOLOv3 was chosen over single-shot alternatives available in the GluonCV model zoo (SSD, Faster R-CNN) for a specific reason relevant to this deployment: it predicts at three scales (13×13, 26×26, 52×52 feature maps for a 416×416 input), which matters directly for this dataset's object-size distribution — `restricted_zone_marker` instances are small, floor-level features, while `forklift` and `fixed_machinery` occupy a much larger portion of the frame. A single-scale detector head would have systematically under-served one end of that size distribution.

```
Input (416×416×3)
   │
   ▼
Darknet-53 backbone (53 conv layers, residual blocks)
   │
   ├──► Scale 1: 13×13 grid  → large objects   (forklift, fixed_machinery)
   ├──► Scale 2: 26×26 grid  → medium objects  (person_ppe_compliant/violation)
   └──► Scale 3: 52×52 grid  → small objects   (restricted_zone_marker)
```

---

## Anchor Re-Clustering

GluonCV's default YOLOv3 anchors are derived from COCO's object-scale distribution via the original Darknet k-means procedure. Reusing them for a visually different domain (fixed elevated industrial cameras vs. COCO's varied natural photography) means the anchor priors don't match what the detector actually needs to predict.

`training/anchor_clustering.py` re-runs k-means (k=9, matching YOLOv3's 3-anchors-per-scale × 3-scales design) directly on the training set's bounding-box width/height distribution (normalised to the 416×416 input resolution), using IoU-distance rather than Euclidean distance as the clustering metric — standard practice for anchor clustering, since Euclidean distance in (w,h) space does not correlate well with the IoU a candidate anchor will actually achieve against ground-truth boxes.

This was the single highest-impact change in the ablation study (see [`evaluation_and_results.md`](evaluation_and_results.md#ablation-study)) — a result consistent with the intuition that anchor priors matter more when the target domain's object geometry diverges further from the pretraining domain's.

---

## Training Regime

| Setting | Value |
|---|---|
| Backbone init | COCO-pretrained Darknet-53 (GluonCV model zoo) |
| Epochs | 180 |
| LR schedule | Cosine decay, 4-epoch linear warm-up |
| Learning rate (backbone) | 1e-4 (discriminative — lower rate for pretrained features) |
| Learning rate (detection head) | 1e-3 |
| Optimiser | SGD with momentum 0.9, weight decay tuned via HPO |
| Batch size | 32 (distributed across 4× V100 on ml.p3.8xlarge) |
| Input resolution | 416×416 |
| Loss | YOLOv3 composite loss (objectness + classification + box regression) |

**Why discriminative learning rates?** The Darknet-53 backbone's early layers encode general edge/texture features that transfer well from COCO; the detection head must learn industrial-domain-specific object statistics essentially from scratch. A single global learning rate forces a compromise between "don't disturb useful pretrained features" and "adapt the head fast enough to converge within the epoch budget." Splitting the rate avoids that compromise directly.

---

## Domain-Specific Augmentation

Beyond GluonCV's standard flip/crop transforms, five industrial-domain-targeted augmentations were added (`data_pipeline/augmentation.py`), each chosen in response to a specific failure mode observed in a pilot evaluation of the baseline model on held-out validation frames:

| Augmentation | Failure mode it targets |
|---|---|
| Mosaic (4-image composite) | Poor small-object (`restricted_zone_marker`) recall — mosaic increases small-object density per training batch |
| HSV colour jitter | Overfitting to the specific lighting conditions of the source footage |
| Synthetic motion blur | Camera vibration on elevated warehouse mounts, observed to cause missed detections in pilot footage |
| Synthetic low-light/glare | Warehouse skylight glare and shadowed loading-bay areas, a recurring false-negative pattern in the pilot |
| Occlusion cutout | `fixed_machinery` frequently partially occludes workers/forklifts in the source footage |

Each augmentation was validated individually on a small held-out pilot subset before being included in the final augmentation pipeline — this is documented as part of the ablation study's "augmentation" condition rather than treated as an unexamined grab-bag of standard tricks.

---

## Hard-Negative Mining for the Minority Class

`restricted_zone_marker` at 4.2% of instances risked being under-learned by a standard loss (the detector could achieve a low average loss by simply predicting it rarely). `training/train_yolo3_sagemaker.py` implements a hard-negative mining pass: after an initial 60-epoch warm-up, the training loop identifies validation-set false negatives for this class specifically and oversamples the corresponding training images for the remaining epochs. This is a targeted response to a specific, measured problem (verified via the per-class AP breakdown at the 60-epoch checkpoint), not a generic class-balancing default applied without diagnosis.

---

## Framework Longevity — A Note on GluonCV/MXNet

This project was built on Apache MXNet and GluonCV. Apache MXNet was formally retired by the Apache Software Foundation in September 2023, with its codebase moved to the Apache Attic (archival, read-only status) in February 2024. GluonCV, being built directly on MXNet, is affected by the same status.

This is disclosed here deliberately rather than glossed over, because it is itself a relevant finding for anyone assessing this project's infrastructure choices: a framework that was a reasonable, well-supported choice at project inception can become a maintenance liability within a single dissertation-to-viva timeline. For any continuation of this work — including the PhD research trajectory in [`phd_research_trajectory.md`](phd_research_trajectory.md) — the practical implication is that a re-implementation on a currently maintained framework (PyTorch, or a JAX-based detection stack) would be a prerequisite, not an optional modernisation. The core experimental findings (anchor re-clustering's outsized contribution, the augmentation choices, the quantisation trade-off curve) are framework-agnostic and would be expected to transfer.
