"""
train_yolo3_sagemaker.py — SageMaker script-mode training entry point.

Fine-tunes GluonCV's COCO-pretrained yolo3_darknet53_coco on the industrial
dataset, with re-clustered anchors (training/anchor_clustering.py),
discriminative learning rates (backbone vs. detection head), and a
hard-negative mining pass for the minority restricted_zone_marker class.

Runs as a SageMaker Training Job (script mode) on ml.p3.8xlarge (4x V100),
using a custom container extending the standard MXNet Deep Learning
Container with GluonCV installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import gluoncv as gcv
import mxnet as mx
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms import presets
from gluoncv.loss import YOLOV3Loss
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet import autograd, gluon

logger = logging.getLogger("train_yolo3")

CLASS_SCHEMA = [
    "person_ppe_compliant",
    "person_ppe_violation",
    "forklift",
    "fixed_machinery",
    "restricted_zone_marker",
]
RARE_CLASS_IDX = CLASS_SCHEMA.index("restricted_zone_marker")

INPUT_SIZE = 416
TOTAL_EPOCHS = 180
WARMUP_EPOCHS = 4
HARD_NEGATIVE_MINING_START_EPOCH = 60   # After warm-up convergence, per docs/model_architecture.md

BACKBONE_LR = 1e-4
HEAD_LR = 1e-3


def build_model(re_clustered_anchors: list[list[float]], ctx: list[mx.Context]) -> gluon.Block:
    """
    Loads the COCO-pretrained yolo3_darknet53_coco model and replaces both
    the classification head (5 classes, not COCO's 80) and the anchor
    boxes (re-clustered on the industrial dataset, not COCO defaults).
    """
    net = gcv.model_zoo.get_model(
        "yolo3_darknet53_coco", pretrained=True, ctx=ctx
    )
    net.reset_class(classes=CLASS_SCHEMA, reuse_weights=None)

    # Replace default COCO anchors with the re-clustered set produced by
    # anchor_clustering.py — this is the highest-impact single change
    # identified in the ablation study (docs/evaluation_and_results.md).
    net.anchors = mx.nd.array(re_clustered_anchors, ctx=ctx[0])

    net.hybridize()
    return net


def build_optimizer(net: gluon.Block) -> gluon.Trainer:
    """
    Discriminative learning rates: a lower rate for the pretrained
    Darknet-53 backbone (general features that transfer well from COCO)
    and a higher rate for the detection head (industrial-domain-specific,
    learned largely from scratch). See docs/model_architecture.md#training-regime.
    """
    backbone_params = [p for name, p in net.collect_params().items() if "darknet" in name.lower()]
    head_params = [p for name, p in net.collect_params().items() if "darknet" not in name.lower()]

    # gluon.Trainer doesn't natively support per-group LR without a custom
    # param dict split — implemented here via two trainers stepped together.
    backbone_trainer = gluon.Trainer(
        gluon.ParameterDict({p.name: p for p in backbone_params}),
        "sgd", {"learning_rate": BACKBONE_LR, "momentum": 0.9},
    )
    head_trainer = gluon.Trainer(
        gluon.ParameterDict({p.name: p for p in head_params}),
        "sgd", {"learning_rate": HEAD_LR, "momentum": 0.9},
    )
    return backbone_trainer, head_trainer


def cosine_lr(epoch: int, base_lr: float) -> float:
    """Cosine decay with linear warm-up, per the training regime table."""
    import math
    if epoch < WARMUP_EPOCHS:
        return base_lr * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, TOTAL_EPOCHS - WARMUP_EPOCHS)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def identify_hard_negatives(net, val_loader, ctx) -> set[str]:
    """
    Identifies validation-set false negatives specifically for
    restricted_zone_marker (the 4.2%-instance-share minority class) at the
    epoch-60 checkpoint. Returns image identifiers to oversample for the
    remainder of training — see docs/model_architecture.md
    #hard-negative-mining-for-the-minority-class.
    """
    hard_negative_ids = set()
    for batch in val_loader:
        data, label, img_ids = batch
        preds = net(data.as_in_context(ctx[0]))
        # Check whether restricted_zone_marker ground truth exists but was
        # not predicted above confidence 0.5 — a false negative for this class.
        gt_classes = label[:, :, 4].asnumpy()
        pred_classes = preds[0].asnumpy()  # class predictions
        pred_scores = preds[1].asnumpy()   # confidence scores

        for i, img_id in enumerate(img_ids):
            has_rare_gt = (gt_classes[i] == RARE_CLASS_IDX).any()
            detected_rare = ((pred_classes[i] == RARE_CLASS_IDX) & (pred_scores[i] > 0.5)).any()
            if has_rare_gt and not detected_rare:
                hard_negative_ids.add(img_id)

    logger.info("Identified %d hard-negative images for restricted_zone_marker", len(hard_negative_ids))
    return hard_negative_ids


def train(args):
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]

    with open(args.anchors_path) as f:
        anchor_data = json.load(f)
    re_clustered_anchors = anchor_data["anchors"]
    logger.info("Loaded re-clustered anchors (mean best-anchor IoU: %.4f)",
                anchor_data["mean_best_anchor_iou"])

    net = build_model(re_clustered_anchors, ctx)
    backbone_trainer, head_trainer = build_optimizer(net)
    loss_fn = YOLOV3Loss()

    train_loader = _build_data_loader(args.train_data, args.batch_size, is_training=True)
    val_loader = _build_data_loader(args.val_data, args.batch_size, is_training=False)
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=CLASS_SCHEMA)

    hard_negative_ids: set[str] = set()

    for epoch in range(TOTAL_EPOCHS):
        backbone_trainer.set_learning_rate(cosine_lr(epoch, BACKBONE_LR))
        head_trainer.set_learning_rate(cosine_lr(epoch, HEAD_LR))

        # Hard-negative oversampling kicks in only after the epoch-60
        # warm-up checkpoint, once there's a stable model to diagnose
        # false negatives against — see hard-negative mining docstring above.
        if epoch == HARD_NEGATIVE_MINING_START_EPOCH:
            hard_negative_ids = identify_hard_negatives(net, val_loader, ctx)
            train_loader = _build_data_loader(
                args.train_data, args.batch_size, is_training=True,
                oversample_ids=hard_negative_ids, oversample_factor=3,
            )

        epoch_loss = 0.0
        for batch in train_loader:
            data, label = batch[0].as_in_context(ctx[0]), batch[1].as_in_context(ctx[0])
            with autograd.record():
                obj_loss, center_loss, scale_loss, cls_loss = net(data, label)
                total_loss = obj_loss + center_loss + scale_loss + cls_loss
            total_loss.backward()
            backbone_trainer.step(args.batch_size)
            head_trainer.step(args.batch_size)
            epoch_loss += total_loss.mean().asscalar()

        if epoch % 10 == 0 or epoch == TOTAL_EPOCHS - 1:
            map_result = _evaluate(net, val_loader, metric, ctx)
            logger.info("Epoch %d: loss=%.4f mAP@0.5=%.4f", epoch, epoch_loss, map_result)

    net.export(os.path.join(args.model_dir, "yolo3_industrial"))
    logger.info("Training complete. Model exported to %s", args.model_dir)


def _build_data_loader(data_path, batch_size, is_training, oversample_ids=None, oversample_factor=1):
    # Simplified — production version wires in IndustrialAugmentationPipeline
    # from data_pipeline/augmentation.py for is_training=True, and applies
    # oversample_factor duplication of oversample_ids within the loader's
    # index list when hard-negative mining is active.
    raise NotImplementedError("Data loader construction — see data_pipeline/ for transform pipeline")


def _evaluate(net, val_loader, metric, ctx) -> float:
    metric.reset()
    for batch in val_loader:
        data, label = batch[0].as_in_context(ctx[0]), batch[1]
        pred_classes, pred_scores, pred_boxes = net(data)
        metric.update(pred_boxes, pred_classes, pred_scores, label[:, :, :4], label[:, :, 4:5])
    return metric.get()[1]  # mAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val-data", type=str, default=os.environ.get("SM_CHANNEL_VAL"))
    parser.add_argument("--anchors-path", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", 0)))
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())
