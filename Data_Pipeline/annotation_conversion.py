"""
annotation_conversion.py — CVAT/Pascal VOC annotations to GluonCV RecordIO.

Converts the unified 5-class annotation set (exported from CVAT in Pascal
VOC XML format) into GluonCV's RecordIO shard format for efficient
SageMaker training data loading.
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import mxnet as mx
import numpy as np
from mxnet.recordio import IRHeader, MXIndexedRecordIO, pack_img

logger = logging.getLogger("annotation_conversion")

# Unified 5-class schema — order matters, this defines the class index
# used everywhere downstream (training labels, evaluation, confusion matrix).
CLASS_SCHEMA = [
    "person_ppe_compliant",
    "person_ppe_violation",
    "forklift",
    "fixed_machinery",
    "restricted_zone_marker",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_SCHEMA)}


def parse_voc_annotation(xml_path: Path) -> tuple[str, list[list[float]]]:
    """
    Parses a single Pascal VOC XML annotation file (as exported by CVAT)
    into (image_filename, [[class_idx, xmin, ymin, xmax, ymax], ...]),
    with box coordinates normalised to [0, 1] by image width/height —
    the format GluonCV's LST/RecordIO detection loader expects.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASS_TO_IDX:
            logger.warning("Unknown class '%s' in %s — skipping object", class_name, xml_path.name)
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text) / img_w
        ymin = float(bbox.find("ymin").text) / img_h
        xmax = float(bbox.find("xmax").text) / img_w
        ymax = float(bbox.find("ymax").text) / img_h

        boxes.append([CLASS_TO_IDX[class_name], xmin, ymin, xmax, ymax])

    return filename, boxes


def build_record_io(annotation_dir: Path, image_dir: Path, output_prefix: Path) -> dict:
    """
    Builds a .rec/.idx RecordIO shard pair from all VOC XML annotations
    in annotation_dir, reading corresponding images from image_dir.
    Returns per-class instance counts for downstream dataset reporting
    (feeds the class-distribution table in the main README/methodology doc).
    """
    record = MXIndexedRecordIO(str(output_prefix) + ".idx", str(output_prefix) + ".rec", "w")

    class_counts = {name: 0 for name in CLASS_SCHEMA}
    n_images = 0

    xml_files = sorted(annotation_dir.glob("*.xml"))
    for idx, xml_path in enumerate(xml_files):
        filename, boxes = parse_voc_annotation(xml_path)
        if not boxes:
            continue  # Skip images with no valid annotated objects

        img_path = image_dir / filename
        if not img_path.exists():
            logger.warning("Image %s referenced by %s not found — skipping", filename, xml_path.name)
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # GluonCV detection label format: flattened [class, xmin, ymin, xmax, ymax, ...]
        # header.label encodes box count metadata per GluonCV's im2rec convention.
        label = np.array(boxes, dtype=np.float32).flatten()
        header = IRHeader(flag=len(boxes), label=label, id=idx, id2=0)

        packed = pack_img(header, img_bytes, quality=95, img_fmt=".jpg")
        record.write_idx(idx, packed)

        for box in boxes:
            class_counts[CLASS_SCHEMA[int(box[0])]] += 1
        n_images += 1

    record.close()

    total_instances = sum(class_counts.values())
    logger.info("Wrote %d images, %d total instances to %s", n_images, total_instances, output_prefix)
    for name, count in class_counts.items():
        pct = 100 * count / total_instances if total_instances else 0
        logger.info("  %-26s %6d instances (%.1f%%)", name, count, pct)

    return {"n_images": n_images, "class_counts": class_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True,
                         help="Output path prefix (produces .rec and .idx)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_record_io(args.annotation_dir, args.image_dir, args.output_prefix)


if __name__ == "__main__":
    main()
