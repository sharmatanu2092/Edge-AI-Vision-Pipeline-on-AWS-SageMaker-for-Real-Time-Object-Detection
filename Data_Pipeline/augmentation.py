"""
augmentation.py — Industrial-domain-specific augmentation pipeline.

Five augmentations beyond GluonCV's standard flip/crop transforms, each
targeting a specific failure mode observed in a pilot evaluation of the
baseline model. See docs/model_architecture.md#domain-specific-augmentation
for the rationale behind each one.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


class IndustrialAugmentationPipeline:
    """
    Applied as an additional GluonCV transform stage, composed after the
    standard resize/flip transforms in the training data loader. Each
    augmentation has an independent application probability, applied in a
    fixed order so combined effects (e.g. blur + glare) are reproducible
    across seeds given a fixed random state.
    """

    def __init__(self, mosaic_prob=0.5, hsv_prob=0.8, blur_prob=0.3,
                 glare_prob=0.25, cutout_prob=0.3, seed: int | None = None):
        self.mosaic_prob = mosaic_prob
        self.hsv_prob = hsv_prob
        self.blur_prob = blur_prob
        self.glare_prob = glare_prob
        self.cutout_prob = cutout_prob
        self._rng = random.Random(seed)

    # ── Mosaic: 4-image composite ─────────────────────────────────────────
    def mosaic(self, images: list[np.ndarray], boxes_list: list[np.ndarray],
               output_size: int = 416) -> tuple[np.ndarray, np.ndarray]:
        """
        Composites 4 training images into one, placing each in a quadrant
        around a randomised centre point. Targets poor small-object
        (restricted_zone_marker) recall by increasing small-object density
        per training batch — see docs/model_architecture.md.
        """
        assert len(images) == 4, "Mosaic requires exactly 4 source images"
        cx = self._rng.randint(output_size // 4, 3 * output_size // 4)
        cy = self._rng.randint(output_size // 4, 3 * output_size // 4)

        canvas = np.full((output_size, output_size, 3), 114, dtype=np.uint8)
        quadrants = [(0, 0, cx, cy), (cx, 0, output_size, cy),
                     (0, cy, cx, output_size), (cx, cy, output_size, output_size)]

        all_boxes = []
        for img, boxes, (x0, y0, x1, y1) in zip(images, boxes_list, quadrants):
            qw, qh = x1 - x0, y1 - y0
            resized = cv2.resize(img, (qw, qh))
            canvas[y0:y1, x0:x1] = resized

            if boxes.size > 0:
                scaled = boxes.copy()
                scaled[:, [1, 3]] = boxes[:, [1, 3]] * qw + x0  # xmin, xmax
                scaled[:, [2, 4]] = boxes[:, [2, 4]] * qh + y0  # ymin, ymax
                all_boxes.append(scaled)

        merged_boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 5))
        return canvas, merged_boxes

    # ── HSV colour jitter ─────────────────────────────────────────────────
    def hsv_jitter(self, image: np.ndarray, h_gain=0.015, s_gain=0.5, v_gain=0.4) -> np.ndarray:
        """Targets overfitting to the specific lighting of source footage."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = np.clip(hsv[..., 0] * (1 + self._rng.uniform(-h_gain, h_gain) * 180), 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + self._rng.uniform(-s_gain, s_gain)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + self._rng.uniform(-v_gain, v_gain)), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Synthetic motion blur ─────────────────────────────────────────────
    def motion_blur(self, image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
        """
        Simulates camera vibration on elevated warehouse mounts — a
        recurring cause of missed detections observed in pilot footage.
        """
        angle = self._rng.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
        kernel /= kernel.sum() if kernel.sum() != 0 else 1
        return cv2.filter2D(image, -1, kernel)

    # ── Synthetic glare/low-light ─────────────────────────────────────────
    def synthetic_glare(self, image: np.ndarray) -> np.ndarray:
        """
        Simulates warehouse skylight glare and shadowed loading-bay areas —
        the recurring false-negative pattern this augmentation targets.
        """
        h, w = image.shape[:2]
        overlay = image.copy()

        if self._rng.random() < 0.5:
            # Glare: bright elliptical highlight
            cx, cy = self._rng.randint(0, w), self._rng.randint(0, h // 2)
            axes = (self._rng.randint(w // 6, w // 3), self._rng.randint(h // 8, h // 4))
            cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, (255, 255, 255), -1)
            image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
        else:
            # Shadow: dark region simulating loading-bay shade
            x0 = self._rng.randint(0, w // 2)
            cv2.rectangle(overlay, (x0, 0), (min(x0 + w // 2, w), h), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

        return image

    # ── Occlusion cutout ──────────────────────────────────────────────────
    def occlusion_cutout(self, image: np.ndarray, n_patches: int = 2,
                          patch_size_frac: float = 0.12) -> np.ndarray:
        """
        Targets fixed_machinery frequently partially occluding workers or
        forklifts in the source footage — trains robustness to partial
        object visibility rather than only complete, unoccluded instances.
        """
        h, w = image.shape[:2]
        ph, pw = int(h * patch_size_frac), int(w * patch_size_frac)

        for _ in range(n_patches):
            y0 = self._rng.randint(0, max(1, h - ph))
            x0 = self._rng.randint(0, max(1, w - pw))
            image[y0:y0 + ph, x0:x0 + pw] = self._rng.randint(80, 130)

        return image

    # ── Composed pipeline ─────────────────────────────────────────────────
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Applies the non-mosaic augmentations in sequence, each gated by
        its independent probability. Mosaic is applied separately at the
        batch-composition stage, not per-image, since it needs 4 source
        images rather than operating on one."""
        if self._rng.random() < self.hsv_prob:
            image = self.hsv_jitter(image)
        if self._rng.random() < self.blur_prob:
            image = self.motion_blur(image)
        if self._rng.random() < self.glare_prob:
            image = self.synthetic_glare(image)
        if self._rng.random() < self.cutout_prob:
            image = self.occlusion_cutout(image)
        return image
