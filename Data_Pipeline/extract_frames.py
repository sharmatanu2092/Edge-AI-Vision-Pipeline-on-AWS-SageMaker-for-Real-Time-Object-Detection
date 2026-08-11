"""
extract_frames.py — Video-to-frame extraction with inline anonymisation.

Critical design property: anonymisation (face + licence-plate blurring)
happens BEFORE any frame is written to persistent storage, including the
working S3 bucket used during development. See
docs/ethics_and_data_governance.md for why this ordering matters — an
anonymisation step applied after raw frames are already stored means an
unredacted copy exists somewhere during processing. This module never
creates that copy.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("extract_frames")

# Sampling rate: 1 frame every N frames of source video. Industrial CCTV is
# typically 15-25fps; sampling at ~1fps gives enough temporal diversity for
# a static-object detection dataset without redundant near-duplicate frames.
FRAME_SAMPLE_INTERVAL = 20

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
BLUR_KERNEL = (51, 51)  # Strong blur — anonymisation is tuned toward
                         # over-inclusion, not precision (see ethics doc)


@dataclass
class ExtractionStats:
    source_video: str
    frames_read: int = 0
    frames_saved: int = 0
    faces_blurred: int = 0
    plate_regions_blurred: int = 0


class FrameAnonymiser:
    """
    Detects and blurs face and licence-plate regions. Deliberately simple
    (Haar cascade + heuristic plate regions) — this is a data-governance
    pre-processing pass, not a research contribution, and a heavier model
    here would be effort spent on the wrong part of the pipeline.
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    def anonymise(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )

        for (x, y, w, h) in faces:
            # Over-inclusive margin around the detected box — a missed face
            # is a far worse outcome than an unnecessarily blurred region.
            margin = int(0.25 * w)
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
            frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], BLUR_KERNEL, 0)

        plate_count = self._blur_plate_heuristic_regions(frame)
        return frame, len(faces), plate_count

    def _blur_plate_heuristic_regions(self, frame: np.ndarray) -> int:
        """
        Lightweight heuristic: licence-plate-shaped regions (high-contrast,
        wide-aspect-ratio rectangular regions in the lower half of forklift/
        vehicle bounding areas) are blurred conservatively. Not a learned
        detector — a coarse, safety-margin pass consistent with the
        over-inclusive anonymisation policy described in the ethics doc.
        """
        h, w = frame.shape[:2]
        lower_half = frame[h // 2:, :]
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blurred = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / max(ch, 1)
            if 2.0 < aspect < 5.5 and 40 < cw < 200:
                y_full = y + h // 2
                frame[y_full:y_full + ch, x:x + cw] = cv2.GaussianBlur(
                    frame[y_full:y_full + ch, x:x + cw], BLUR_KERNEL, 0
                )
                blurred += 1
        return blurred


def extract_frames(video_path: Path, output_dir: Path, anonymiser: FrameAnonymiser) -> ExtractionStats:
    stats = ExtractionStats(source_video=str(video_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stats.frames_read += 1

        if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
            # Anonymise BEFORE this frame ever touches disk.
            anonymised, n_faces, n_plates = anonymiser.anonymise(frame)
            stats.faces_blurred += n_faces
            stats.plate_regions_blurred += n_plates

            out_path = output_dir / f"{video_path.stem}_frame{saved_idx:05d}.jpg"
            cv2.imwrite(str(out_path), anonymised, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats.frames_saved += 1
            saved_idx += 1

        frame_idx += 1

    cap.release()
    logger.info(
        "Extracted %d/%d frames from %s (%d faces, %d plate regions blurred)",
        stats.frames_saved, stats.frames_read, video_path.name,
        stats.faces_blurred, stats.plate_regions_blurred,
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True,
                         help="Directory of source .mp4/.avi files")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to write anonymised frames to")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    anonymiser = FrameAnonymiser()

    all_stats = []
    for video_path in sorted(args.video_dir.glob("*.mp4")) + sorted(args.video_dir.glob("*.avi")):
        all_stats.append(extract_frames(video_path, args.output_dir, anonymiser))

    total_saved = sum(s.frames_saved for s in all_stats)
    total_faces = sum(s.faces_blurred for s in all_stats)
    logger.info("Done. %d videos processed, %d frames saved, %d faces anonymised.",
                len(all_stats), total_saved, total_faces)


if __name__ == "__main__":
    main()
