# Ethics & Data Governance

Working with CCTV-style imagery of people, even for a safety-monitoring use case, carries real data protection and research ethics obligations. This document sets out how those were handled, as a first-class part of the project design rather than an afterthought bolted on before submission.

---

## Data Sourcing Principles

No proprietary or site-specific CCTV footage was used at any point in this project. All source material was either:

- Drawn from established public research datasets used in the PPE-detection literature (SHWD), or
- Curated from publicly available industrial safety datasets on Roboflow Universe, filtered for appropriate licensing, or
- Extracted from Creative Commons–licensed factory-floor video with a licence permitting derivative research use

No footage was sourced from a live industrial site, a partner organisation, or any camera the author had privileged access to. This was a deliberate scoping decision, not a limitation discovered late — a dissertation timeline does not comfortably accommodate the data protection impact assessment and site agreements that genuine partner-site footage would require, and building the project around openly licensed material avoided that dependency entirely while still allowing the research questions to be answered.

---

## Anonymisation — Applied Before Persistence, Not After

Every frame extracted from source video passes through an anonymisation stage (part of `data_pipeline/extract_frames.py`) **before** it is written to any persistent storage, including the working S3 bucket used during development. Faces and vehicle licence plates are detected using lightweight pre-trained detectors (a Haar cascade face detector and a simple plate-region heuristic, both applied only for the anonymisation pass and unrelated to the YOLOv3 model under study) and blurred via Gaussian blur before the frame is saved.

This ordering matters: an anonymisation step applied *after* raw frames have already been stored, even temporarily, means an unredacted copy of personally identifiable imagery exists somewhere during processing. Doing it inline, before the first write, means no such copy is ever created.

## Data Minimisation

Once a frame has been annotated (bounding boxes + class labels extracted), the annotation record itself contains no imagery — only coordinates and class IDs. Raw (anonymised) frames are retained only for the duration needed to complete annotation and are not kept as a permanent archive alongside the final training dataset beyond what is needed for reproducibility of the reported results.

---

## Ethics Review

This project's data handling plan — sourcing, anonymisation approach, and retention policy — was designed to be submitted through Liverpool John Moores University's standard postgraduate research ethics review process prior to any data collection or processing work beginning, consistent with LJMU's requirements for any project involving imagery of identifiable individuals, even where that imagery originates from already-public sources. Ethical review of research involving human subjects (including imagery) is treated here as a precondition for the data pipeline to run, not a compliance step retrofitted around already-collected data.

---

## Limitations of This Approach

- Public/Creative-Commons source footage does not perfectly represent the visual conditions of a genuine partner-site deployment (camera placement, lighting, PPE styles specific to a given organisation's policy) — this is the same external-validity limitation flagged in [`methodology.md`](methodology.md#threats-to-validity), and the ethical scoping decision above is a direct contributor to it.
- Automated face/plate anonymisation is not perfect; a heuristic detector will miss some instances. The anonymisation pass is deliberately over-inclusive (tuned toward more false-positive blurring rather than fewer, at some cost to image quality in a minority of frames) rather than tuned for precision, since a missed face is a far worse outcome than an unnecessarily blurred piece of background.

---

## Why This Section Exists in a Technical Portfolio

A production computer-vision system that processes imagery of people carries data governance obligations regardless of whether it began as an academic project or a commercial one. Treating this as integral to the system design — not a separate compliance document written after the model was built — is itself part of what this dissertation set out to demonstrate: end-to-end ownership of an ML system means owning its data handling, not only its accuracy metrics.
