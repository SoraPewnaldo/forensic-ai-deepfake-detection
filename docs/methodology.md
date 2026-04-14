# 🔬 METHODOLOGY — FORENSIC-AI

*(Extracted from R&D Report — No Simplification)*

---

## 1. Overview

The proposed system is a:

> **Staged transfer learning framework for cross-dataset deepfake detection with forensic calibration**

### Core Objectives:

- Learn **manipulation-invariant features across datasets**
- Ensure **robust generalization under domain shift**
- Generate **Likelihood Ratio (LR) outputs for forensic interpretation**

---

## 2. System Architecture

The framework consists of five major components:

1. Frame Processing and Sampling  
2. Spatial Feature Extraction (Pretrained Backbone)  
3. Temporal Modeling Module  
4. Frequency Domain Feature Branch  
5. Likelihood Ratio Calibration Module  

---

## 3. Dataset Strategy (Staged Protocol)

### Stage 1 — Base Training

- **Dataset:** FaceForensics++
- **Objective:** Learn general deepfake artifacts

---

### Stage 2 — Domain Adaptation

- **Dataset:** Hybrid (FF++ + Augmentation + WildDeepfake)
- **Objective:** Adapt to diverse, real-world forensic artifacts
  - FaceForensics++ (Simulated CRF 35 compression)
  - WildDeepfake (mirrored internet fakes)
  - Resolution Degradation (128x128 bicubic)

- **Strategy:**  
  - Fine-tune **only higher layers**  
  - Keep lower layers frozen  

---

### Stage 3 — Unseen Evaluation

- **Dataset:** Celeb-DF v2  
- **Objective:** Simulate real forensic deployment  
- **Constraint:**  
  - No parameter updates allowed  

---

## 4. Video-Level Representation

The system operates at **video level**, not frame level.

### Process:

1. Uniformly sample frames from each video  
2. Extract frame-level embeddings using pretrained backbone  
3. Aggregate embeddings into a video-level representation  

---

## 5. Spatial Feature Extraction

- Backbone: **Pretrained CNN / Vision Transformer**

### Characteristics:

- Initialized using large-scale pretrained weights  
- Captures:
  - facial inconsistencies  
  - texture anomalies  

---

## 6. Temporal Modeling

Frame embeddings are aggregated using:

> **Temporal Attention Mechanism**

### Purpose:

- Assign adaptive importance to frames  
- Capture:
  - temporal inconsistencies  
  - motion irregularities  

### Insight:

- Not all frames contribute equally  
- Frames with stronger manipulation evidence are weighted higher  

---

## 7. Frequency Domain Feature Branch

A parallel branch extracts frequency-based features.

### Process:

1. Transform frames to frequency domain (DCT / FFT)  
2. Apply lightweight CNN for feature extraction  

### Captures:

- compression artifacts  
- blending inconsistencies  
- synthesis traces  

---

## 8. Feature Fusion

Final representation combines:

- Spatial features  
- Temporal features  
- Frequency features  

### Goal:

- Improve robustness against:
  - compression  
  - post-processing  
  - domain shift  

---

## 9. Controlled Partial Fine-Tuning

### Strategy:

- **Stage 1:** Backbone frozen  
- **Stage 2:** Only higher layers unfrozen  

### Purpose:

- Prevent catastrophic forgetting  
- Preserve generalizable features  

---

## 10. Score Generation

- Model outputs a **scalar detection score**

⚠️ Note:  
- This is not the final decision  

---

## 11. Forensic Calibration

Raw scores are converted into:

> **Likelihood Ratios (LR)**

### Hypotheses:

- **H1:** Video is manipulated  
- **H0:** Video is real  

### Formula:

LR = P(Evidence | H1) / P(Evidence | H0)

### Implementation:

- Calibration performed on **held-out dataset**
- Separate from training and testing data  

### Purpose:

- Provide probabilistic evidence  
- Enable court-admissible reporting  

---

## 12. Evaluation Framework

### A. Discrimination Metrics

- AUC  
- Equal Error Rate (EER)  
- Half Total Error Rate (HTER)  
- Calibration Cost (CLLR)

---

### B. Calibration Metrics

- CLLR (Log-Likelihood Ratio Cost)  
- Likelihood distribution analysis  

---

### C. Cross-Dataset Evaluation

- Evaluate performance across different datasets  
- Measure degradation under domain shift  
- Simulate real-world forensic conditions  

---

## 13. Data Splitting Strategy

Strict separation of:

- Training set  
- Validation set  
- Calibration set  
- Test set  

### Purpose:

- Prevent bias  
- Ensure reproducibility  
- Maintain forensic validity  

---

## 14. Key Design Principles

- Cross-dataset generalization > single-dataset accuracy  
- Video-level modeling > frame-level classification  
- Calibration > raw confidence scores  
- Multi-domain features > single modality  

---

## 15. Summary

The methodology proposes a:

> **Multi-stage, multi-modal deepfake detection framework integrating spatial, temporal, and frequency features, with likelihood ratio-based forensic calibration for reliable cross-dataset performance.**

---