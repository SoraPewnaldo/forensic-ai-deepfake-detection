# FORENSIC-AI (V2) — Project Roadmap & Progress Log

> [!IMPORTANT]
> **V2 REBUILD INITIATIVE:**
> After a successful Phase 4 Calibration and a problematic Phase 5 Zero-Shot Evaluation in V1, we deliberately wiped the execution environment to build a strictly typed, strictly cropped, defensively engineered V2 paradigm.
> 
> **MANDATORY PRE-FLIGHT CHECK:**
> Before running ANY command, you MUST ensure the virtual environment is activated:
> ⚙️ `E:\Pojects\workspace\venvs\forensic-ai\Scripts\activate`

## 📊 V2 Project Completion Status: [░░░░░░░░░░] 0%

---

## 🛑 POST-MORTEM LOG (V1 FAILURES TO SOLVE IN V2)

Before we start V2, we officially log the failures of V1 to ensure defensive architectural programming:

| Incident Name | Root Cause | V2 Solution |
| :--- | :--- | :--- |
| **1. The Compression Floor Collapse (Inversion Bias)** | The model evaluated on raw MP4 files rather than cropped faces. Celeb-DF fakes are pristine; Celeb-DF real videos (YouTube) are highly compressed. The model learned to classify H.264 compression blocks as "Fake" rather than facial structural artifacts. | **Strict Crop Enforcement:** The V2 DataLoader will physically reject any raw video file. The entire pipeline runs *exclusively* on pre-extracted 224x224 bounding box crops from `data/processed`. |
| **2. Frame Count Starvation (8 vs 32)** | Celeb-DF evaluation was executed using only 8 temporally spaced frames. Due to motion blur and occlusions, a single bad RetinaFace track ruined 12.5% of the video score, resulting in low ensemble confidence. | **Universal 32-Frame Doctrine:** `config.py` will mandate `32` frames for both train and test sets. We will implement a strict tracking confidence threshold (e.g., `>0.95`) to drop distorted faces. |
| **3. Dataset Path Poisoning** | Complex Windows network paths and recursive filesystem wildcards led to loading mixed formats and failing validation splits over backslashes (`\`). | **`pathlib` Absolute Standardization:** Configuration variables will use pure, resolved explicit `Path().resolve()` objects. No relative wildcards without rigorous logging. |

---

## 🟢 PHASE 1: V2 Infrastructure & Strict Ingestion
- **Task 1.1:** Build `src/config.py` with defensive parameters (32 frames, absolute dataset routing). — **PENDING ⚪**
- **Task 1.2:** Implement strict dataset classes that *only* ingest `processed/` directories. — **PENDING ⚪**
- **Task 1.3:** Re-download Celeb-DF v2 and dynamically extract 32 frames per video featuring bounding box thresholding (>0.95). — **PENDING ⚪**

## ⚪ PHASE 2: ViT Architecture & Fusion Blueprint
- **Task 2.1:** Re-implement ViT Spatial Backbone (Freeze blocks to protect VRAM). — **PENDING ⚪**
- **Task 2.2:** Re-implement the Frequency (DCT) Branch to capture deep neural-texture artifacts. — **PENDING ⚪**
- **Task 2.3:** Re-implement Temporal Attention layers. — **PENDING ⚪**

## ⚪ PHASE 3: Base Training (FF++ Compression Generalization)
- **Task 3.1:** Execute Stage 1 Training on `datasets/ffpp`. — **PENDING ⚪**
- **Task 3.2:** Execute Extreme Compression Augmentation (`p=0.8`) natively to destroy background biases. — **PENDING ⚪**

## ⚪ PHASE 4: Domain Adaptation (WildDeepfake)
- **Task 4.1:** Resume Stage 2 Fine-Tuning using `datasets/wilddeepfake`. — **PENDING ⚪**
- **Task 4.2:** Ensure strict gradient accumulation constraints to maintain 6GB VRAM throughput (>7 it/s). — **PENDING ⚪**

## ⚪ PHASE 5: Forensic Calibration & Zero-Shot
- **Task 5.1:** Re-evaluate Calibration (LLR, CLLR < 0.5) using Isotonic Regression. — **PENDING ⚪**
- **Task 5.2:** Final Zero-Shot Evaluation on Celeb-DF 32-Frame Crops (Target AUC > 0.75). — **PENDING ⚪**

---

## 🚀 CURRENT ACTIVE STEP:
> **Task 1.1: Build src/config.py**
> (Pending user confirmation after Celeb-DF re-acquisition)

---
*Last Updated: 2026-04-05*
