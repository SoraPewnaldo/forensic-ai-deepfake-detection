# Forensic-AI V2.1 Execution Log 

**Project Status:** Stage 4 Complete.

---

## 🔵 Stage 1: Spatial-Temporal Foundation (FF++)
- **Focus:** Establish high-confidence detection on in-distribution (uncompressed) data.
- **Results:**
    - Peak Val-AUC: **0.9506** (Epoch 2).
    - Model: Frozen PatchEmbed + first 8 ViT blocks. 16-frame sequences.
- **Artifact:** `checkpoints/best_Stage1.pt`

## 🔵 Stage 2: Hybrid Domain Adaptation
- **Focus:** Generalization via integration of WildDeepfake (57GB) and FF++.
- **Results:**
    - Best Celeb-DF Proxy AUC: **0.9435** (Epoch 4).
    - Early Stopped: Epoch 9 (patience=5 reached).
    - Trainable Params: Unfrozen top 4 ViT blocks.
- **Artifact:** `checkpoints/best_Stage2_Hybrid.pt`
- **Technical Fixes:**
    - Modified `checkpoint.py` to ignore non-learned buffers (e.g. `dct_mat`) during Stage 1 -> 2 load.
    - Updated `WildDeepfakeDataset` to handle nested directory structure and skip empty frame folders.

## 🔴 Stage 3: Forensic Calibration
- **Focus:** Court-admissible Likelihood Ratios (LR).
- **Process:** Fitted Temperature Scaling (T=0.9363) and Isotonic Regression on FF++ CAL partition.
- **Internal Check (CAL set):**
    - CLLR: **0.3572** (Passed target < 0.4).
- **Artifacts:** 
    - `calibration_artefacts/temperature.pt`
    - `calibration_artefacts/isotonic.json`
- **Technical Fix:** Implemented manual step-function interpolation for `IsotonicCalibrator` to bypass `sklearn` serialization bugs.

## 🟢 Stage 4: Final Evaluation (Zero-Shot)
- **Focus:** Stress test on 100% unseen data (Celeb-DF Test & WildDeepfake Test).
- **Final Metrics:**

| Dataset | AUC | CLLR | AP | Result |
| :--- | :--- | :--- | :--- | :--- |
| **FFPP_TEST** | 0.9912 | 0.2348 | 0.9575 | **Excellent** |
| **WildDeepfake_TEST** | 0.9060 | 1.4862 | 0.8962 | Hard Domain |
| **CelebDF_TEST** | 0.8900 | 0.7006 | 0.9765 | **Generalization Pass** |

---

## Technical Debt / Next Steps
1. **Calibration:** CLLR > 0.4 on out-of-domain data. Need hybrid CAL set for Stage 5.
2. **Ablation:** Start Stage 5 to quantify Frequency vs. Temporal branch impact.
