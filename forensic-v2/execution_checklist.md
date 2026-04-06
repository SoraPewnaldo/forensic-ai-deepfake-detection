# V2.1 Go-Live Execution Checklist & Safety Boundaries

This document acts as the strict scientific referee during pipeline execution. It defines the exact bounds of mathematical stability ensuring the final paper remains publishable.

## 🔴 Stage 0: Sanity Check (Pre-Flight)

**Dataset Integrity**
- [x] FF++ splits loaded strictly from frozen JSON.
- [x] Verified `Train ∩ Val ∩ Cal ∩ Test = ∅` (zero subject leakage/overlap).
- [x] `Celeb-DF` strictly quarantined (zero reference in training loops).

**Tensor Definitions**
- [x] Sequence limit fixed: `T=16` frames.
- [x] Bounding constraint: Face crops exactly `256 x 256`.
- [x] Compression invariance: JPEG augment active at `p=0.8` bounding `Q=[20, 55]`.

**Architectural Integrity**
- [x] CLIP blocks `0–7` frozen; representations projected `768 -> 512`.
- [x] Frequency branch explicitly isolated to exact orthonormal DCT-II matrices (`256 -> 512`).

---

## 🔵 Stage 1: Base Training (FF++)
**Execution Command:** `python -m src.training.stage1_ffpp`

| Metric Target | Expected Stable Range | Scientific Red Flag 🚨 | Problem Indicates |
| :--- | :--- | :--- | :--- |
| **Train Loss** | Smooth, steady descent | Unstable / bouncing | LR is too high or optimizer failing |
| **Val AUC (FF++)** | `0.75` – `0.85` | `> 0.95` | Catastrophic Overfitting / Data Leakage |
| **Val HTER** | `0.20` – `0.30` | Stuck at `0.50` | Model has collapsed to a single class bias |

---

## 🔵 Stage 2: Hybrid Domain Adaptation
**Execution Command:** `python -m src.training.stage2_hybrid`

| Observation Target | Expected Behavior | Scientific Red Flag 🚨 | Problem Indicates |
| :--- | :--- | :--- | :--- |
| **FF++ AUC** | Slight dip (expected variance) | Total collapse | Catastrophic forgetting has occurred |
| **WildDeepfake AUC** | Steady increase | Remains stagnant | `BalancedHybridSampler` mathematically broken |
| **Loss Curve** | Smoother than Stage 1 | Spiking | Gradients colliding between domains |

---

## 🔴 Stage 3: Forensic Calibration
**Execution Command:** `python -m src.calibration.run_calibration`

| Metric Target | Expected Boundary | Scientific Red Flag 🚨 | Problem Indicates |
| :--- | :--- | :--- | :--- |
| **Temperature ($T$)** | `0.5` – `5.0` | $T \approx 10.0$ | Model is destructively overconfident |
| **CLLR (FF++ Cal)** | `< 0.5` | `> 0.6` | Calibration logic mapping is fundamentally failing |

---

## 🟢 Stage 4: Evaluation & Zero-Shot Generalization
**Execution Command:** `python -m src.evaluation.run_evaluation`

| Dataset | Expected AUC | Expected HTER | Expected CLLR | Red Flag 🚨 |
| :--- | :--- | :--- | :--- | :--- |
| **FF++** (In-Distribution) | `0.80` – `0.90` | `0.15` – `0.25` | — | — |
| **WildDeepfake** (Domain) | `0.75` – `0.85` | `0.20` – `0.30` | — | — |
| **Celeb-DF** (Zero-Shot) | **`0.70` – `0.80`** | `0.22` – `0.30` | **`< 0.4`** | `< 0.60` (No Generalization) <br> `> 0.85` (Leakage Occurred) <br> `CLLR > 0.5` (Not Forensic) |

---

## 🟣 Stage 5: Scientific Ablation Training
**Protocol Directive:** We do NOT use runtime toggles. We spin up completely isolated training branches across identical seeds.

1.  **Model A:** Spatial Backbone Only
2.  **Model B:** Spatial + Temporal
3.  **Model C:** Full Tri-Branch Model (Spatial + Frequency + Temporal)

**Required Analytical Outcome:** Performance must scale symmetrically (`A < B < C`). If Frequency alone equals Full Model, Frequency is exploiting a systemic dataset flaw.
