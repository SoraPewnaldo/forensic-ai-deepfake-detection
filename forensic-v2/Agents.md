# 🤖 AGENT INSTRUCTIONS — FORENSIC-AI

---

## ⚠️ ENVIRONMENT RULES (STRICT)

* ALWAYS activate virtual environment before running code:

```bash
source ~/venvs/forensic-ai/bin/activate
```

* NEVER use system Python
* NEVER install packages globally

---

## 💻 SYSTEM CONSTRAINTS

* GPU: RTX 3060 (6GB VRAM)
* Must optimize memory usage:

  * batch size ≤ 8
  * prefer mixed precision (FP16)

---

## 🧠 PROJECT RULES

* Follow modular structure:

```
src/
 ├── datasets/
 ├── models/
 ├── training/
 └── utils/
```

* Do NOT write monolithic scripts
* Keep dataset, model, and training separated

---

## 📊 METHODOLOGY (STRICT)

Pipeline must follow:

1. Frame extraction
2. Spatial feature extraction (CNN/ViT)
3. Temporal aggregation (attention)
4. Frequency features (DCT/FFT)
5. Feature fusion
6. Classification
7. Likelihood Ratio calibration

---

## ⚠️ FORBIDDEN SHORTCUTS

* Do NOT simplify to binary classifier only
* Do NOT ignore calibration stage
* Do NOT train and test on same dataset

---

## 🧪 TRAINING PROTOCOL

* Stage 1: Train on FaceForensics++
* Stage 2: Fine-tune on WildDeepfake
* Stage 3: Test on Celeb-DF (no training)

---

## 📏 EVALUATION

Must include:

* AUC
* HTER / EER
* CLLR
* Tippett plots

---

## 🧠 CODING RULES

* Write clean, modular PyTorch code
* Include comments explaining logic
* Ensure reproducibility

---

## 🎯 GOAL

Build a **forensically reliable deepfake detection system**
with strong cross-dataset generalization and LR calibration.

---
