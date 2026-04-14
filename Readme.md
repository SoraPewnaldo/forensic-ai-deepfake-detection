# 🕵️‍♂️ Deepfake Forensic Research

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-cu121-EE4C2C.svg)
![Release](https://img.shields.io/badge/Release-V2-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

> A cross-dataset generalizable deepfake detection system with likelihood ratio-based forensic calibration suitable for court-admissible digital evidence analysis.

## 📖 Overview

Deepfake detection systems often struggle to generalize out-of-the-box in real-world ("in-the-wild") scenarios. Moreover, typical detection models output uncalibrated probabilities, which hold no weight in digital forensic investigations. 

This project solves both problems by using a **Multi-Branch Architecture** (Spatial, Temporal, Frequency) coupled with **Forensic Calibration** (Platt/Temperature Scaling) to produce standardized, interpretable Likelihood Ratios (CLLR < 0.4).

### Key Features
- **Spatial Branch**: CLIP-ViT Backbone targeting deep spatial inconsistencies.
- **Temporal Module**: Attention-based weighting capturing unnatural intra-frame motions.
- **Frequency Branch**: DCT/FFT analysis specifically isolating GAN and compression artifacts.
- **Forensic Validation**: Translates raw binary probabilities into Likelihood Ratios used in court.
- **Cross-Dataset Generalization**: Evaluated on FF++, WildDeepfake, and Celeb-DF-v2.

---

## 📂 Project Structure

```bash
📦 Deepfake-Forensic-Research
├── 📁 src                   # Main source code
│   ├── 📁 models            # Network architectures (Backbone, Frequency, Temporal, Fusion)
│   ├── 📁 training          # Stage 1 and Stage 2 training scripts
│   ├── 📁 calibration       # Forensic Calibration components
│   ├── 📁 evaluation        # Evaluation matrices, visualizers (Tippett Plots)
│   └── 📁 utils             # Data loaders, metric calculations, and logs
├── 📁 docs                  # Development history, logs, methodology explanations 
├── 📁 evaluation_results    # Results and output plots (Tippett, ROC, etc.)
├── 📄 config.yaml           # Global configurations
├── 📄 requirements.txt      # Python dependencies
└── 📄 README.md
```

---

## 🚀 Installation & Setup

You will need a GPU-enabled machine with at least 6GB of VRAM (e.g., RTX 3060).

**1. Clone the repository**
```bash
git clone https://github.com/SoraPewnaldo/Deepfake-Forensic-Research.git
cd Deepfake-Forensic-Research
```

**2. Create a virtual environment**
```bash
python -m venv venvs/forensic-ai
source venvs/forensic-ai/bin/activate  # On Windows use: venvs\forensic-ai\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 💾 Model Checkpoints

Due to GitHub's file size limits, the final trained `.pt` model weights are hosted in the **Releases** section of this repository.

*   Download the model weights here: **[🔗 Releases Page](https://github.com/SoraPewnaldo/Deepfake-Forensic-Research/releases)**
*   Place the downloaded `.pt` files inside the `checkpoints/` folder.

*(Note: Checkpoints include `best_Stage1_FFPP.pt` and `best_Stage2_Hybrid.pt`)*

---

## 📊 Methodology & Training Stages

This project utilizes a strictly staged training protocol to ensure domain adaptation and dataset leak prevention:

1. **Stage 1 (Base Training - FF++)**: The network's temporal and frequency layers are trained heavily on FaceForensics++.
2. **Stage 2 (Domain Adaptation - WildDeepfake)**: Layers are slowly unfrozen for in-the-wild fine-tuning.
3. **Stage 3 (Evaluation Only - Celeb-DF-v2)**: Strict unseen validation. No training occurs on this set. 
4. **Stage 4 (Calibration)**: Logistic Regression and KD-Estimation convert raw outputs to a Likelihood Ratio.

For detailed technical methodology, refer to the `docs/Technical_Methodology.md` file.

---

## 📈 Evaluation Metrics

The system is measured against two core paradigms:

**1. Discrimination Metrics:**
- **AUC, HTER, EER**: Evaluates how well the model separates real vs. fake videos.

**2. Calibration Metrics (Forensic standard):**
- **CLLR (Cost of Log-Likelihood Ratio)**: Defines court-admissibility.
- **Tippett Plots**: Demonstrates the separation of likelihood ratios visually.

---

## 👨‍💻 Author

- **Sora Pewnaldo (Ayush Dakwal)** - *Machine Learning & Digital Forensics*

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
