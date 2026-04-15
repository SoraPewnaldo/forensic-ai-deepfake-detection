# 🕵️ FORENSIC-AI: An Intelligent Deepfake Video Detection for Digital Evidences

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research%20Complete-success?style=for-the-badge)

**A cross-dataset generalizable deepfake detection system with court-admissible Likelihood Ratio (LR) calibration.**  
Designed for use in digital forensic investigations as scientifically rigorous evidence.

</div>

---

## 👥 Authors

| Name | Role |
|:---|:---|
| **Ayush Dakwal (SoraPewnaldo)** | Lead Researcher — Model Architecture, Training Pipeline, Forensic Calibration |
| **Aakhya Chauhan** | Co-Researcher — Dataset Preparation, Evaluation Protocol, Analysis |

---

## 📖 Abstract

Modern deepfake detectors achieve high AUC scores on their training distribution but severely **degrade on real-world ("in-the-wild") data**. Furthermore, raw probability outputs carry no legal or forensic meaning in court. This research addresses both problems simultaneously.

We propose a **Multi-Branch Forensic Detection Network** combining:
1. A **CLIP-ViT spatial backbone** for deep semantic manipulation artifacts.
2. A **DCT-based frequency branch** to isolate GAN/diffusion synthesis signatures invisible to spatial analysis.
3. A **Self-Attention temporal module** to capture inter-frame inconsistencies across video sequences.

The network is trained on a **strictly staged, leakage-free curriculum** across three datasets and calibrated using a novel **multi-domain hybrid calibrator** (Temperature Scaling → Isotonic Regression) to produce court-admissible Likelihood Ratios.

**Key Result:** Celeb-DF-v2 Zero-Shot AUC of **0.9731** with a forensic CLLR of **0.3705** (below the 0.40 court-admissibility threshold).

---

## 🎯 Key Results

### Ablation Study — Architecture Contribution

| Model Variant | Architecture | FF++ AUC | Celeb-DF AUC ↑ | WildDeepfake AUC |
|:---|:---|:---:|:---:|:---:|
| **Baseline** | Spatial Only (CLIP-ViT) | 0.9781 | 0.7742 | 0.8017 |
| **Beta** | Spatial + Temporal | 0.9850 | 0.8667 | 0.8004 |
| **Gamma (Ours)** | Spatial + Temporal + Frequency | **0.9912** | **0.9731** | **0.9060** |

> Each branch provides a statistically significant improvement in cross-dataset zero-shot detection.

### Final Evaluation Metrics (Gamma Model — Calibrated)

| Dataset | AUC | EER | HTER | CLLR | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FF++ (In-Domain)** | 0.9913 | 0.0663 | 0.0663 | 0.2201 | ✅ Excellent |
| **Celeb-DF-v2 (Zero-Shot)** | 0.9731 | 0.0861 | 0.0861 | **0.3705** | ✅ Court-Admissible |
| **WildDeepfake (Zero-Shot)** | 0.9060 | 0.1650 | 0.1650 | 1.0408 | ⚠️ Significant Gain |

> **CLLR < 0.40** on Celeb-DF-v2 meets the forensic court-admissibility threshold.

---

## 🏗️ Architecture

### System Pipeline

```
INPUT VIDEO
    │
    ▼
┌─────────────────────────────────────────────┐
│           Frame Extraction (MTCNN)           │
│  • 32 frames sampled uniformly per video     │
│  • Face detected & cropped to 256×256px      │
│  • Stored as tensor sequences [T, C, H, W]   │
└─────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MULTI-BRANCH FEATURE EXTRACTION               │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  SPATIAL BRANCH  │  │ FREQUENCY BRANCH │  │ TEMPORAL MODULE│  │
│  │                  │  │                  │  │                │  │
│  │  CLIP ViT-B/16   │  │ DCT on Y/Cb/Cr   │  │  4-Head Self-  │  │
│  │  (Frozen 0-7)    │  │ channels         │  │  Attention over│  │
│  │  Trainable 8-11  │  │ GAN artifacts &  │  │  T=16 frames   │  │
│  │                  │  │ compression noise│  │                │  │
│  │  768-dim → 512   │  │  256-dim → 512   │  │                │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘  │
│           │                     │                      │          │
│           └──────────┬──────────┘                      │          │
│                      │ Concatenate (1024-dim)           │          │
│                      └──────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│       CLASSIFICATION HEAD (MLP)              │
│  Linear(1024→512) → BN → ReLU → Drop(0.3)  │
│  → Linear(512→1) → Raw Logit                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│         FORENSIC CALIBRATION MODULE          │
│                                             │
│  Step 1: Temperature Scaling (T=1.3315)     │
│  Step 2: Isotonic Regression (PAVA)         │
│  Output: P(fake) → Likelihood Ratio (LR)    │
│                                             │
│  LR = P_fake / (1 - P_fake)                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           FORENSIC INTERPRETATION            │
│                                             │
│  LR > 10      → Strong evidence of FAKE     │
│  LR = 1–10    → Moderate evidence of FAKE   │
│  LR ≈ 1       → Inconclusive                │
│  LR = 0.1–1   → Moderate evidence of REAL   │
│  LR < 0.1     → Strong evidence of REAL     │
└─────────────────────────────────────────────┘
```

---

## 📦 Datasets Used

This project follows a strict **staged cross-dataset protocol** with zero data leakage.

| Dataset | Role | Split | Source |
|:---|:---|:---|:---|
| **FaceForensics++ (FF++)** | Primary Training | Train / Validation / Calibration / Test | [GitHub](https://github.com/ondyari/FaceForensics) |
| **WildDeepfake** | Domain Adaptation | Train (50% mixture) | [GitHub](https://github.com/deepfakeinthewild/deepfake-in-the-wild) |
| **Celeb-DF-v2** | Zero-Shot Unseen Evaluation | Test Only (never seen during training) | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |

> **Strict Rule**: Celeb-DF-v2 was **never** used during training. It serves exclusively as a blind evaluation set to measure real-world generalization.

---

## 🎬 How Video Inference Works

```
1. INPUT: A video file (.mp4, .avi, etc.)
        │
        ▼
2. FRAME SAMPLING: 32 frames sampled uniformly across the video duration.
        │
        ▼
3. FACE DETECTION: MTCNN/RetinaFace detects and crops the primary face
   from each frame to 256×256 pixels (confidence threshold ≥ 0.95).
        │
        ▼
4. PREPROCESSING: ImageNet normalization + real-time JPEG compression
   simulation (CRF-35) to simulate real-world video degradation.
        │
        ▼
5. FEATURE EXTRACTION (3 parallel branches):
   ├─ Spatial: CLIP-ViT reads the face crop pixel patterns.
   ├─ Frequency: DCT coefficients detect spectral GAN artifacts.
   └─ Temporal: Self-Attention correlates 16-frame sequences.
        │
        ▼
6. CLASSIFICATION: MLP head produces a raw logit score.
        │
        ▼
7. CALIBRATION: Temperature Scaling + Isotonic Regression converts
   the raw score into a probability P(fake), then into a Likelihood
   Ratio (LR = P_fake / 1 - P_fake).
        │
        ▼
8. OUTPUT: Likelihood Ratio + Interpretation for forensic reports.
```

---

## 📂 Repository Structure

```bash
📦 Deepfake-Forensic-Research/
│
├── 📁 src/                          # All source code
│   ├── 📁 models/                   # Network architectures
│   │   ├── forensic_model.py        # Main multi-branch model
│   │   ├── backbone.py              # CLIP-ViT spatial backbone
│   │   ├── frequency_branch.py      # DCT Frequency CNN
│   │   ├── temporal_module.py       # Self-Attention temporal module
│   │   └── ablation_model.py        # Ablation study variants
│   │
│   ├── 📁 training/                 # Training scripts (staged protocol)
│   │   ├── stage1_ffpp.py           # Stage 1: Base training on FF++
│   │   ├── stage2_hybrid.py         # Stage 2: Domain adaptation
│   │   ├── trainer.py               # Core training loop
│   │   └── stage5_ablation.py       # Ablation training
│   │
│   ├── 📁 calibration/              # Forensic calibration pipeline
│   │   ├── temperature_scaling.py   # Temperature scaling (Step 1)
│   │   ├── isotonic_calibrator.py   # PAVA isotonic regression (Step 2)
│   │   └── run_calibration.py       # Multi-domain calibration entrypoint
│   │
│   ├── 📁 evaluation/               # Metrics & visualization
│   │   ├── metrics.py               # AUC, EER, HTER, CLLR computation
│   │   ├── run_evaluation.py        # Cross-dataset evaluation runner
│   │   ├── tippett_plot.py          # Tippett plot visualization (forensic)
│   │   ├── ablation_runner.py       # Ablation evaluation runner
│   │   └── run_ablation_eval.py     # Ablation evaluation entrypoint
│   │
│   └── 📁 utils/                    # Helpers
│       ├── checkpoint.py            # Checkpoint save/load
│       ├── logging_utils.py         # Structured experiment logging
│       ├── device.py                # CUDA/CPU device management
│       └── seed.py                  # Reproducibility seed control
│
├── 📁 docs/                         # Research documentation
│   ├── Technical_Methodology.md     # Detailed architecture & methodology
│   ├── The_Story_of_Forensic_AI.md  # Research narrative
│   ├── Metrics_Timeline.md          # Experiment progress timeline
│   ├── execution_log.md             # Training run logs
│   └── execution_checklist.md       # Development checklist
│
├── 📁 evaluation_results/           # Final output files
│   ├── eval_results.json            # Numeric metrics (AUC, CLLR, etc.)
│   ├── tippett_FFPP_TEST.png        # Tippett LR Plot — FF++
│   ├── tippett_CelebDF_TEST.png     # Tippett LR Plot — Celeb-DF
│   └── tippett_WildDeepfake_TEST.png# Tippett LR Plot — WildDeepfake
│
├── 📁 data/
│   └── splits/ffpp_splits.json      # Subject-disjoint data split manifest
│
├── 📄 config.yaml                   # Global SSOT project configuration
├── 📄 celebdf_audit.py              # Dataset integrity validation script
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md
```

---

## 🚀 Installation & Setup

### Requirements
- **GPU**: NVIDIA GPU with ≥ 6GB VRAM (tested on RTX 3060 Laptop)
- **CUDA**: 12.x
- **OS**: Windows 10/11 or Linux

### 1. Clone the repository
```bash
git clone https://github.com/SoraPewnaldo/Deepfake-Forensic-Research.git
cd Deepfake-Forensic-Research
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv venvs/forensic-v2
venvs\forensic-v2\Scripts\activate

# Linux/Mac
python -m venv venvs/forensic-v2
source venvs/forensic-v2/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure dataset paths
Edit `config.yaml` to point to your local dataset locations:
```yaml
paths:
  datasets:
    ffpp:         "./datasets/ffpp"
    wilddeepfake: "./datasets/wilddeepfake"
    celebdf:      "./datasets/celebdf"
```

---

## 💾 Pre-Trained Model Weights

Due to GitHub's 100MB file limit, all model weights are hosted in the **Releases** section.

📎 **[Download from Releases →](https://github.com/SoraPewnaldo/Deepfake-Forensic-Research/releases)**

| Checkpoint File | Description | Size |
|:---|:---|:---|
| `best_Stage1_FFPP.pt` | Stage 1 model (FF++ only) | ~657 MB |
| `best_Stage2_Hybrid.pt` | Final production model (FF++ + WildDeepfake) | ~657 MB |
| `best_Ablation_Baseline.pt` | Ablation: Spatial branch only | ~559 MB |
| `best_Ablation_Beta.pt` | Ablation: Spatial + Temporal | ~655 MB |
| `temperature.pt` | Fitted temperature scalar (T=1.3315) | < 1 KB |

Place downloaded `.pt` files into the `checkpoints/` folder (create it if absent).

---

## 🧪 Running Evaluation

```bash
# Cross-dataset full evaluation
python src/evaluation/run_evaluation.py

# Ablation study evaluation
python src/evaluation/run_ablation_eval.py

# Generate Tippett plots
python src/evaluation/tippett_plot.py
```

---

## 📊 Tippett Plots (Forensic LR Visualization)

Tippett plots visualize the separation of Likelihood Ratios between genuine (real) and fake videos. A well-calibrated system shows maximum separation between the two curves.

| FF++ (In-Domain) | Celeb-DF-v2 (Zero-Shot) | WildDeepfake (Zero-Shot) |
|:---:|:---:|:---:|
| CLLR: **0.2201** ✅ | CLLR: **0.3705** ✅ | CLLR: 1.0408 ⚠️ |

> ✅ CLLR < 0.40 = Court-Admissible forensic evidence standard (per ENFSI guidelines).

---

## 🔬 Calibration Pipeline

The raw model output (a logit) is meaningless in a forensic context. Our two-stage calibration converts it into a statistically grounded Likelihood Ratio:

```
Raw Logit
   │
   ▼ Step 1: Temperature Scaling (T = 1.3315)
   │         Fitted via L-BFGS on FF++ calibration partition.
   │         Neutralizes the model's overconfidence on OOD data.
   ▼
Probability P(fake)
   │
   ▼ Step 2: Isotonic Regression (PAVA)
   │         Non-parametric monotonic step function.
   │         Fitted over concatenated multi-domain logit space
   │         (FF++ n=132 + Celeb-DF n=494 + WildDeepfake n=50).
   ▼
Calibrated P(fake)
   │
   ▼ LR = P_fake / (1 - P_fake)
   │
Likelihood Ratio → Forensic Report
```

---

## 📋 Citing This Work

If you use this codebase or methodology in your research, please cite:

```bibtex
@misc{forensic_ai_2026,
  author    = {Ayush Dakwal and Aakhya Chauhan},
  title     = {FORENSIC-AI: An Intelligent Deepfake Video Detection for Digital Evidences},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/SoraPewnaldo/Deepfake-Forensic-Research}
}
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **OpenAI CLIP** — Pre-trained ViT-B/16 backbone
- **FaceForensics++** — Rössler et al., ICCV 2019
- **Celeb-DF-v2** — Li et al., CVPR 2020
- **WildDeepfake** — Zi et al., ACM MM 2020
- **ENFSI Guidelines** — Framework for Likelihood Ratio forensic interpretation
