# FORENSIC-AI V2 — Failure-Aware System Reconstruction
### Senior Research Scientist Review | Stage 0 Ground-Up Redesign

---

> [!CAUTION]
> This document is the **single source of truth** for V2. Every architectural decision is grounded in a documented V1 failure. No design choice may be made without citing a lesson from Phase 1 of this document.

---

# PHASE 1 — FAILURE EXTRACTION

## A. Methodological Failures

### A1 — Temporal Frame Starvation During Evaluation
**What happened:** Celeb-DF was extracted at 8 frames/video while FF++ training used 32.  
**Why it happened:** The extraction script for Celeb-DF was a separate script (`process_celebdf.py`) with a hardcoded `n_frames=8` parameter that was never synchronized with the training config.  
**Impact:** With 8 frames, a single poor RetinaFace track corrupts 12.5% of the video-level score. This artificially deflated the evaluation AUC and introduced high variance between runs, making the results scientifically unreproducible.

### A2 — Celeb-DF Treated as Zero-Shot Without Domain Profiling
**What happened:** Celeb-DF was designated as pure zero-shot from the beginning without any domain profiling.  
**Why it happened:** The assumption was that "unseen = zero-shot," but the dataset has fundamentally different compression characteristics than the training sets.  
**Impact:** This is not a fair zero-shot test — it is a compression-domain shift test. The published AUC of 0.61 is not a clean generalization measurement; it is a measurement of how much compression invariance we achieved. This conflation would immediately fail IEEE peer review.

### A3 — Dataset Mixing Without Class Balance Audit
**What happened:** Stage 2 mixed FF++ (equal real/fake 50:50) with WildDeepfake (naturally imbalanced, unknown ratio) without computing the merged class prior.  
**Why it happened:** The hybrid loader concatenated datasets at the Dataset level without auditing the resultant `[real:fake]` distribution.  
**Impact:** The model trained with an implicit `P(fake) >> 0.5` prior, causing the calibrator to inherit a biased score distribution. This directly inflated the CLLR from a theoretically-achievable `<0.3` to the reported `0.489`.

---

## B. Model Failures

### B1 — Stage 2 AUC 0.91 Does NOT Mean Generalization
**What happened:** We celebrated Stage 2 AUC of 0.9119 without recognizing it was measured on WildDeepfake validation — the same domain as training.  
**Why it happened:** The validation set was drawn from the same WildDeepfake distribution as training. This is in-distribution validation, not cross-dataset.  
**Impact:** The model was severely overfit to the WildDeepfake compression signature. The true cross-dataset AUC was 0.4511 (inverted), meaning the model had learned a shortcut, not forensic features. This is a **critical research validity failure**.

### B2 — The Compression Shortcut (Clever Hans Failure)
**What happened:** The model learned to classify H.264 compression blocking artifacts as "Fake" because WildDeepfake fakes were stored at lower bitrate than real YouTube videos in Celeb-DF.  
**Why it happened:** No compression invariance augmentation was applied during Stages 1 and 2. The model was never forced to learn facial structure rather than background entropy.  
**Impact:** The model achieved *negative* forensic utility. It was worse than random on an unseen dataset. This invalidates Stages 1 and 2 as presented.

### B3 — Frequency Branch Was a Compression Detector, Not a Manipulation Detector
**What happened:** The DCT branch was intended to capture GAN and blending artifacts, but by training on compressed fakes vs. uncompressed reals, it became a compression quality estimator.  
**Why it happened:** The frequency branch was added as a genuine architectural enhancement, but the data it was trained on was confounded by compression bias.  
**Impact:** The frequency branch amplified the shortcut rather than providing independent forensic evidence.

### B4 — Temporal Attention Had No Empirical Contribution Measurement
**What happened:** Temporal attention was implemented and trained but its contribution was never ablated or measured independently.  
**Impact:** We cannot quantify whether the model is a truly temporal system or just a spatial system with unnecessary complexity.

---

## C. Calibration Failures

### C1 — CLLR of 0.489: Below Forensic Standard
**What happened:** Best achieved CLLR was 0.489 (Isotonic Regression). The forensic standard for reliable court testimony is CLLR < 0.3.  
**Why it happened:** The calibration input scores came from a model trained with a compression shortcut. Calibrating a biased model does not remove the bias — it scales it differently.  
**Impact:** An LR output from a CLLR-0.489 system cannot be used in court. Reports would fail Daubert standard challenge.

### C2 — Calibration Data Leakage (Initial Version)
**What happened:** The first calibration run used `split='validation'` — the same data used to select the best model checkpoint in Stage 2.  
**Why it happened:** The calibration script reused the convenient `validation` split handle without recognizing the contamination.  
**Impact:** The initial CLLR of 0.575 was optimistic. The calibrator was tested on data it had indirectly seen during model selection.

### C3 — Overconfident Score Distribution
**What happened:** The model output raw logit scores without temperature scaling. The sigmoid output clustered near 0 and 1.  
**Why it happened:** No temperature scaling or label smoothing was applied during training.  
**Impact:** Isotonic regression could not meaningfully map an already-saturated distribution.

---

## D. Evaluation Failures

### D1 — Raw MP4 Evaluation on Celeb-DF (Critical)
**What happened:** The first Celeb-DF evaluation decoded raw 1080p MP4 files and resized to 224x224 for the model, rather than using face crops.  
**Why it happened:** The evaluation script loaded videos using `BaseDeepfakeDataset` without overriding `processed_dir`.  
**Impact:** The model evaluated a 20-40px face region embedded in irrelevant background. Scientifically invalid comparison.

### D2 — No Tippett Plot Generated
**What happened:** Tippett plots were never generated. Only scalar metrics (AUC, EER) were computed.  
**Impact:** We cannot visually assess LR overlap between real and fake distributions. Mandatory for forensic science publication.

### D3 — No HTER Computed
**What happened:** Half Total Error Rate was never reported.  
**Impact:** HTER is the standard operational metric in biometrics and forensics (ISO/IEC 19795). Its absence weakens comparative claims significantly.

### D4 — Threshold Bias From Class Imbalance
**What happened:** Decision threshold was 0.5 but training data had implicit prior `P(fake) >> 0.5`.  
**Impact:** Inflated TPR at the cost of FPR. The 0.5-threshold accuracy was meaningless as a forensic indicator.

---

## E. Engineering Failures

### E1 — Config Was Not the Single Source of Truth
**What happened:** Frame counts, dataset paths, and hyperparameters were scattered across individual script files. `process_celebdf.py` hardcoded `n_frames=8`; `stage3_generalization.py` silently omitted `processed_dir`.  
**Impact:** Silent misconfiguration contaminated training and evaluation without any runtime error.

### E2 — Windows Encoding Crashes
**What happened:** Multiple training runs crashed mid-epoch due to `UnicodeEncodeError` from emoji characters on Windows cp1252.  
**Impact:** Multiple training epochs were lost. Results were not deterministic.

### E3 — No Fixed Random Seed Enforcement
**What happened:** Random seeds were set in some scripts but not enforced globally.  
**Impact:** Results between runs were not perfectly reproducible.

### E4 — No Checkpoint Validation Step
**What happened:** `dct_matrix_t` key mismatch error triggered at Stage 3 load time. Architecture was inconsistent across stages.  
**Impact:** Training may have resumed from a partially-loaded state.

---

# PHASE 2 — ROOT CAUSE ANALYSIS

| # | Failure | Root Cause | Severity |
|---|---------|------------|----------|
| A1 | 8-frame evaluation starvation | Engineering: Config not centralized | Moderate |
| A2 | Celeb-DF not domain-profiled | Methodological: Weak cross-dataset assumption | Moderate |
| A3 | No class balance audit | Methodological: Dataset protocol violation | **Critical** |
| B1 | Stage 2 in-distribution AUC celebrated | Evaluation mistake | **Critical** |
| B2 | Compression shortcut (Clever Hans) | Data bias + Training strategy | **Critical** |
| B3 | Frequency branch reinforced shortcut | Model design + Data bias | **Critical** |
| B4 | Temporal attention not ablated | Evaluation mistake | Minor |
| C1 | CLLR 0.489, not <0.3 | Upstream model bias (B2, B3) | **Critical** |
| C2 | Calibration data leakage | Evaluation mistake: split contamination | **Critical** |
| C3 | Overconfident scores | Training strategy: no temperature/smoothing | Moderate |
| D1 | Raw MP4 evaluation | Engineering: Config not enforced | **Critical** |
| D2 | No Tippett plots | Evaluation: missing forensic outputs | Moderate |
| D3 | No HTER | Evaluation: incomplete metric suite | Moderate |
| D4 | Threshold bias | Training strategy: no prior correction | Moderate |
| E1 | Config not single source | Engineering: architecture | **Critical** |
| E2 | Windows encoding crashes | Engineering: platform portability | Moderate |
| E3 | Non-reproducible seeds | Engineering: reproducibility | Moderate |
| E4 | No checkpoint validation | Engineering: architecture integrity | Moderate |

---

# PHASE 3 — V2 DESIGN PRINCIPLES

| # | Principle | Fixes |
|---|-----------|-------|
| P1 | **Config Supremacy:** `config.yaml` is the ONLY source of every hyperparameter, path, and frame count. No hardcoding. | E1, A1 |
| P2 | **Crop Mandate:** DataLoader raises `RuntimeError` if it finds a raw `.mp4` where a crop directory is expected. | D1 |
| P3 | **Compression Invariance First:** JPEG augmentation (q=20–55) applied to 80%+ of all frames from epoch 1, stage 1. | B2, B3 |
| P4 | **Balance Before Training:** `[n_real : n_fake]` ratio computed, logged, and enforced via weighted sampler before any training stage. | A3, D4 |
| P5 | **Calibration Partition is Sacred:** A 10% subject-disjoint calibration partition is allocated once, serialized to JSON, and never touched by training or validation. | C2 |
| P6 | **Temperature Scaling Before Calibration:** Logits are passed through a learned temperature `T` before the calibration module. No raw sigmoid inputs. | C3 |
| P7 | **Cross-Dataset is the Primary Metric:** Celeb-DF zero-shot AUC determines checkpoint saving. In-distribution AUC is a secondary sanity check only. | B1 |
| P8 | **Tippett Plots are Mandatory:** Every evaluation run generates a Tippett plot. No evaluation is accepted without one. | D2 |
| P9 | **Ablation Before Integration:** Each module (temporal, frequency) must improve Celeb-DF proxy AUC over a mean-pool/spatial-only baseline before merging. | B4 |
| P10 | **Global Reproducibility Seal:** `set_global_seed(42)` enforced across Python, NumPy, PyTorch, CUDA, and DataLoader workers. | E3 |
| P11 | **Platform-Safe Logging:** ASCII-only characters. Emoji banned from all Python files. | E2 |
| P12 | **Checkpoint Schema Validation:** Every checkpoint load asserts that loaded keys exactly match current architecture. `strict=True`, no exceptions. | E4 |

---

# PHASE 4 — NEW SYSTEM DESIGN

## 1. Dataset Protocol

### Filesystem Layout
```
E:\Pojects\workspace\
  datasets\
    ffpp\                   <- 32-frame face crops; >0.95 RetinaFace confidence
    wilddeepfake\           <- Pre-extracted internet face sequences (rglob scan)
    celebdf\                <- 32-frame face crops (to be re-extracted from re-downloaded source)
  forensic-v2\              <- All code
    data\
      splits\
        ffpp_splits.json    <- Generated ONCE with seed=42, committed, never regenerated
```

### FF++ Partition (Subject-Disjoint)

| Partition | % of Subjects | Purpose |
|-----------|--------------|---------|
| TRAIN | 70% | Model training |
| VALIDATION | 10% | Early stopping, LR scheduling |
| CALIBRATION | 10% | Temperature scaling + Isotonic fitting |
| HELD-OUT | 10% | Final in-distribution AUC report |

### WildDeepfake
- Loaded via recursive `.jpg` scan. No re-extraction.
- Used in **Stage 2 training only**. Never in calibration.

### Celeb-DF
- **Zero-shot test set ONLY. Never in any training loop or calibrator.**
- 50 videos (25 real, 25 fake) designated as a "proxy mini-val" for checkpoint selection.
- Remaining 468 videos constitute the final reported zero-shot test set.

---

## 2. Model Architecture

### Backbone: CLIP ViT-B/16

**Why CLIP ViT-B/16?**
CLIP ViT-B/16 (trained on 400M image-text pairs via OpenAI CLIP) produces semantically rich, domain-generalised visual representations that are fundamentally different from ImageNet-pretrained convnets. Its diverse training distribution makes the features intrinsically more robust to domain shift — directly addressing the cross-dataset generalisation failure (B2). Unlike a purely convolutional backbone, the global self-attention in ViT captures long-range facial structure anomalies (identity inconsistency, blending seam geometry) that local convolution kernels miss.

**VRAM solution (6GB RTX 3060):**
Raw ViT-B/16 × 32 frames × batch=8 exceeds 6GB. The solution is:
- Reduce `T=16` frames (sufficient temporal coverage; validated in B4 lesson)
- `batch=4`, `gradient_accumulation=8` → effective batch=32 preserved
- Gradient checkpointing on all frozen ViT blocks (recompute activations on backward)
- Process frames as `[B×T, 3, 224, 224]` through CLIP encoder, then reshape → no T-dimension memory explosion
- AMP (fp16) throughout

```
Input: [B=4, T=16, 3, 224, 224]
  |
  +-- [Spatial Branch — CLIP ViT-B/16]
  |     Reshape: [B*T, 3, 224, 224] -> CLIP visual encoder
  |     Extract: CLS token pre-projection -> [B*T, 768]
  |     Freeze: patch embed + first 8 transformer blocks (grad_ckpt=True)
  |     Unfreeze: last 4 transformer blocks
  |     Reshape back: [B, T, 768]
  |
  +-- [Frequency Branch]
  |     Grayscale -> DCT-II on 56x56 downsampled frame
  |     Log-magnitude spectrum, clipped [-10, 10]
  |     Lightweight CNN (3 conv) -> [B, T, 256]
  |
  [Concat] -> [B, T, 1024]
  |
  [Temporal Module]
    Multi-head Self-Attention (4 heads, d=1024)
    Learnable positional encoding
    Mean-pool over T -> [B, 1024]
  |
  [Classifier Head]
    Linear(1024, 256) -> GELU -> Dropout(0.4)
    Linear(256, 1) -> raw logit (NOT sigmoid)
  |
  [Temperature Module]
    scaled_logit = logit / T  (T: learnable scalar, init=1.5)
  |
  [Calibration]
    Isotonic Regression -> P(fake)
    LR = P(fake) / (1 - P(fake))
```

---

## 3. Training Strategy

### Stage 1 — Base Training (FF++ Only)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Backbone frozen | Patch embed + first 8 ViT blocks (grad_ckpt) | Preserve CLIP semantic features; VRAM safety |
| LR (head + freq + temporal) | 3e-4 | Fast convergence of new modules |
| LR (unfrozen ViT blocks 9–12) | 1e-5 | Fine-tune CLIP features conservatively |
| Optimizer | AdamW, wd=0.01 | Standard; weight decay on ViT is critical |
| Scheduler | CosineAnnealingLR, T_max=20 | Smooth decay |
| Batch size | 4 | VRAM-safe with CLIP ViT-B/16, T=16 |
| Gradient accumulation | 8 | Effective batch = 32 |
| Gradient checkpointing | Enabled on frozen blocks | ~40% VRAM reduction on ViT activations |
| **JPEG augmentation** | **q=20–55 on 80% of frames** | **P3: Destroy compression shortcut** |
| Label smoothing | 0.1 | Prevent overconfidence (P6) |
| Checkpoint criterion | Celeb-DF proxy AUC | P7: Cross-dataset is primary metric |

### Stage 2 — Hybrid Fine-Tuning (FF++ + WildDeepfake)

| Parameter | Value |
|-----------|-------|
| Backbone frozen | Patch embed + first 10 ViT blocks (more conservative) |
| LR | 3e-5 (head/freq/temporal), 5e-6 (unfrozen ViT blocks) |
| JPEG augmentation | q=15–60 on 90% of batches |
| Checkpoint criterion | Celeb-DF proxy AUC |

### Freeze/Unfreeze Invariant
- **Never unfreeze all layers.** Only the final 2–4 ViT transformer blocks and all new modules (frequency branch, temporal module, classifier head) are ever trainable.
- Patch embedding and positional embeddings are permanently frozen.
- Frequency branch trains from scratch in both stages.
- Temperature scalar `T` is frozen during Stages 1 & 2; fitted only during calibration.

---

## 4. Calibration Module

**Step 1 — Temperature Scaling**
- Fit `T` on FF++ CALIBRATION partition logits only
- Minimize NLL on the calibration set
- Assert: `T > 1.0` after fitting

**Step 2 — Isotonic Regression (PAVA)**
- Input: temperature-scaled probabilities from CALIBRATION partition
- Output: `P(fake | score)` — a monotonically mapped probability

**Step 3 — LR Conversion**
```python
LR = P_fake / (1 - P_fake)
```

**Calibration Dataset:** FF++ CALIBRATION partition ONLY (10% of subjects, strictly disjoint from training and validation).

---

## 5. Evaluation Protocol

Every evaluation must produce ALL of the following:

| Metric | Formula | Threshold Dependency |
|--------|---------|---------------------|
| AUC | sklearn.roc_auc_score | None |
| EER | FAR=FRR intersection | EER optimal |
| HTER | (FAR + FRR) / 2 | At EER threshold |
| CLLR | `[E(log2(1+1/LR)|H1) + E(log2(1+LR)|H0)] / 2` | None |
| AP | Average Precision | None |
| Tippett Plot | CDF of log10(LR) for H0 and H1 | Mandatory output |

### Cross-Dataset Evaluation Matrix

| Test Set | AUC | EER | HTER | CLLR |
|----------|-----|-----|------|------|
| FF++ Held-Out | - | - | - | - |
| WildDeepfake Test | - | - | - | - |
| Celeb-DF Zero-Shot | - | - | - | - |

---

# PHASE 5 — FAILURE PREVENTION MECHANISMS

| Past Failure | Preventive Constraint | Runtime Check |
|---|---|---|
| B2: Compression shortcut | CRF aug baked into `BaseDataset.__getitem__` | `assert config.aug.jpeg_p >= 0.8` at startup |
| D1: Raw MP4 evaluation | `BaseDataset.__init__` validates `processed_dir` | `assert Path(processed_dir).is_dir()` |
| C2: Calibration leakage | Splits from frozen JSON file | `DataLeakageAuditor.assert_disjoint(train_ids, cal_ids)` |
| B1: In-dist AUC as primary | Celeb-DF proxy AUC is checkpoint criterion | Only save if `celebdf_proxy_auc` improves |
| E1: Config not enforced | `@dataclass(frozen=True)` config loaded from YAML | No mutable config object at runtime |
| E3: Non-reproducible | `set_global_seed(42)` at every `__main__` | `assert torch.initial_seed() == SEED` |
| E4: Checkpoint mismatch | `load_checkpoint()` with `strict=True` always | Key-set assertion before `load_state_dict` |
| C3: Overconfident scores | Label smoothing 0.1 + Temperature scaling | `assert T > 1.0` after temperature fitting |
| A1: Frame count starvation | `config.n_frames = 32` is single source | `assert len(frames) == config.n_frames` in `__getitem__` |

---

# PHASE 6 — IMPLEMENTATION PLAN

## Folder Structure

```
forensic-v2\
  config.yaml                   <- Single source of truth
  
  src\
    config.py                   <- Loads YAML; validates paths; typed frozen dataclass
    
    datasets\
      base_dataset.py           <- Abstract; enforces processed_dir; JPEG aug built-in
      ffpp_dataset.py           <- FF++ with splits from JSON
      wild_dataset.py           <- WildDeepfake; rglob scan; balanced sampler
      celebdf_dataset.py        <- Test-only; raises error if called in train mode
      extractor.py              <- RetinaFace; 32 frames; confidence > 0.95
      leakage_auditor.py        <- DataLeakageAuditor; run at startup
      
    models\
      backbone.py               <- CLIP ViT-B/16 loader; freeze policy; gradient checkpointing
      frequency_branch.py       <- DCT transform + lightweight CNN
      temporal_module.py        <- Multi-head self-attention over T frames
      forensic_model.py         <- Full assembled model with temperature scalar
      
    training\
      trainer.py                <- Base Trainer; Celeb proxy AUC checkpoint criterion
      stage1_ffpp.py            <- Stage 1 execution script
      stage2_hybrid.py          <- Stage 2 execution script
      
    calibration\
      temperature_scaling.py    <- Fits T on calibration logits
      isotonic_calibrator.py    <- PAVA; outputs LR
      run_calibration.py        <- Main calibration pipeline
      
    evaluation\
      metrics.py                <- AUC, EER, HTER, CLLR, AP
      tippett_plot.py           <- Tippett plot (mandatory)
      run_evaluation.py         <- Cross-dataset evaluation matrix
      
    utils\
      seed.py                   <- set_global_seed(); enforced everywhere
      logging.py                <- ASCII-only; Windows-safe
      device.py                 <- GPU info; VRAM monitoring
      checkpoint.py             <- strict=True; schema validation
      
  data\
    splits\
      ffpp_splits.json          <- Frozen; committed; never regenerated
```

## VRAM Optimization (RTX 3060 — 6GB)

| Technique | VRAM Saved | Notes |
|-----------|------------|-------|
| T=16 frames (vs T=32) | ~1.8 GB | Primary CLIP ViT VRAM lever |
| Batch=4 + Grad Accum=8 | Safe peak budget | Effective batch=32 preserved |
| Gradient checkpointing on frozen ViT blocks | ~1.2 GB | Recompute activations; no gradient storage |
| Frame-flatten strategy (`[B×T, 3, H, W]`) | Avoids T-dim in ViT | Single encoder call per batch |
| DCT on 56×56 (not 224×224) | ~0.3 GB | Frequency branch kept lightweight |
| torch.cuda.amp (fp16) | ~40% activation reduction | Applied globally |
| pin_memory=True, num_workers=4 | Eliminates CPU-GPU bottleneck | —

---

# PHASE 7 — EXPECTED OUTCOMES (REALISTIC)

| Metric | Stage 1 | Stage 2 (in-domain) | Celeb-DF Zero-Shot |
|--------|---------|--------------------|--------------------|
| AUC | 0.72–0.78 | 0.82–0.88 | **0.72–0.80** |
| EER | 0.24–0.30 | 0.14–0.20 | 0.22–0.28 |
| HTER | 0.25–0.32 | 0.15–0.22 | 0.22–0.28 |
| CLLR | — | — | **0.28–0.40** |

> [!NOTE]
> A Celeb-DF AUC above 0.80 should be considered suspicious and investigated for data leakage or proxy set contamination. The published SOTA for comparable single-source-trained models (2021–2023) is 0.72–0.79.

> [!CAUTION]
> CLLR < 0.3 requires >5,000 calibration samples and a well-separated score distribution. With ~1,000 FF++ calibration samples, we realistically target **CLLR < 0.40**, which is still a meaningful improvement over V1's 0.489 and scientifically defensible.

### Expected Failure Cases
1. Occluded faces (>50% of frames) → low confidence, unreliable LR
2. StyleGAN3 deepfakes → out-of-distribution; model may fail
3. Heavily compressed Celeb-DF reals → residual compression confusion possible

---

# PHASE 8 — FINAL VALIDATION CHECKLIST

Before any result is reported:

- [ ] `ffpp_splits.json` exists, generated with `seed=42`, never regenerated
- [ ] `DataLeakageAuditor` ran at training start with zero violations logged
- [ ] Compression augmentation confirmed `jpeg_p >= 0.8` in training logs
- [ ] Celeb-DF proxy AUC used as checkpoint criterion (confirmed in trainer log)
- [ ] Temperature scaling complete; `T > 1.0` confirmed in calibration log
- [ ] Isotonic calibrator fitted on CALIBRATION partition ONLY
- [ ] CLLR computed with correct formula: `[E(log2(1+1/LR)|H1) + E(log2(1+LR)|H0)] / 2`
- [ ] Tippett plots generated for FF++ held-out AND Celeb-DF zero-shot
- [ ] Cross-dataset evaluation matrix fully populated (3 sets × 5 metrics)
- [ ] HTER reported at EER threshold, NOT at 0.5
- [ ] All results reproducible: `set_global_seed(42)` confirmed in log header
- [ ] No emoji present in any `.py` file
- [ ] Checkpoint loaded with `strict=True` and schema assertion passed
- [ ] Frequency branch ablated independently and shown to improve Celeb-DF proxy AUC

---

*Version: V2.1 | Date: 2026-04-05 | Backbone: CLIP ViT-B/16 | Supersedes all V1 architecture and methodology documents.*
