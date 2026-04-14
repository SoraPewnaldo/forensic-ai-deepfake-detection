# V2.1 Pipeline Execution Tracker
HF token : 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

Use this document to track physical execution of the V2.1 Forensic AI pipeline.

## 🔴 Stage 0 — Sanity Check
- [x] FF++ splits loaded from JSON correctly
- [x] No subject overlap (leakage visually confirmed zero)
- [x] Celeb-DF isolated and verified absent from training
- [x] T=16 sequence length enforced
- [x] JPEG augmentation active
- [x] Frequency Spatial Balance + Downsample implemented

## 🔵 Stage 1 — Base Training (FF++)
- **Command:** `python -m src.training.stage1_ffpp`
- [x] Training Launched (Benchmarking Benchmarks)
- [x] VRAM Stable (< 6GB)
- [x] Val AUC reached >0.75
- [x] Checkpoint `best_Stage1.pt` saved

## 🔵 Stage 2 — Hybrid Domain Adaptation
- **Command:** `python -m src.training.stage2_hybrid`
- [x] Hybrid DataLoader (50/50 sampler) verified running
- [x] Training Launched (loads Stage 1 weights)
- [x] Top 4 CLIP blocks unfrozen
- [x] WildDeepfake AUC increasing
- [x] Checkpoint `best_Stage2_Hybrid.pt` saved

## 🔴 Stage 3 — Calibration
- **Command:** `python -m src.calibration.run_calibration`
- [x] Execution launched on purely disjoint CAL partition
- [x] Temperature solved (T between 0.5 and 10.0)
- [x] Isotonic Regression mapped
- [x] Likelihood Ratios generated

## 🟢 Stage 4 — Final Evaluation
- **Command:** `python -m src.evaluation.run_evaluation`
- [x] FF++ Evaluated
- [x] WildDeepfake Evaluated
- [x] Celeb-DF Evaluated (Zero-Shot)
- [x] Target Reached: Celeb-DF CLLR < 0.40 (Actual: 0.37)

## 🟣 Stage 5 — Scientific Ablation Training
- [x] Baseline Model trained (Spatial Only)
- [x] Model Beta trained (Spatial + Temporal)
- [x] Model Gamma trained (Full Spatial + Frequency + Temporal)
- [x] Evaluation tables generated for methodology
