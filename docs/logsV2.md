# V2.1 Execution Logs

Store raw terminal outputs, metrics, and experimental variables here during the training run.

---

## Stage 1 (FF++ Base) Metrics Log

| Epoch | Train Loss | Val AUC (FF++) | Val HTER (FF++) | Notes |
| :---: | :---: | :---: | :---: | :--- |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

*Summary / Best Epoch:* 

---

## Stage 2 (Hybrid Domain Adaptation) Metrics Log

| Epoch | Train Loss | Val AUC (FF++) | Val AUC (Wild) | Notes |
| :---: | :---: | :---: | :---: | :--- |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

*Summary / Best Epoch:* 

---

## Stage 3 Calibration Log

*   **Optimal Temperature T:** [Target 0.5 - 5.0]  --> 
*   **Isotonic Shift:** 
*   **Cal Partition CLLR:** [Target < 0.5] -->

---

## Stage 4 Final Generalized Results (V2.1 Main Run)

| Dataset | Split Type | AUC | HTER | CLLR |
| :--- | :--- | :--- | :--- | :--- |
| **FaceForensics++** | In-Distribution | | | |
| **WildDeepfake** | Domain-Adapted | | | |
| **Celeb-DF** | STRICT ZERO-SHOT | | | |

*Interpretation:* 

---

## Stage 5 Ablation Study (Zero-Shot on Celeb-DF)

| Model Variation | Celeb-DF AUC | Celeb-DF CLLR | Notes |
| :--- | :--- | :--- | :--- |
| **Spatial ViT-B/16 Only** | | | Baseline |
| **Spatial + Temporal** | | | Removes local biases |
| **Spatial + Temporal + Freq** | | | The Full System |

*Interpretation:* 

---

## Environment Log & VRAM Telemetry

* **Peak VRAM observed:** 
* **Throughput (Sec/Iter):** 
