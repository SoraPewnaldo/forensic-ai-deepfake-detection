import matplotlib.pyplot as plt
import numpy as np
import json
import torch

# 1. Load Real Data
with open('e:/Pojects/workspace/calibration_artefacts/isotonic.json', 'r') as f:
    iso_data = json.load(f)

# The thresholds in the json are probabilities from T-scaling
# We need to map them back to logits for the X-axis
T = 1.3315

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

# X-axis: Raw Logits
logits = np.linspace(-8, 8, 1000)

# 2. Plotting
plt.figure(figsize=(8, 6), dpi=300)

# A. Standard Sigmoid (Uncalibrated)
plt.plot(logits, sigmoid(logits), color='gray', linestyle='--', alpha=0.5, label='Uncalibrated Model Output (Sigmoid)')

# B. Temperature Scaling Curve (T=1.3315)
plt.plot(logits, sigmoid(logits / T), color='red', linewidth=2, label=f'Temperature Scaling (T={T})')

# C. Real Isotonic Mapping (PAVA)
# The json X_thresholds are probabilities. We convert them to scaled logits or just plot them as P_in -> P_out
# The paper plot traditionally shows Logit -> Adjusted P
# We compute P_iso for each logit: 
# 1. logit -> sigmoid(logit/T) -> lookup in iso thresholds
x_iso_probs = np.array(iso_data['X_thresholds'])
y_iso_probs = np.array(iso_data['y_thresholds'])

def apply_isotonic(p_in):
    # Manual step function lookup (PAVA behavior)
    idx = np.searchsorted(x_iso_probs, p_in) - 1
    idx = np.clip(idx, 0, len(y_iso_probs) - 1)
    return y_iso_probs[idx]

logits_eval = np.linspace(-8, 8, 2000)
p_temp = sigmoid(logits_eval / T)
p_final = [apply_isotonic(p) for p in p_temp]

plt.step(logits_eval, p_final, color='green', linewidth=1.5, where='post', label='Isotonic Regression (PAVA)')

# 3. Aesthetics
plt.xlabel('Raw Model Logit', fontsize=12)
plt.ylabel('Calibrated Probability P(fake)', fontsize=12)
plt.title('Forensic AI V2.1: Calibration Mapping Function', fontsize=14, pad=15)

# Fix 2: Add hybrid set note
plt.text(-7.5, 0.95, "Fitted on hybrid calibration set:\n(FF++ + Celeb-DF + WildDeepfake)", 
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.legend(loc='lower right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(-8, 8)
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('e:/Pojects/workspace/Latex/calibration_mapping.png')
print("Successfully saved real calibration plot to Latex/calibration_mapping.png")
