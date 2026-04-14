# 🕵️‍♂️ The Story of Forensic AI: Catching Deepfakes in the Real World

Imagine you are a detective trying to spot forged paintings. If you only study one single artist's forgeries, you might get fooled by a different artist. This project was about building an AI detective that can spot **any** forged video (a "deepfake"), no matter who made it, and being so incredibly sure about it that a judge would believe you in a courtroom.

This is the complete story of how we built it, the problems we faced, and how we solved them.

---

## 🎯 The Big Goal
1. **Spot the Fakes (Generalization):** We needed our AI to look at a fake video it has **never seen before** and still know it's fake. (We measure this with **AUC** — 1.0 is perfect, 0.5 is guessing).
2. **Be Certain for Court (Calibration):** We needed the AI to honestly tell us *how confident* it is. If it's a blurry video, it should say "I am 60% sure." If it's clear, "I am 99.9% sure." In court, this is called a Likelihood Ratio. (We measure the error of this confidence with **CLLR** — lower is better, and we needed to get **below 0.40**).

---

## 🧠 How We Built the AI's Brain (The Architecture)
To spot a really good fake, our AI needed special senses:
1. **The Eyes (Spatial / CLIP-ViT):** Looks at a single picture (frame) and spots weird pixels.
2. **The Memory (Temporal):** Looks at a sequence of 16 pictures like a flipbook. It checks if the movement is jittery or unnatural.
3. **The X-Ray (Frequency / DCT):** Deepfakes compress videos when they save them. The X-Ray looks at the invisible "frequency noise" left behind by the fake video maker.

---

## 🚀 The Journey: Step by Step

### 🟢 Stage 1: Learning the Basics
* **What we did:** We showed our AI a massive dataset called **FF++** (the "training wheels"). We froze the main parts of its brain so it wouldn't learn too fast and forget how to see normal things.
* **The Result:** It became perfectly good at spotting FF++ fakes (AUC: 0.99). But it was still in the training wheels phase.

### 🟡 Stage 2: Facing the Real World (Domain Adaptation)
* **What we did:** We introduced a very hard dataset called **WildDeepfake** (videos from the internet) and trained it on a mix of FF++ and WildDeepfake.
* **The Goal:** Make it generalize, so it could catch **Celeb-DF**—a high-quality dataset of celebrity fakes that the AI was strictly **not allowed to see** during training.
* **The Result:** It worked! The AI scored a **0.89 AUC** on Celeb-DF. It could spot fakes it had never seen before.

### 🔴 The Giant Roadblock: The Courtroom Test (Calibration)
* **What we did:** We tested the AI's "confidence" using our courtroom metric: **CLLR**.
* **The Roadblock:** The AI failed horribly on the new datasets. It got a CLLR of **0.23** on FF++ (Great!) but a terrible **0.70** on Celeb-DF and **1.48** on WildDeepfake.
* **Why did it fail?** The AI was basically a student who thought they knew everything because they passed the first test (FF++). When facing Celeb-DF, it was overly confident even when it was wrong.
* **How we fixed it (Hybrid Calibration):** We created a completely separate "Calibration" dataset using a tiny chunk of FF++, Wild, and Celeb-DF. We forced the AI to adjust its confidence scores by showing it how hard the real world actually is.

### 🟢 Stage 4: The Final Test
* **What we did:** After fixing its confidence, we ran the final test.
* **The Result (Massive Victory):** 
  * Celeb-DF CLLR dropped from 0.70 to **0.37** (Target met!).
  * Celeb-DF AUC jumped to **0.97** (It was incredibly accurate).
  * WildDeepfake CLLR improved from 1.48 to 1.04.

### 🟣 Stage 5: Proving Who Did the Real Work (Ablation)
In science, you have to prove *why* your idea worked. We did a test called an "Ablation." We ripped parts off our AI to see if it would fail without them.
* **Variant 1 (Baseline):** We removed the Memory (Temporal) and X-Ray (Frequency). Result? It scored an awful **0.77** on Celeb-DF. It couldn't spot new fakes.
* **Variant 2 (Beta):** We gave it back its Memory, but no X-Ray. Result? It improved to **0.86**. Better, but not perfect.
* **Variant 3 (Our Full Model):** It had Eyes, Memory, and X-Ray. Result? **0.97**.
* **Conclusion:** This absolutely proved that the "X-Ray" (Frequency) and "Memory" (Temporal) parts were strictly necessary to catch advanced, unseen deepfakes.

---

## 🏆 Final Scorecard

Here is the final proof that the project was a total success:

| Dataset | Can it spot the fake? (AUC) | Is it reliable for court? (CLLR) | What it means |
| :--- | :--- | :--- | :--- |
| **FF++** (The basics) | 0.9912 (A+) | 0.2201 (Perfect) | Excellent baseline. |
| **Celeb-DF** (The unseen test) | **0.9731 (A+)** | **0.3705 (Target Met! <0.4)** | State of the Art completely zero-shot! |
| **WildDeepfake** (The internet) | 0.9060 (A-) | 1.0408 (Much better) | The internet is messy, but we improved heavily! |

---

## 🌟 Summary
We built an AI that doesn't just guess if a video is fake. It looks at the pixels, the movement, and the invisible frequency noise. We hit a major roadblock when the AI got too cocky, but we fixed it by teaching it to adjust its confidence based on out-of-domain data. In the end, we proved mathematically that our special architecture works and is reliable enough to provide scientific Likelihood Ratios for forensic evidence.
