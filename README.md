

# Patent IPC Classification Pipeline via Qwen 3

This repository contains a professional pipeline for efficient fine-tuning of the **Qwen 3** language model (**Causal LM** architecture) for the task of automatic patent classification according to the **International Patent Classification (IPC)** system.

The project implements **Masked Instruction Fine-Tuning (Target-Only Masking)** using the `trl` library, which eliminates gradient noise caused by prompt tokens and focuses the computation of the loss function (**Cross-Entropy Loss**) exclusively on the assistant's responses.

---

## 📊 Performance Metrics

The table below compares the baseline model and the fine-tuned checkpoint. Evaluation was performed using both a strict **Exact Match** criterion and more flexible multi-class classification metrics.

| Metric                     | Base Qwen 3 | Fine-tuned Qwen 3 | Improvement (Delta) |
| :------------------------- | :---------: | :---------------: | :-----------------: |
| **Exact Match Accuracy**   |    0.000    |     **0.272**     |        +27.2%       |
| **Partial Match Accuracy** |    0.088    |     **0.512**     |        +42.4%       |
| **Average Precision**      |    0.044    |     **0.480**     |        +43.6%       |
| **Average Recall**         |    0.069    |     **0.413**     |        +34.4%       |
| **F1 Score**               |    0.054    |     **0.444**     |        +39.0%       |

---

## 🛠️ Training Architecture and Configuration

The training process was fully optimized for deployment on commercially accessible server hardware.

* **Hardware:** Single **NVIDIA Tesla T4** GPU (16 GB VRAM).
* **Parameter Efficiency:** **LoRA (Low-Rank Adaptation)** was employed with rank $r=16$ and scaling factor $\alpha=32$. Less than 2% of the model parameters were updated.
* **Loss Isolation (Target-Only Masking):** Using `DataCollatorForCompletionOnlyLM`, all prompt tokens preceding the `<|im_start|>assistant\n` tag were masked out, ensuring that the loss was computed solely on assistant-generated tokens.

### Training Logs and Convergence

The model demonstrated stable convergence, with both training and validation losses steadily decreasing throughout the entire training run (149 steps):

| Step            | Training Loss | Validation Loss |
| :-------------- | :-----------: | :-------------: |
| 25              |    0.794368   |     0.514481    |
| 50              |    0.439181   |     0.434189    |
| 75              |    0.433634   |     0.384287    |
| 100             |    0.410555   |     0.386848    |
| 125             |    0.404966   |     0.363983    |
| **149 (Final)** |  **0.377406** |   **0.361086**  |



---

## 📈 Evaluation Visualizations

### 1. Exact vs Partial Match Accuracy

The base model is practically incapable of producing outputs in the required IPC format without prior adaptation. After fine-tuning, the model successfully extracts at least one valid IPC code in more than half of the evaluated cases.

![Accuracy Comparison](image.png)

### 2. Precision and Recall

Fine-tuning substantially reduced hallucinations: instead of generating arbitrary text, the model learned to focus on tokens corresponding to IPC classification codes.

![Precision vs Recall](image-1.png)

### 3. Aggregate Metrics and Performance Trajectory

The summary plots clearly illustrate the substantial improvement achieved across all monitored patent classification metrics.

![F1 Score Comparison](image-2.png)

![Aggregated Performance Trajectory](image-3.png)


##  How to repeat results?

You can find MAIN.ipynb

Open it in Google Collab and run using Tesla T4
(or localy inside VS CODE)
