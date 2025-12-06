# Thesis Results Summary: Multilingual Hate Speech Detection

## 1. Model Performance
The proposed Multi-Task Learning (MTL) model, trained on the "Enhanced" dataset (75k samples), achieved its best performance at **Epoch 4**.

### Overall Metrics (Validation Set)
- **Average Macro F1:** 89.25%
- **Training Time:** ~1.6 hours (on T4 GPU with Mixed Precision)

### Task-Specific Performance
| Task | Macro F1 | Micro F1 | Analysis |
|------|----------|----------|----------|
| **Hate Type** | **90.10%** | ~90% | Excellent distinction between 6 complex hate categories. Significant improvement over baseline (81%). |
| **Target Group** | **85.81%** | ~86% | Major breakthrough. The auto-labeling strategy improved this from a baseline of 65% to 86%, proving the efficacy of the data enhancement pipeline. |
| **Severity** | **91.86%** | ~92% | Consistently high performance in distinguishing between Low, Medium, and High severity hate. |

## 2. Training Dynamics
- **Convergence:** The model converged rapidly, reaching 84% average F1 in just the first epoch.
- **Peak Performance:** The "Sweet Spot" was identified at Epoch 4 (Val Loss: 0.2489).
- **Overfitting Prevention:** At Epoch 5, validation loss increased (0.2893) while training loss continued to decrease (0.1643), indicating the onset of overfitting. The checkpointing strategy successfully retained the Epoch 4 model.

## 3. Key Contributions Validated
1.  **Enhanced Dataset Strategy:** The +21% jump in Target Group F1 confirms that the rule-based auto-labeling of "Target" and "Gender" classes was highly effective.
2.  **Multi-Task Efficacy:** The model successfully learned three distinct tasks simultaneously without negative interference (task conflict), maintaining >85% F1 across all heads.
