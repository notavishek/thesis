# Migration Plan: XLM-RoBERTa Large → Base

## Executive Summary
Switching from `xlm-roberta-large` (550M params) to `xlm-roberta-base` (270M params) due to dataset size constraints (54k training samples vs 200k recommended for Large).

---

## Step-by-Step Implementation Plan

### STEP 1: Update Model Architecture
**File:** `colab_training_final.ipynb` (Cell 3)

**Changes:**
```python
# OLD
MODEL_NAME = 'xlm-roberta-large'

# NEW
MODEL_NAME = 'xlm-roberta-base'
```

**Why:** Base model is perfectly sized for 54k training samples (109% of recommended 50k).

---

### STEP 2: Adjust Hyperparameters
**File:** `colab_training_final.ipynb` (Cell 7)

**Changes:**
```python
# OLD
DROPOUT = 0.4  # Heavy regularization for Large

# NEW
DROPOUT = 0.2  # Standard regularization for Base

# Keep these unchanged:
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 5
BATCH_SIZE = 16  # Can increase to 32 if memory allows
MAX_LENGTH = 160
```

**Why:** Base model has lower capacity, so less aggressive regularization is needed.

---

### STEP 3: Update Dataset Path
**File:** `colab_training_final.ipynb` (Cell 1 or wherever CSV is loaded)

**Changes:**
```python
# OLD
DATA_FILE = 'UNIFIED_BALANCED_GENERATED.csv'

# NEW
DATA_FILE = 'Final_Dataset.csv'
```

**Why:** Using the cleaned, deduplicated, conflict-resolved dataset.

---

### STEP 4: Update Loss Weights (If Needed)
**File:** `colab_training_final.ipynb` (Cell 4)

**Current Setup:**
```python
ht_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Hate Type
tg_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])            # Target
sv_weights = torch.tensor([1.0, 3.0, 3.0, 3.0])            # Severity (Equalized)

TASK_WEIGHTS = (1.0, 1.0, 1.5)  # Boosted Severity
```

**Action:** KEEP AS IS. These weights were designed to fix the severity bias and are still valid.

---

### STEP 5: Update Checkpoint Naming
**File:** `colab_training_final.ipynb` (Cell 8)

**Changes:**
```python
# OLD
RUN_NAME = 'xlmr_smart'
CHECKPOINT_PATH = 'checkpoints/xlmr_smart_best.pt'

# NEW
RUN_NAME = 'xlmr_base_final'
CHECKPOINT_PATH = 'checkpoints/xlmr_base_final_best.pt'
```

**Why:** Clear naming prevents confusion with old Large model checkpoints.

---

### STEP 6: Update Custom Prediction Script
**File:** `custom_predict.py`

**Changes:**
```python
# Update model name
MODEL_NAME = 'xlm-roberta-base'

# Update checkpoint path
CHECKPOINT_PATH = 'checkpoints/xlmr_base_final_best.pt'
```

---

### STEP 7: Expected Performance Changes

**Training Time (T4 GPU):**
- Large: 45-60 min/epoch → **Base: 20-30 min/epoch** (2x faster)
- Total (5 epochs): 4-5 hours → **2-2.5 hours**

**Memory Usage:**
- Large: ~12-14 GB VRAM → **Base: ~6-8 GB VRAM**
- Can now use Batch Size 32 comfortably on T4

**Model Performance (Expected):**
- Large (with 54k samples): 76-78% F1 (with overfitting risk)
- **Base (with 54k samples): 75-80% F1 (better generalization)**

---

## Files to Keep vs Delete

### ✅ KEEP (Essential for Training)
- `colab_training_final.ipynb` (Main training notebook)
- `custom_predict.py` (Inference script)
- `dataset/Final_Dataset.csv` (Cleaned dataset)
- `requirements.txt` (Dependencies)
- `main.ipynb` (Local training/testing)
- `split_unified_data.py` (Dataset pipeline reference)
- `.gitignore`, `.github/` (Version control)
- `venv/` (Virtual environment)
- `checkpoints/` (Model saves)

### ❌ DELETE (Obsolete)
- All old analysis scripts (`analyze_*.py`, `check_*.py`, etc.)
- Old markdown reports (`ALL_FIXES_COMPLETE.md`, `DATASET_COMPARISON.md`, etc.)
- Old test results (`comprehensive_test_results.csv`, `comprehensive_test_summary.txt`)
- Redundant notebooks (`colab_training.ipynb` - keep only `_final` version)
- Old augmentation scripts (`augment_dataset.py`, `generate_*.py`)
- Deployment folder (if not actively using)
- `thesis_for_gpu.zip` (outdated archive)
- `wandb/` (old W&B logs - will regenerate)

---

## Pre-Training Checklist

- [ ] Backup old `checkpoints/xlmr_smart_best.pt` (rename to `_large_archive.pt`)
- [ ] Upload `Final_Dataset.csv` to Colab `/content/`
- [ ] Update all 6 changes in `colab_training_final.ipynb`
- [ ] Verify W&B API key is set in Cell 12
- [ ] Test with 1 epoch first to validate pipeline
- [ ] Run full 5-epoch training
- [ ] Compare Base vs Large performance (expect Base to be within 1-2% F1)

---

## Rollback Plan (If Base Underperforms)

If Base model achieves <70% F1:
1. Revert to `xlm-roberta-large`
2. Increase `DROPOUT = 0.5`
3. Add `LABEL_SMOOTHING = 0.1`
4. Reduce `LEARNING_RATE = 1e-5`
5. Increase `WEIGHT_DECAY = 0.05`

**Confidence:** 90% that Base will outperform Large on this dataset due to better size match.
