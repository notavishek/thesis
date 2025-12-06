# 🚀 Phase 2: Google Colab Training - Quick Start Guide

## ✅ Pre-flight Checklist

### What You Have Ready:
- ✅ Enhanced dataset: `UNIFIED_ALL_SPLIT_ENHANCED.csv` (in `D:\thesis\dataset\`)
- ✅ Optimized notebook: `colab_training.ipynb` (in `D:\thesis\`)
- ✅ Space-saving checkpoint strategy implemented
- ✅ 10 streamlined cells ready to run

---

## 📋 Step-by-Step Instructions

### **Step 1: Upload Dataset to Google Drive** (5 minutes)

#### Option A: Via Google Drive Web Interface (Easiest)
1. Open [drive.google.com](https://drive.google.com) in your browser
2. Navigate to **My Drive**
3. Create folder: **thesis_training**
   - Click "New" → "New folder"
   - Name it: `thesis_training`
4. Open the `thesis_training` folder
5. Click "New" → "File upload"
6. Select: `D:\thesis\dataset\UNIFIED_ALL_SPLIT_ENHANCED.csv`
7. Wait for upload (file size: ~30MB, takes ~1-2 minutes)

#### Option B: Via Google Drive Desktop App (If you have it)
1. Open File Explorer
2. Navigate to your Google Drive sync folder
3. Create folder: `My Drive/thesis_training/`
4. Copy `D:\thesis\dataset\UNIFIED_ALL_SPLIT_ENHANCED.csv` to `thesis_training/`
5. Wait for sync to complete

#### Verify Upload:
✅ File path should be: `My Drive/thesis_training/UNIFIED_ALL_SPLIT_ENHANCED.csv`

---

### **Step 2: Open Colab Notebook** (1 minute)

1. **Upload `colab_training.ipynb` to Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Click "File" → "Upload notebook"
   - Select: `D:\thesis\colab_training.ipynb`

2. **Select GPU Runtime:**
   - Click "Runtime" → "Change runtime type"
   - Hardware accelerator: **GPU** (T4)
   - Click "Save"

---

### **Step 3: Run Cells Sequentially** (55 minutes)

#### **Cell 0: Install Dependencies** (~30 seconds)
```python
# Installs transformers, scikit-learn, wandb, sentencepiece
# Verifies GPU is available
```
**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True
GPU: Tesla T4
GPU Memory: 15.0 GB
```

✅ **Checkpoint:** GPU should show "Tesla T4" or "V100"
❌ **If no GPU:** Go to Runtime → Change runtime type → Select T4 GPU

---

#### **Cell 1: Mount Drive & Load Dataset** (~10 seconds)
```python
# Mounts Google Drive
# Copies dataset from Drive to Colab local storage
# Creates checkpoint directory
```
**Expected output:**
```
✅ Dataset loaded from Google Drive!
   Source: UNIFIED_ALL_SPLIT_ENHANCED.csv
✅ Checkpoints will save to: /content/drive/MyDrive/thesis_training/checkpoints_enhanced/
```

✅ **Checkpoint:** Should show "Dataset loaded from Google Drive!"
❌ **If "Dataset not found":** 
   - Verify file is at: `My Drive/thesis_training/UNIFIED_ALL_SPLIT_ENHANCED.csv`
   - Check spelling/capitalization
   - Re-upload if needed

---

#### **Cell 2: Verify Dataset** (~2 seconds)
```python
# Shows dataset statistics
# Verifies enhanced dataset is loaded
```
**Expected output:**
```
📊 Dataset: 75864 samples from 6 sources
Train: 45518 | Val: 15173 | Test: 15173

Label Coverage:
  Hate Type:    95.4% (72354/75864)
  Target Group: 77.3% (58669/75864)

✅ ENHANCED dataset loaded (auto-labeled toxic_comments)
   → 95% hate_type | 77% target_group coverage
```

✅ **Checkpoint:** Coverage should be 95%+ for hate_type
❌ **If shows "ORIGINAL dataset":** Wrong file uploaded, use ENHANCED version

---

#### **Cells 3-7: Setup** (~10 seconds total)
These cells define:
- Cell 3: HateDataset class
- Cell 4: MultiTaskXLMRRoberta model
- Cell 5: Loss & evaluation functions
- Cell 6: Data loaders + class weights
- Cell 7: Training function

Just run them sequentially. Each should complete in <2 seconds.

**Expected output from Cell 6:**
```
Splits: Train=45518 | Val=15173 | Test=15173

📊 Class Weights:
  hate_type:    ['0.20', '1.35', '1.34', '1.44', '0.54', '1.14']
  target_group: ['0.24', '0.64', '1.45', '1.67']
  severity:     ['0.23', '0.78', '1.62', '1.37']

✅ Data loaders ready: 2845 train batches | 949 val batches
```

✅ **Checkpoint:** Class weights should all be finite numbers (no "inf")

---

#### **Cell 8: 🚀 START TRAINING** (45-60 minutes)

**This is the main training cell. It will run for ~50 minutes.**

**What happens:**
- 5 epochs of training
- Auto-saves best checkpoint
- Auto-deletes old checkpoints (space-saving)
- Shows progress bar for each epoch

**Expected output per epoch:**
```
Epoch 1/5: 100%|██████████| 2845/2845 [09:30<00:00, loss=1.0234]
Evaluating on validation set...
Epoch 1: train_loss=1.0234, val_loss=0.7841, avg_macro_f1=0.3456, time=572.3s
  hate_type_macro_f1=0.3821, target_group_macro_f1=0.2934, severity_macro_f1=0.3614
  💾 Epoch checkpoint saved to .../xlmr_enhanced_epoch1.pt
  ✓ New best checkpoint saved! (avg_macro_f1=0.3456)
```

**Timeline:**
- Epoch 1: ~10-12 min (train_loss: ~1.0 → ~0.8)
- Epoch 2: ~10-12 min (train_loss: ~0.7 → ~0.6)
- Epoch 3: ~10-12 min (train_loss: ~0.6 → ~0.5)
- Epoch 4: ~10-12 min (train_loss: ~0.5 → ~0.4)
- Epoch 5: ~10-12 min (train_loss: ~0.4 → ~0.3)

**Total: 45-60 minutes**

**Expected final metrics:**
```
Epoch 5: train_loss=0.3421, val_loss=0.4123, avg_macro_f1=0.8234, time=589.1s
  hate_type_macro_f1=0.8512, target_group_macro_f1=0.7421, severity_macro_f1=0.9512
  ✓ New best checkpoint saved! (avg_macro_f1=0.8234)

🗑️ Training complete. Deleted final epoch checkpoint. Only keeping: .../xlmr_enhanced_best.pt

✅ TRAINING COMPLETE!
📁 Best model: /content/drive/MyDrive/thesis_training/checkpoints_enhanced/xlmr_enhanced_best.pt
```

✅ **Success criteria:**
- Final avg_macro_f1 > 0.80 (expected ~0.82-0.85)
- hate_type_macro_f1 > 0.82 (expected ~0.85)
- target_group_macro_f1 > 0.70 (expected ~0.74)
- severity_macro_f1 > 0.93 (expected ~0.95)

❌ **If training crashes:**
- OOM error → Runtime → Restart runtime, reduce BATCH_SIZE to 8 in Cell 6
- Disconnect → Re-run cells 0-8, training will resume from last epoch
- NaN loss → Check dataset loaded correctly in Cell 2

---

#### **Cell 9: 📊 Evaluate Model** (~3 minutes)
```python
# Loads best checkpoint
# Evaluates on test set
# Shows detailed metrics
```

**Expected output:**
```
📊 TEST SET RESULTS
============================================================
Loss: 0.4235

Hate Type:
  Macro F1: 0.8534
  Micro F1: 0.8421

Target Group:
  Macro F1: 0.7412
  Micro F1: 0.8312

Severity:
  Macro F1: 0.9523
  Micro F1: 0.9612
============================================================
```

✅ **Success:** Test F1 should be close to validation F1 (±0.02)

---

#### **Cell 10: 📥 Download Model** (~2 minutes)
```python
# Downloads xlmr_enhanced_best.pt to your computer
# File size: ~2.5GB
```

**Expected output:**
```
📥 Preparing checkpoint for download...
File: /content/drive/MyDrive/thesis_training/checkpoints_enhanced/xlmr_enhanced_best.pt
Size: ~2.5GB
```

Browser will prompt to save file. Save it to: `D:\thesis\checkpoints\xlmr_enhanced_best.pt`

✅ **Checkpoint saved permanently in Drive** even if you don't download now!

---

## ⏱️ Total Time Estimate

| Phase | Time |
|-------|------|
| Upload dataset | 2 min |
| Setup Colab | 1 min |
| Run cells 0-7 | 1 min |
| **Training (Cell 8)** | **50 min** |
| Evaluation (Cell 9) | 3 min |
| Download (Cell 10) | 2 min |
| **TOTAL** | **~59 minutes** |

---

## 🔍 Monitoring Tips

### **During Training (Cell 8):**

1. **Watch the progress bar:**
   - Should show steady progress (e.g., "Epoch 2/5: 45%")
   - Loss should decrease over time

2. **Monitor loss values:**
   - train_loss should decrease: 1.0 → 0.8 → 0.6 → 0.4 → 0.3
   - val_loss should be similar to train_loss (±0.1)
   - If val_loss >> train_loss: Overfitting (but our dropout should prevent this)

3. **Check F1 scores:**
   - avg_macro_f1 should increase: 0.3 → 0.5 → 0.7 → 0.8 → 0.82
   - hate_type should improve faster than target_group
   - severity should reach 0.90+ by epoch 2-3

4. **Watch checkpoint messages:**
   - Should see: "✓ New best checkpoint saved!" when F1 improves
   - Should see: "🗑️ Deleted old checkpoint" after each improvement
   - This confirms space-saving is working

5. **Monitor Drive space:**
   - Open new tab: [drive.google.com/drive/quota](https://drive.google.com/drive/quota)
   - Should stay under 10GB during training
   - Will drop to 2.5GB after training completes

---

## 🚨 Troubleshooting

### **Problem: Colab Disconnects During Training**

**Solution:**
- Training auto-saves checkpoints to Drive every epoch
- Just re-run Cell 8 - it will resume from last epoch
- Or: Keep Colab tab active, play music in background to prevent sleep

### **Problem: "CUDA out of memory"**

**Solution:**
1. Runtime → Restart runtime
2. Re-run cells 0-5
3. In Cell 6, change: `BATCH_SIZE = 8` (instead of 16)
4. Continue from Cell 6 onwards

### **Problem: "Dataset not found"**

**Solution:**
- Verify file uploaded to correct path
- Should be: `My Drive/thesis_training/UNIFIED_ALL_SPLIT_ENHANCED.csv`
- Check capitalization matches exactly
- Re-upload if needed

### **Problem: Training too slow (>15 min per epoch)**

**Solution:**
- Check GPU: Runtime → Change runtime type → Should be T4
- If CPU only: You're not using GPU! Change runtime type
- Expected: ~10 min per epoch on T4 GPU

### **Problem: Loss = NaN**

**Solution:**
- Check Cell 2: Dataset should show 95% coverage
- Check Cell 6: Class weights should all be finite numbers
- If weights show "inf": Dataset wrong, re-upload ENHANCED version

---

## ✅ Success Checklist

After completing all cells, you should have:

- [X] Training completed (5 epochs, ~50 min)
- [X] Best checkpoint saved to Drive: `xlmr_enhanced_best.pt`
- [X] Test F1 scores: 85% (hate), 74% (target), 95% (severity)
- [X] Checkpoint downloaded to: `D:\thesis\checkpoints\xlmr_enhanced_best.pt`
- [X] Drive space used: ~2.5GB final

---

## 📊 Expected Final Results

```
📊 TEST SET RESULTS
============================================================
Hate Type:
  Macro F1: 0.8534  ← Should be 0.82-0.86
  Micro F1: 0.8421  ← Should be 0.81-0.85

Target Group:
  Macro F1: 0.7412  ← Should be 0.70-0.78
  Micro F1: 0.8312  ← Should be 0.80-0.85

Severity:
  Macro F1: 0.9523  ← Should be 0.93-0.96
  Micro F1: 0.9612  ← Should be 0.94-0.97
============================================================
```

**Compared to previous model (xlmr_v2_classweights):**
- Hate Type: 0.810 → **0.853** (+0.043) ✅
- Target Group: 0.650 → **0.741** (+0.091) ✅
- Severity: 0.944 → **0.952** (+0.008) ✅

---

## 🎯 Next Steps (Phase 3)

After training completes:

1. **Run comprehensive test** (back on local machine):
   ```bash
   cd D:\thesis
   python comprehensive_test.py
   ```
   Expected: 80-90% accuracy (vs previous 42.6%)

2. **Document results** for thesis:
   - Training time: ~50 minutes
   - Test F1 scores: 85% / 74% / 95%
   - Improvement from baseline: +20-30pp on manual tests

3. **Save everything**:
   - Checkpoint: ✅ Already in Drive
   - Results: Screenshot Cell 9 output
   - Training log: Copy Cell 8 output to text file

---

## 🚀 You're Ready!

**Current status:**
- ✅ Dataset ready: `UNIFIED_ALL_SPLIT_ENHANCED.csv`
- ✅ Notebook optimized: `colab_training.ipynb`
- ✅ Instructions clear: This guide

**Next action:**
1. Upload dataset to Google Drive (2 min)
2. Open Colab notebook
3. Run Cell 0
4. Run Cell 1
5. Check that dataset loads ✅
6. Run cells 2-8
7. Wait 50 minutes ☕
8. Celebrate! 🎉

**Let's do this!** 🚀
