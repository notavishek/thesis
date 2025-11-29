# AI Agent Instructions for Multilingual Hate Detection Project

## Project Overview
This is a **multilingual hate speech detection thesis** using a **multi-task learning architecture** with XLM-RoBERTa. The project unifies 6 diverse hate speech datasets into a single standardized format, then trains a single model to predict three interconnected tasks simultaneously: hate type, target group, and severity.

### Key Challenge: Incomplete Multi-Task Labels
Many datasets have missing annotations (e.g., target_group=-1). The architecture handles this via **masked loss computation** — only tasks with valid labels contribute to loss for each sample. See `multitask_loss()` in Cell 4 of `main.ipynb`.

---

## Architecture Overview

### Data Pipeline (main.py + split_unified_data.py)
1. **Source Datasets** (in `dataset/`):
   - Bengali Hate v1 & v2 (hate_type, limited target_group)
   - BLP25 Subtask 1B (target_group only)
   - ETHOS (severity only)
   - OLID (hate_type, target_group via subtask_c)
   - Toxic Comments (multilingual, severity proxy)

2. **Unified Format** (8 columns):
   ```python
   ['id', 'text', 'language', 'hate_type', 'target_group', 'severity', 'confidence', 'source_dataset']
   ```
   Missing labels are set to `-1` (then masked during training).

3. **Dataset Splits** (stratified by language × is_hate):
   - Train/Val/Test: 60%/20%/20% (see `split_unified_data.py`)
   - Always load from `dataset/UNIFIED_ALL_SPLIT.csv`

### Model Architecture
- **Backbone**: XLM-RoBERTa-Large (1024 hidden dim, multilingual)
- **Multi-head classifier** (`MultiTaskXLMRRoberta` in Cell 3):
  - 3 classification heads sharing backbone
  - hate_type: 6 classes (0=other, 1-5=specific types)
  - target_group: 4 classes (0=other, 1=individual, 2=org/group, 3=community)
  - severity: 4 classes (0-3 severity levels)
  - CLS token pooling strategy

---

## Notebook Structure & Execution Order

| Cell | Purpose | Key Outputs |
|------|---------|-------------|
| 1 | EDA on splits | Data distribution sanity checks |
| 2 | HateDataset loader | Tokenization, masking logic for incomplete labels |
| 3 | MultiTaskXLMRRoberta model | Architecture definition |
| 4 | multitask_loss function | Masked cross-entropy per task |
| 5 | Mini-batch validation | Forward pass + loss computation example |
| 6 | Data loaders (full + smoke) | SEED=1337, MAX_LENGTH=160, BATCH_SIZE=16 |
| 7 | evaluate() helper | Computes loss + macro/micro F1 per task |
| 8 | train_model() function | Main training loop with early stopping + W&B logging |
| 9 | Smoke test subsets | 512/256/256 samples for quick validation |
| 10 | Smoke training | 1 epoch smoke test to verify pipeline |
| 11 | Smoke evaluation | Load checkpoint, evaluate on smoke val/test |
| 12 | W&B setup | Authenticate with project 'multilingual-hate-detection' |
| 13 | Full training | 5 epochs on full dataset, W&B tracking enabled |
| 14 | Final evaluation | Load best checkpoint, report val/test metrics |

**Execution Strategy**:
- Run Cells 1-9 sequentially first (loads data, defines model/loss, prepares dataloaders)
- Run Cell 10 for quick smoke test (~2-5 min on GPU)
- Run Cell 13 for full training (~30-60 min on GPU depending on hardware)
- Run Cell 14 to evaluate final results

---

## Project-Specific Patterns

### 1. Masked Loss with Incomplete Labels
```python
# In multitask_loss() - ONLY compute loss where mask=True
ht_mask = masks['hate_type'].bool()
if ht_mask.any():
    loss_ht = F.cross_entropy(hate_type_logits[ht_mask], targets['hate_type'][ht_mask])
```
**Why**: Many source datasets only annotate one task. Don't force gradients on -1 labels.

### 2. Feature Dictionary Format
Dataset returns a flat dict with keys: `input_ids, attention_mask, hate_type, target_group, severity, hate_type_mask, target_group_mask, severity_mask`
- **Labels as int**: Direct tensor creation, no one-hot encoding
- **Masks as bool**: Used in loss computation to filter invalid samples

### 3. Configuration Dictionary Pattern
```python
training_config = {
    'epochs': 3,
    'learning_rate': 2e-5,
    'weight_decay': 1e-2,
    'warmup_ratio': 0.1,
    'grad_clip': 1.0,
    'patience': 2,
    'dropout': 0.2,
    'task_weights': (1.0, 1.0, 1.0)  # Can balance tasks if needed
}
```
Pass config to `train_model()` to enable easy hyperparameter sweeps.

### 4. Checkpoint Naming Convention
```python
best_ckpt_path = os.path.join(CHECKPOINT_DIR, f'{run_name}_best.pt')
# Examples: 'xlmr_smoke_best.pt', 'xlmr_full_large_best.pt'
```

### 5. W&B Integration
- **Automatic logging**: Every epoch logs train_loss, val_loss, all task-specific F1 scores
- **API key in Cell 12**: Hardcoded for reproducibility (not best practice for production)
- **Project name**: `multilingual-hate-detection`
- **Run names**: Use descriptive names like `xlmr_full_large` or `xlmr_base_subset`

---

## Key Hyperparameters & Constants

```python
SEED = 1337  # Reproducibility across runs
MAX_LENGTH = 160  # Token limit (XLM-R vocab ~250k)
BATCH_SIZE = 16  # Per-device batch size
CHECKPOINT_DIR = 'checkpoints/'

# Training defaults
learning_rate = 2e-5  # Standard for transformer fine-tuning
weight_decay = 1e-2
warmup_ratio = 0.1  # 10% of total steps
grad_clip = 1.0  # Prevent exploding gradients
```

---

## Common Debugging & Extension Points

### Adding a New Dataset
1. Create `map_newdataset(df)` function in `split_unified_data.py`
2. Map columns to UNIFIED_COLUMNS schema (use -1 for missing annotations)
3. Append to concatenation in main pipeline
4. Reload `UNIFIED_ALL_SPLIT.csv` in notebook

### Adjusting Task Weights
If one task dominates loss, modify `task_weights` in training_config:
```python
training_config['task_weights'] = (1.0, 2.0, 0.5)  # Boost target_group, down-weight severity
```

### GPU Memory Issues
- Reduce BATCH_SIZE (default 16 → try 8)
- Reduce MAX_LENGTH (160 → 128)
- Switch model to `xlm-roberta-base` (not `-large`)

### Model Not Converging
- Check data distribution in Cell 1 (class imbalance?)
- Verify masks are computed correctly (Cell 2: `masks = {key: (val != -1) ...}`)
- Inspect sample batch features (Cell 5)

---

## Dependencies & Environment
- **PyTorch**: Via transformers library
- **Transformers**: 0.11.1+ (HuggingFace)
- **Scikit-learn**: For F1 metrics
- **Weights & Biases**: Optional but recommended (Cell 12)
- **CUDA**: Recommended for 5-epoch runs (~30-60 min on V100, 2-3 hr on CPU)

---

## Important Caveats
1. **No cross-validation**: Using fixed train/val/test splits from `split_unified_data.py` (stratified by language + is_hate)
2. **Imbalanced classes**: Check `df['hate_type'].value_counts()` — class 0 dominates in some datasets
3. **Multilingual complexity**: Bengali, English, Banglish mixed. Model must generalize across scripts.
4. **Incomplete annotations intentional**: Don't impute -1 labels; masked loss handles it.
