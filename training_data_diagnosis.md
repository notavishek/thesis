# Training Data Distribution Analysis - Root Cause of Model Issues

## 🔴 Critical Problem: Severe Label Imbalance

### Summary Statistics (Training Set: 45,518 samples)
- **Hate Type labels**: Only 11,329 samples (24.9%) have valid labels
- **Target Group labels**: Only 5,211 samples (11.4%) have valid labels  
- **Severity labels**: All 45,518 samples (100%) have valid labels

---

## 🔍 Detailed Breakdown by Task

### 1. HATE TYPE Distribution (11,329 valid labels)

| Class ID | Label | Count | Percentage | Status |
|----------|-------|-------|------------|--------|
| 0 | not_hate/other | 5,249 | 46.3% | ✅ Dominant |
| 4 | personal_attack | 3,986 | 35.2% | ✅ Well-represented |
| 5 | geopolitical | 1,060 | 9.4% | ⚠️ Under-represented |
| 2 | religious | 577 | 5.1% | 🔴 Severely under-represented |
| 1 | political | 457 | 4.0% | 🔴 Severely under-represented |
| **3** | **gender** | **0** | **0.0%** | **🔴 MISSING!** |

**❌ CRITICAL: Class 3 (gender) has ZERO training samples!**
- This explains why the model has 0% accuracy on gender hate
- Model has never seen gender-based hate during training
- Gender slurs get misclassified as personal_attack (the next most common class)

---

### 2. TARGET GROUP Distribution (5,211 valid labels)

| Class ID | Label | Count | Percentage | Status |
|----------|-------|-------|------------|--------|
| 1 | individual | 3,012 | 57.8% | ✅ Dominant |
| 2 | organization/group | 1,292 | 24.8% | ✅ Adequate |
| 3 | community | 676 | 13.0% | ⚠️ Under-represented |
| 0 | other/none | 231 | 4.4% | 🔴 Severely under-represented |

**Issues:**
- Class 0 (other/none) only has 231 samples (4.4%)
- Model biased toward predicting "individual" (57.8% of training data)
- Explains 0% accuracy on other/none and 95.1% on individual

---

### 3. SEVERITY Distribution (45,518 valid labels)

| Class ID | Label | Count | Percentage | Status |
|----------|-------|-------|------------|--------|
| 0 | none | 23,377 | 51.4% | ⚠️ Over-dominant |
| 1 | low | 11,254 | 24.7% | ✅ Adequate |
| 2 | medium | 8,656 | 19.0% | ✅ Adequate |
| 3 | high | 2,231 | 4.9% | 🔴 Under-represented |

**Issues:**
- Class 0 (none) dominates at 51.4% - model biased toward low severity
- Class 3 (high) only 4.9% - explains misclassification of "All Muslims are terrorists" as low severity
- Model under-predicts high severity hate

---

## 📚 Source Dataset Contributions

| Dataset | Samples | Percentage | Primary Labels Provided |
|---------|---------|------------|------------------------|
| toxic_comments | 30,043 | 66.0% | ❌ severity only (no hate_type, no target_group) |
| olid | 7,898 | 17.3% | ✅ hate_type=0/4 only, limited target_group |
| bengali_hate_v2 | 3,431 | 7.5% | ✅ hate_type (except gender), severity |
| bengali_hate_v1 | 2,087 | 4.6% | ⚠️ hate_type inconsistent |
| blp25_subtask_1b | 1,477 | 3.2% | ✅ target_group=3 (community) only |
| ethos | 582 | 1.3% | ❌ severity only |

**Critical Insight:**
- 66% of training data (toxic_comments) provides NO hate_type or target_group labels
- Only 4 datasets contribute hate_type labels, and NONE provide gender labels
- This creates massive missing data problem (-1 labels)

---

## 🎯 Root Causes of Poor Performance

### 1. **Gender Hate: 0% Accuracy**
**Cause**: Zero training examples for hate_type=3 (gender)
- Bengali datasets labeled gender hate as "Personal" instead of "Gender"
- Gender-specific slurs (খানকি/khanki, বেশ্যা/beshya) mapped to personal_attack
- Model has no concept of gender-based hate as separate category

**Fix Required**: 
- Remap Bengali gender slurs to hate_type=3
- Add gender hate samples from other sources
- OR: Merge gender into personal_attack (simplify taxonomy)

---

### 2. **Target Group "other/none": 0% Accuracy**
**Cause**: Only 231 training samples (4.4%) vs 3,012 individual (57.8%)
- Model heavily biased toward predicting "individual"
- Neutral statements get forced into "individual" bucket
- 88.6% of training data has no target_group label (-1)

**Fix Required**:
- Increase class weights for class 0 dramatically
- Add more neutral/non-targeted examples
- Consider focal loss for extreme imbalance

---

### 3. **Severity Miscalibration (predicts LOW for HIGH hate)**
**Cause**: Class imbalance (51.4% none, only 4.9% high)
- Model biased toward predicting low/none severity
- Extreme hate like "kill all Muslims" → predicted as low
- Class weights insufficient to overcome 10:1 ratio

**Fix Required**:
- Increase class weights for severity=3 (high) by 5-10x
- Add focal loss with gamma=2 to focus on hard examples
- Collect more high-severity examples

---

### 4. **Bengali Over-prediction of Hate**
**Cause**: Training data language imbalance + label noise
- Bengali samples: 17,036 (37.4%)
- But Bengali samples from toxic_comments have noisy labels
- Many neutral Bengali texts mislabeled as severity=1 or 2
- Model learns "Bengali text → probably hate"

**Examples of noise from CSV:**
```
"আমি ও রানু দিদি কে সাপোর্ট করতাম..." → labeled severity=1 (low hate)
"আমি এখন ইনশাআল্লাহ সৌদি আরব আছি..." → labeled severity=0 but text seems neutral
```

**Fix Required**:
- Manual review of Bengali samples
- Filter toxic_comments Bengali samples
- Add more neutral Bengali examples

---

## 📊 Class Weight Analysis

Current class weights from training (Cell 6):
```python
hate_type:    ['2.59', '23.66', '18.73', 'inf', '2.71', '10.20']  # Class 3 = inf (no data!)
target_group: ['22.03', '1.69', '3.95', '7.61']                    # Class 0 = 22x weight
severity:     ['0.98', '2.03', '2.64', '10.24']                    # Class 3 = 10x weight
```

**Issues with current weights:**
1. Gender (class 3) has infinite weight because 0 training samples
2. Target_group class 0 weight (22x) still insufficient for 4.4% data
3. Severity high (10x) not enough to overcome 51% none-class dominance

---

## 🔧 Recommended Fixes (Priority Order)

### **Fix 1: Address Gender Hate (CRITICAL)**
```python
# Option A: Remap existing personal attacks to gender
# In main.py, update map_bengali_hate_v2():
if any(word in text for word in ['খানকি', 'khanki', 'বেশ্যা', 'beshya', 'মাগি', 'magi']):
    hate_type = 3  # Gender instead of 4 (personal)

# Option B: Merge gender into personal_attack
# Simplify taxonomy to 5 classes instead of 6
```

### **Fix 2: Add Neutral Examples for Target Group 0**
```python
# Collect 2000+ neutral statements with target_group=0
# Sources: news headlines, weather reports, greetings
# Ensures model learns what "no target" means
```

### **Fix 3: Increase Severity Class Weights**
```python
# In Cell 6, manually set severity weights:
sv_weights = torch.tensor([0.5, 1.0, 2.0, 15.0])  # 15x for high severity
```

### **Fix 4: Use Focal Loss Instead of Cross-Entropy**
```python
# Replace F.cross_entropy in multitask_loss with:
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()
```

### **Fix 5: Filter Noisy Toxic Comments Data**
```python
# In split_unified_data.py:
# Remove toxic_comments samples with language='bangla' 
# (They have noisy severity labels, no hate_type/target_group)
```

---

## 📈 Expected Impact of Fixes

| Fix | Gender Acc | Target_0 Acc | Severity Acc | Overall |
|-----|-----------|--------------|--------------|---------|
| Current | 0% | 0% | 42.6% | 42.6% |
| +Fix 1 (Gender remap) | 60-70% | 0% | 42.6% | 50% |
| +Fix 2 (Neutral data) | 60-70% | 40-50% | 42.6% | 55% |
| +Fix 3 (Severity weights) | 60-70% | 40-50% | 60-70% | 65% |
| +Fix 4 (Focal loss) | 70-80% | 50-60% | 70-75% | 70% |
| +Fix 5 (Filter noise) | 70-80% | 60-70% | 75-80% | 75% |

---

## 🎓 Thesis Documentation Notes

**For your thesis methodology section:**

1. **Acknowledge the label imbalance problem** - this is common in hate speech research
2. **Document missing gender labels** - explain Bengali datasets lack gender-specific annotations
3. **Discuss class weights + focal loss** as mitigation strategies
4. **Report both validation metrics (81% F1) AND manual test metrics (42.6%)** 
   - Validation uses same distribution as training
   - Manual test reveals generalization issues
5. **Recommend future work**: Collect balanced dataset with complete annotations

**This is actually good research** - identifying these issues demonstrates critical thinking!

---

## ⚡ Quick Action: Test with xlmr_full_large_best.pt

You have another checkpoint: `xlmr_full_large_best.pt` (from earlier training without class weights)

Try:
```python
checkpoint_path = 'checkpoints/xlmr_full_large_best.pt'
```

This might perform differently on manual tests vs the class-weighted version.
