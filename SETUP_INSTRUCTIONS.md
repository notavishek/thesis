# 🚀 Setup Instructions for NVIDIA GPU (RTX 4060)

## Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- NVIDIA GPU with CUDA support (RTX 4060 ✅)
- NVIDIA drivers installed (check with `nvidia-smi` in terminal)

---

## Step 1: Extract and Open Folder
Extract the zip and open a PowerShell terminal in that folder.

---

## Step 2: Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

---

## Step 3: Install PyTorch with CUDA 12.6
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## Step 4: Verify CUDA Works
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
Should output: `CUDA: True | GPU: NVIDIA GeForce RTX 4060`

---

## Step 5: Install Other Dependencies
```powershell
pip install transformers==4.46.0 tokenizers sentencepiece
pip install pandas numpy scikit-learn scipy
pip install wandb tqdm jupyter ipykernel
```

---

## Step 6: Verify Everything Works
```powershell
python -c "from transformers import XLMRobertaTokenizer; print('Transformers OK')"
```

---

## Step 7: Run the Notebook

1. Open VS Code
2. Open `main.ipynb`
3. Select the `venv` kernel
4. Run cells 1-9 sequentially (setup)
5. Skip cell 10-11 (smoke test) if you want to go straight to full training
6. Run cell 12 (W&B login - optional, can set `use_wandb=False`)
7. Run cell 13 (Full training) - **~30-45 min on RTX 4060**
8. Run cell 14 (Final evaluation)

---

## ⚡ Expected Performance (RTX 4060)

| Stage | Time |
|-------|------|
| Setup cells (1-9) | ~2-3 min |
| Full training (5 epochs) | ~30-45 min |
| Final evaluation | ~5 min |
| **Total** | **~40-55 min** |

---

## 🔧 Troubleshooting

### CUDA not available
```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install NVIDIA drivers from:
# https://www.nvidia.com/download/index.aspx
```

### Out of Memory (OOM)
Edit cell 6 in the notebook to reduce batch size:
```python
BATCH_SIZE = 8  # Reduce from 16 if OOM errors occur
```

### W&B Login Issues
In cell 13, change `use_wandb=True` to `use_wandb=False` to skip W&B logging.

---

## 📁 Required Files
Make sure these files are present:
- `main.ipynb` - Main training notebook
- `dataset/UNIFIED_ALL_SPLIT.csv` - Training data (~75k samples)
- `requirements_gpu.txt` - Python dependencies for GPU
- `checkpoints/` folder (will be created automatically)
