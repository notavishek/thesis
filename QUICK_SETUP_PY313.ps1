# Quick Setup for Python 3.13 + NVIDIA GPU (RTX 4060)
# Run these commands one by one in PowerShell

# Step 1: Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Step 2: Install PyTorch with CUDA 12.6 (supports Python 3.13)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Step 3: Verify CUDA works
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Step 4: Install transformers and dependencies (specific versions for 3.13 compatibility)
pip install transformers==4.46.0 tokenizers sentencepiece
pip install pandas numpy scikit-learn scipy
pip install wandb tqdm jupyter ipykernel

# Step 5: Open notebook in VS Code and select the venv kernel
# Then run all cells
