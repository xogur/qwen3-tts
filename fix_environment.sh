#!/bin/bash
echo "🚀 [Fix] Starting Environment Fix for Flash Attention..."

# 1. Uninstall incompatible Nightly/Preview Torch
echo "📦 [1/3] Uninstalling current Torch (likely incompatible version)..."
pip uninstall -y torch torchvision torchaudio flash-attn

# 2. Install Stable Torch (2.5.1) compatible with Flash Attention Wheels
echo "📦 [2/3] Installing Stable Torch 2.5.1 (CUDA 12.4)..."
# Using standard PyPI which works for most Linux CUDA 12 setups, 
# or explicitly pointing to cu124 to be safe.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 3. verify
echo "🔍 [Check] Verifying Torch Version..."
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 4. Install Build Tools (Ninja) for Flash Attention Source Build
echo "🔧 [3.5/5] Installing Build Tools..."
pip install ninja packaging

# 5. Run Flash Attention Installer
echo "⚡ [4/5] Installing Flash Attention..."
python install_flash_attn.py

# 6. Upgrade Transformers & Accelerate (Fix for RTX 5090 CUDA Kernel Error)
echo "🚀 [5/5] Upgrading Transformers & Accelerate for RTX 5090 support..."
# Force uninstall to clear old metadata
pip uninstall -y qwen-tts
# Install strict requirements first
pip install --upgrade transformers accelerate optimum
# Reinstall local package in editable mode with relaxed dependencies
pip install -e .

echo "✅ [Done] Environment fixed! Please restart your API server."
