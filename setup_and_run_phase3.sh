#!/bin/bash

echo "🚀 [Phase 3] Qwen3 TTS Optimization (Flash Attention) Setup"
echo "================================================================"

# 1. Install Build Tools
echo "📦 [Build Tools] Checking ninja and packaging..."
pip install ninja packaging
if [ $? -ne 0 ]; then
    echo "❌ [Error] Failed to install build tools."
    exit 1
fi

# 2. Install Flash Attention
if python -c "import flash_attn" &> /dev/null; then
    echo "✅ [Check] Flash Attention is already installed."
else
    echo "⚙️ [Install] Installing Flash Attention (this may take time)..."
    pip install flash-attn --no-build-isolation
    
    if [ $? -ne 0 ]; then
        echo "❌ [Error] Flash Attention installation failed."
        exit 1
    fi
    echo "✅ [Install] Done."
fi

# 3. Start Server
echo "🚀 [Start] Starting api_server.py with Flash Attention..."
export USE_FLASH_ATTN=1
python api_server.py
