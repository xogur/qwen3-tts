#!/bin/bash
# 파일 위치: Qwen3-TTS/start_server.sh
# 목적: Qwen3-TTS 서버를 Flash-Attention 최적화 모드로 안전하게 실행합니다.

# 1. Flash Attention 엔진을 강제로 사용하도록 지시하여 응답 속도를 극대화합니다.
export USE_FLASH_ATTN=1

# 2. 그래픽카드(VRAM) 메모리 파편화를 방지하여 서버가 오래 켜져 있어도 속도를 일정하게 유지합니다.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "🚀 Qwen3-TTS 고속 API 서버 구동을 시작합니다..."
echo "=================================================="

# 3. 실제 파이썬 API 서버를 Uvicorn 다중 워커(4개)로 실행하여 동시 접속을 완벽히 분리합니다.
uvicorn api_server:app --host 0.0.0.0 --port 18003 --workers 4