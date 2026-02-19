import os
import sys
import time
import io
import asyncio
import logging
from contextlib import asynccontextmanager

# ------------------------------------------------------------------
# [Optimization] CUDA 메모리 및 연산 최적화 플래그 (서버 시작 전 설정)
# ------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# cuDNN 벤치마크: 최적 알고리즘 자동 선택
torch.backends.cudnn.benchmark = True
# TF32 연산 활성화 (RTX 30xx 이상) - FP32 대비 최대 2배 빠른 matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from qwen_tts import Qwen3TTSModel

# ------------------------------------------------------------------
# [Lifespan] 서버 시작과 종료 시 실행될 로직 (최신 FastAPI 표준)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Flash Attention 설정
    attn_impl = "sdpa"
    use_flash_attn = os.getenv("USE_FLASH_ATTN", "1") == "1"

    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("✨ [Optimization] Flash-Attention 2 엔진 활성화")
        except ImportError as e:
            print(f"⚠️ [Optimization] Flash-Attention 로드 실패: {e}")
            print("ℹ️ [Optimization] PyTorch Native SDPA 모드로 전환합니다.")
            attn_impl = "sdpa"
    else:
        print("🛡️ [Compatibility] PyTorch Native SDPA 모드를 사용합니다.")

    # GPU 정보 로깅
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_cap = torch.cuda.get_device_capability(0)
        print(f"🖥️ [System] GPU: {gpu_name} (Capability: {gpu_cap})")

        if torch.cuda.is_bf16_supported():
            print("✅ [System] bfloat16 가속 지원됨")
            dtype_policy = torch.bfloat16
        else:
            print("⚠️ [System] bfloat16 미지원 -> float16으로 전환")
            dtype_policy = torch.float16
    else:
        print("❌ [System] CUDA 없음. CPU 모드.")
        dtype_policy = torch.float32

    try:
        # 모델 로드
        # [Optimization] 0.6B 경량 모델 사용 (1.7B 대비 2~3배 빠른 추론)
        model_name = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        print(f"📦 [System] 모델 로딩: {model_name}")
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype_policy,
            attn_implementation=attn_impl,
            trust_remote_code=True
        )

        # ⚠️ torch.compile은 TTS 모델에서 성능 저하를 유발합니다.
        # TTS는 입력 길이가 매번 달라지는데, reduce-overhead 모드의 CUDA Graphs는
        # 동일한 입력 Shape에서만 이득이 있고, Shape이 바뀌면 매번 재컴파일됩니다.
        # 테스트 결과: "안녕!" 317ms -> 4525ms (14배 느려짐)
        # 따라서 torch.compile을 사용하지 않습니다.
        print("ℹ️ [Optimization] torch.compile 비활성화 (가변 입력 길이 TTS에 부적합)")

        print("✅ [System] 모델 로드 완료. Warmup을 시작합니다...")

        # Warmup: GPU 초기화 및 커널 캐싱
        try:
            print("🔥 [System] Warmup 2회 실행 중...")
            warmup_texts = ["안녕하세요.", "오늘 날씨가 좋네요."]
            for i, text in enumerate(warmup_texts):
                print(f"  - Warmup {i+1}/{len(warmup_texts)}...")
                with torch.inference_mode():
                    model.generate_custom_voice(
                        text=text,
                        language="Korean",
                        speaker="Sohee",
                        non_streaming_mode=False,  # 스트리밍 모드 (prefill 오버헤드 감소)
                        max_new_tokens=1024,
                        temperature=0.7,
                        top_k=30,
                        repetition_penalty=1.1
                    )
            print("✅ [System] Warmup 완료. 서비스 준비 됨.")
        except Exception as e:
             print(f"⚠️ [System] Warmup 중 에러 발생 (무시하고 진행): {e}")

    except Exception as e:
        print(f"❌ [Error] 모델 로드 실패: {e}")
        raise RuntimeError(f"모델 로드 실패: {e}")

    yield
    print("🛑 [System] 서버를 종료합니다.")

# FastAPI 앱 인스턴스 생성
app = FastAPI(title="Qwen3-TTS API Service", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    speaker: str = "Sohee"
    language: str = "Korean"

# 글로벌 모델 변수 초기화
model = None

# ------------------------------------------------------------------
# [API 엔드포인트] 음성 생성
# ------------------------------------------------------------------
@app.post("/tts/generate")
async def generate_speech(request: TTSRequest):
    if not model:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    print(f"🗣️ [Request] 텍스트 처리 중: {request.text[:30]}...")

    try:
        start_time = time.time()

        # [Optimization] inference_mode: no_grad보다 더 적극적인 최적화
        # CUDA Stream은 단일 순차 추론에서는 오버헤드만 추가하므로 제거
        with torch.inference_mode():
            wavs, sr = model.generate_custom_voice(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                non_streaming_mode=False,  # [Optimization] 스트리밍 모드 (prefill 오버헤드 감소)
                max_new_tokens=1024,
                temperature=0.7,
                top_k=30,
                repetition_penalty=1.1
            )

        if wavs is not None and len(wavs) > 0:
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            buffer.seek(0)

            latency = (time.time() - start_time) * 1000
            print(f"⚡ [Process] 성공 ({latency:.1f}ms)")

            return StreamingResponse(buffer, media_type="audio/wav")
        else:
            raise HTTPException(status_code=500, detail="음성 데이터 생성 실패")

    except Exception as e:
        print(f"❌ [Error] 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18003)
