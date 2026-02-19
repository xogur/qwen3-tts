import os
import sys
import time
import io
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import torch
import uvicorn
from qwen_tts import Qwen3TTSModel

# ------------------------------------------------------------------
# [Lifespan] 서버 시작과 종료 시 실행될 로직 (최신 FastAPI 표준)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 실행
    global model
    # RTX 5090 등 최신 카드 호환성 문제로 Flash Attention 기본 비활성화
    # 'no kernel image is available' 에러는 주로 컴파일된 커널(Flash Attn)과 GPU 아키텍처 불일치에서 발생
    attn_impl = "eager"
    use_flash_attn = os.getenv("USE_FLASH_ATTN", "0") == "1"

    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("✨ [Optimization] Flash-Attention 2 엔진 활성화 (강제 설정)")
        except ImportError:
            print("⚠️ [Optimization] Flash-Attention 라이브러리가 없습니다. Eager 모드로 동작합니다.")
    else:
        print("🛡️ [Compatibility] 안정성을 위해 Default Attention (Eager) 모드를 사용합니다.")

    # GPU 정보 로깅
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_cap = torch.cuda.get_device_capability(0)
        print(f"🖥️ [System] GPU Detected: {gpu_name} (Capability: {gpu_cap})")
        
        # bfloat16 지원 확인
        if torch.cuda.is_bf16_supported():
            print("✅ [System] bfloat16 가속 지원됨")
            dtype_policy = torch.bfloat16
        else:
            print("⚠️ [System] bfloat16 미지원 -> float16으로 전환")
            dtype_policy = torch.float16
    else:
        print("❌ [System] CUDA를 찾을 수 없습니다. CPU 모드로 전환합니다.")
        dtype_policy = torch.float32

    try:
        # 모델 로드
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="auto", # auto로 변경하여 최적 배치
            torch_dtype=dtype_policy,
            attn_implementation=attn_impl,
            trust_remote_code=True
        )
        if hasattr(model, "generate_custom_voice"):
             print("✨ [Optimization] pyTorch 2.0 Compile 적용 중... (시간이 조금 걸릴 수 있습니다)")
             try:
                 # 모델의 핵심 연산 부분을 컴파일 (Fullgraph는 오류 가능성이 높아 부분 컴파일)
                 # mode='reduce-overhead'는 CUDA 그래프를 적극 사용하여 런타임 오버헤드를 줄임
                 if hasattr(model, "model"):
                     model.model = torch.compile(model.model, mode="reduce-overhead")
                     print("✅ [Optimization] torch.compile 적용 완료 (Inner-Model)")
                 else:
                     print("⚠️ [Optimization] 컴파일 대상(model.model)을 찾을 수 없어 건너뜁니다.")
             except Exception as e:
                 print(f"⚠️ [Optimization] 컴파일 실패, 일반 모드로 진행합니다: {e}")

        print("✅ [System] 모델 로드 완료. Warmup을 시작합니다...")
        
        # Warmup: 컴파일 오버헤드를 미리 해소하기 위해 더미 데이터로 1회 실행
        try:
            print("🔥 [System] Warmup 1회 실행 중...")
            with torch.no_grad():
                model.generate_custom_voice(
                    text="안녕하세요.",
                    language="Korean",
                    speaker="Sohee"
                )
            print("✅ [System] Warmup 완료. 서비스 준비 됨.")
        except Exception as e:
             print(f"⚠️ [System] Warmup 중 에러 발생 (무시하고 진행): {e}")

    except Exception as e:
        print(f"❌ [Error] 모델 로드 실패: {e}")
        # 실무에서는 여기서 프로세스를 종료하거나 알람을 보냅니다.
        raise RuntimeError(f"모델 로드 실패: {e}")
        
    yield
    # 서버 종료 시 실행 (리소스 정리 등)
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

    print(f"🗣️ [Request] 텍스트 처리 중: {request.text[:20]}...")
    
    try:
        start_time = time.time()
        
        with torch.no_grad():
            wavs, sr = model.generate_custom_voice(
                text=request.text,
                language=request.language,
                speaker=request.speaker
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
