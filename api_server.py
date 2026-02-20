import os
import sys
import time
import io
import asyncio
import logging
from contextlib import asynccontextmanager
from queue import Queue, Empty
from threading import Thread, Event
import numpy as np

# ------------------------------------------------------------------
# [Optimization] CUDA 메모리 및 연산 최적화 플래그 (서버 시작 전 설정)
# ------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# cuDNN 벤치마크: 최적 알고리즘 자동 선택
torch.backends.cudnn.benchmark = False
# TF32 연산 활성화 (RTX 30xx 이상) - FP32 대비 최대 2배 빠른 matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from typing import Dict, Any, AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from qwen_tts import Qwen3TTSModel
import soundfile as sf # For header generation

# ------------------------------------------------------------------
# [Lifespan] 서버 시작과 종료 시 실행될 로직 (최신 FastAPI 표준)
# ------------------------------------------------------------------
model = None

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
            dtype=dtype_policy,
            attn_implementation=attn_impl,
            trust_remote_code=True
        )

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
                        non_streaming_mode=False,
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

# ------------------------------------------------------------------
# [API 엔드포인트] 음성 생성 (스트리밍 지원)
# ------------------------------------------------------------------
@app.post("/tts/generate")
async def generate_speech(request: TTSRequest):
    if not model:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    print(f"🗣️ [Request] 텍스트 처리 중 (스트리밍): {request.text[:30]}...")

    # ------------------------------------------------------------------
    # 스트리밍 로직: Forward Hook를 사용하여 생성된 토큰을 실시간 캡처
    # ------------------------------------------------------------------
    token_queue = Queue()
    stop_event = Event()
    
    class StreamerAbort(Exception):
        pass

    def capture_tokens_hook(module, input, output):
        if stop_event.is_set():
            raise StreamerAbort("Client disconnected")
            
        # Qwen3TTSTalkerOutputWithPast의 hidden_states[1]에 현재 스텝의 codec_ids가 포함됨
        if output.hidden_states is not None and len(output.hidden_states) >= 2:
            try:
                # codec_ids: [Batch, Length, Codebooks]
                # Prefill 단계에서는 None일 수 있으므로 체크
                if output.hidden_states[1] is not None:
                    codec_ids = output.hidden_states[1]
                    # CPU로 이동하여 큐에 즉시 삽입 (메인 스레드에서 디코딩)
                    token_queue.put(codec_ids.detach().cpu())
            except Exception as e:
                print(f"⚠️ [Hook Error] 토큰 캡처 실패: {e}")

    def audio_generator():
        # 1. Hook 등록 (Talker 모델의 Forward Pass 감시)
        # model.model -> Qwen3TTSForConditionalGeneration
        # model.model.talker -> Qwen3TTSTalkerForConditionalGeneration
        hook_handle = model.model.talker.register_forward_hook(capture_tokens_hook)
        
        # 2. 백그라운드 스레드에서 생성 시작
        def run_generation():
            try:
                # [Optimization] inference_mode 사용
                with torch.inference_mode():
                    model.generate_custom_voice(
                        text=request.text,
                        language=request.language,
                        speaker=request.speaker,
                        non_streaming_mode=False, 
                        max_new_tokens=1024,
                        temperature=0.7,
                        top_k=30,
                        repetition_penalty=1.1
                    )
            except StreamerAbort:
                pass # 정상적인 중단
            except Exception as e:
                print(f"❌ [Gen Thread Error] 생성 중 오류: {e}")
            finally:
                token_queue.put(None) # 종료 신호
                stop_event.set()

        gen_thread = Thread(target=run_generation)
        gen_thread.start()

        # 3. 오디오 디코딩 및 전송
        # 초기 WAV 헤더 전송 (24000Hz, 16bit Mono)
        # 파일 크기를 알 수 없으므로 헤더만 전송하고 이후 raw PCM 데이터 전송
        dummy_buffer = io.BytesIO()
        sf.write(dummy_buffer, np.array([], dtype=np.int16), 24000, format='WAV', subtype='PCM_16')
        header_bytes = dummy_buffer.getvalue()
        yield header_bytes

        accumulated_tokens = []
        BATCH_SIZE = 5 # 5 프레임마다 디코딩 (약 200ms 지연)
        start_time = time.time()
        first_byte_sent = False

        try:
            while True:
                try:
                    # 타임아웃을 두어 교착 상태 방지
                    token = token_queue.get(timeout=1.0)
                except Empty:
                    if not gen_thread.is_alive():
                        break # 스레드가 죽었고 큐가 비었으면 종료
                    continue

                if token is None:
                    break
                
                # [Fix-3] 토큰 차원 확인 및 조정
                # token shape: [Batch, Codes] (2D) -> [Batch, 1, Codes] (3D)로 변환 필요
                # 그래야 cat(dim=1) 했을 때 [Batch, Time, Codes]가 됨
                if token.dim() == 2:
                    token = token.unsqueeze(1)
                
                accumulated_tokens.append(token)

                if len(accumulated_tokens) >= BATCH_SIZE:
                    # [Batch, Time, Code] 형태로 결합
                    codes = torch.cat(accumulated_tokens, dim=1).to(model.device)
                    accumulated_tokens = [] # 버퍼 비우기

                    # 오디오 디코딩
                    # speech_tokenizer는 model.model 안에 있음
                    wavs, sr = model.model.speech_tokenizer.decode({"audio_codes": codes})
                    
                    if len(wavs) > 0:
                        audio_chunk = wavs[0]
                        # float32 -> int16 PCM 변환
                        audio_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                        yield audio_int16.tobytes()

                        if not first_byte_sent:
                            latency = (time.time() - start_time) * 1000
                            print(f"⚡ [Stream] 첫 오디오 청크 전송 ({latency:.1f}ms)")
                            first_byte_sent = True

            # 남은 토큰 처리
            if accumulated_tokens:
                try:
                    codes = torch.cat(accumulated_tokens, dim=1).to(model.device)
                    wavs, sr = model.model.speech_tokenizer.decode({"audio_codes": codes})
                    if len(wavs) > 0:
                         audio_int16 = (np.clip(wavs[0], -1.0, 1.0) * 32767).astype(np.int16)
                         yield audio_int16.tobytes()
                except Exception:
                    pass

        except Exception as e:
            print(f"❌ [Stream Error] 스트리밍 루프 중 오류: {e}")
        finally:
            stop_event.set() # 스레드 종료 신호
            hook_handle.remove() # 훅 제거
            # gen_thread.join() # 대기하지 않음 (응답 즉시 종료)

    return StreamingResponse(audio_generator(), media_type="audio/wav")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18003)