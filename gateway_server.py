import asyncio
import http.client
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


@dataclass(frozen=True)
class Backend:
    name: str
    url: str


def _load_backends() -> list[Backend]:
    raw = os.getenv(
        "QWEN_TTS_BACKENDS",
        "http://127.0.0.1:18011,http://127.0.0.1:18012,"
        "http://127.0.0.1:18013,http://127.0.0.1:18014",
    )
    urls = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not urls:
        raise RuntimeError("QWEN_TTS_BACKENDS must contain at least one backend URL")
    return [Backend(name=f"backend-{index + 1}", url=url) for index, url in enumerate(urls)]


BACKENDS = _load_backends()
QUEUE_TIMEOUT_SECONDS = float(os.getenv("QWEN_TTS_GATEWAY_QUEUE_TIMEOUT", "5"))
CONNECT_TIMEOUT_SECONDS = float(os.getenv("QWEN_TTS_GATEWAY_CONNECT_TIMEOUT", "10"))
READ_CHUNK_SIZE = int(os.getenv("QWEN_TTS_GATEWAY_CHUNK_SIZE", "8192"))

backend_queue: asyncio.Queue[Backend] = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for backend in BACKENDS:
        backend_queue.put_nowait(backend)
    yield


app = FastAPI(title="Qwen3-TTS Gateway", lifespan=lifespan)


def _open_backend_stream(backend: Backend, path: str, body: bytes) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    parsed = urlparse(backend.url)
    conn = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=CONNECT_TIMEOUT_SECONDS,
    )
    conn.request(
        "POST",
        path,
        body=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        },
    )
    response = conn.getresponse()
    return conn, response


async def _proxy_tts(path: str, request: Request) -> StreamingResponse:
    body = await request.body()
    try:
        backend = await asyncio.wait_for(backend_queue.get(), timeout=QUEUE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="All TTS backends are busy. Retry shortly.")

    try:
        conn, response = await asyncio.to_thread(_open_backend_stream, backend, path, body)
    except Exception as exc:
        backend_queue.put_nowait(backend)
        raise HTTPException(status_code=502, detail=f"{backend.name} connection failed: {exc}") from exc

    if response.status != 200:
        error_body = await asyncio.to_thread(response.read)
        conn.close()
        backend_queue.put_nowait(backend)
        detail = error_body.decode("utf-8", errors="replace")[:500] or response.reason
        raise HTTPException(status_code=response.status, detail=detail)

    def stream() -> Iterator[bytes]:
        try:
            while True:
                chunk = response.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            conn.close()
            backend_queue.put_nowait(backend)

    return StreamingResponse(stream(), media_type="audio/wav")


@app.post("/tts/generate")
async def generate(request: Request) -> StreamingResponse:
    return await _proxy_tts("/tts/generate", request)


@app.post("/tts/generate/en")
async def generate_en(request: Request) -> StreamingResponse:
    return await _proxy_tts("/tts/generate/en", request)


def _check_backend(backend: Backend) -> dict:
    parsed = urlparse(backend.url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=2)
    try:
        conn.request("GET", "/health")
        response = conn.getresponse()
        body = response.read()
        parsed_body = json.loads(body.decode("utf-8")) if body else {}
        return {
            "name": backend.name,
            "url": backend.url,
            "status_code": response.status,
            "healthy": response.status == 200,
            "detail": parsed_body,
        }
    except Exception as exc:
        return {
            "name": backend.name,
            "url": backend.url,
            "status_code": None,
            "healthy": False,
            "error": str(exc),
        }
    finally:
        conn.close()


@app.get("/health")
async def health() -> dict:
    checks = await asyncio.gather(
        *(asyncio.to_thread(_check_backend, backend) for backend in BACKENDS)
    )
    healthy_count = sum(1 for check in checks if check["healthy"])
    return {
        "status": "healthy" if healthy_count else "unhealthy",
        "available_backends": backend_queue.qsize(),
        "healthy_backends": healthy_count,
        "total_backends": len(BACKENDS),
        "backends": checks,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("QWEN_TTS_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("QWEN_TTS_GATEWAY_PORT", "18003"))
    uvicorn.run("gateway_server:app", host=host, port=port, workers=1)
