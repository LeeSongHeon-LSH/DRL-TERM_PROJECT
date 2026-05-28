# syntax=docker/dockerfile:1
# 단일 trainer 컨테이너 이미지 (PRM / μ / 정책 모두 로컬 GPU 적재).
# BuildKit cache mount로 uv 다운로드 캐시 재사용.
# 진짜 GPU에서 돌리려면 호스트에 NVIDIA Container Toolkit 설치 필요:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#
# 이미지 빌드:
#   docker compose build                    # 기본 (Ampere/Ada — 3090, 4090 등)
#
# 가중치는 컨테이너 안에 굽지 않고, hf_cache 볼륨으로 호스트와 공유.
# 워커가 처음 시작될 때 PRM/정책을 자동으로 다운로드 (lazy load).

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 시스템 deps (vLLM build 의존성, git, ca-certs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 (단일 binary)
ENV UV_INSTALL_DIR=/usr/local/bin
ADD https://astral.sh/uv/0.11.2/install.sh /tmp/install-uv.sh
RUN sh /tmp/install-uv.sh && rm /tmp/install-uv.sh

WORKDIR /app

# 1) deps 먼저 (캐시 효과)
COPY pyproject.toml ./
COPY uv.lock* ./

# BuildKit 캐시 마운트: uv 다운로드 캐시(/root/.cache/uv)를 재사용하여 네트워크 다운로드 최소화
# torch는 호스트의 cu124 인덱스에서 받음. base 이미지의 torch는 덮어쓰일 수 있어 --reinstall로 강제 동기화
# --extra serve: 추론 PC의 PRM FastAPI 서버(scripts/serve_prm_http.py)를 위해 fastapi/uvicorn 포함
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra gpu --extra serve --no-dev --link-mode=copy

# 2) 소스
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# uv venv를 PATH에 노출
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app
ENV HF_HOME=/cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 모든 service 공통 default — 각 service가 command override
CMD ["python", "-c", "print('Specify a service in docker-compose.yml')"]

# 모든 service 공통 default — 각 service가 command override
CMD ["python", "-c", "print('Specify a service in docker-compose.yml')"]
