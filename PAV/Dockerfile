# 단일 trainer 컨테이너 이미지 (PRM / μ / 정책 모두 로컬 GPU 적재).
# 진짜 GPU에서 돌리려면 호스트에 NVIDIA Container Toolkit 설치 필요:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#
# 이미지 빌드:
#   docker compose build                    # 기본 (Ampere/Ada — 3090, 4090 등)
#   Blackwell은 compose 파일에서 BASE_IMAGE/TORCH_INDEX ARG로 자동 분기
#
# 가중치는 컨테이너 안에 굽지 않고, hf_cache 볼륨으로 호스트와 공유.
# 워커가 처음 시작될 때 PRM/정책을 자동으로 다운로드 (lazy load).

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

FROM ${BASE_IMAGE}

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
# torch는 호스트의 cu124 인덱스에서 받음. base 이미지의 torch는 덮어쓰일 수 있어 --reinstall로 강제 동기화
# --extra serve: 추론 PC의 PRM FastAPI 서버(scripts/serve_prm_http.py)를 위해 fastapi/uvicorn 포함
# Blackwell: cu128 인덱스를 추가로 지정 (PyPI는 기본 유지, torch만 cu128에서 설치)
ARG TORCH_INDEX=
RUN if [ -n "$TORCH_INDEX" ]; then \
      uv sync --extra gpu --extra serve --no-dev --link-mode=copy \
        --extra-index-url "$TORCH_INDEX"; \
    else \
      uv sync --extra gpu --extra serve --no-dev --link-mode=copy; \
    fi

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
