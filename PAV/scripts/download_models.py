"""HuggingFace에서 필요한 모델 가중치 다운로드.

다운로드 대상:
  - Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B (~2.9GB) — PRM (경량)
  - Qwen/Qwen2.5-Math-7B-Instruct             (~14GB)  — π / μ base

(7B PRM이 필요하면 MODELS["prm"]을 -7B로 교체)

캐시 위치:
  - HF_HOME 환경변수 (default ~/.cache/huggingface)
  - --local-dir 지정 시 해당 폴더에 그대로 저장

인증:
  - HF_TOKEN 환경변수 또는 `huggingface-cli login` 으로 사전 인증.
  - 두 모델 모두 public이라 토큰 없이도 가능 (rate limit 회피용으로 권장).

사용 예:
  uv run python scripts/download_models.py                     # 전체 ~/.cache로
  uv run python scripts/download_models.py --only prm          # PRM만
  uv run python scripts/download_models.py --local-dir ./models
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("download")


MODELS = {
    "prm":    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    "policy": "Qwen/Qwen2.5-Math-7B-Instruct",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--only",
        choices=list(MODELS.keys()),
        default=None,
        help="한 모델만 다운로드 (default: 둘 다)",
    )
    ap.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="HF 캐시 대신 지정한 디렉토리에 저장 (각 모델은 model_id의 마지막 토큰으로 sub-dir)",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="병렬 파일 다운로드 worker 수 (기본 8)",
    )
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF 토큰 (미지정 시 HF_TOKEN env / cached login 사용)",
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="다운로드 없이 대상 모델만 출력하고 종료",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    targets = MODELS if args.only is None else {args.only: MODELS[args.only]}

    if args.list_only:
        log.info("다운로드 대상:")
        for k, v in targets.items():
            log.info(f"  - {k:<8s}  {v}")
        return

    # huggingface_hub은 base deps에 포함됨 (pyproject)
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    token = args.token or os.environ.get("HF_TOKEN") or None

    for key, repo_id in targets.items():
        local_dir = None
        if args.local_dir:
            local_dir = str(Path(args.local_dir) / repo_id.split("/")[-1])
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        log.info(f"\n[{key}] downloading {repo_id} …")
        try:
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=token,
                max_workers=args.max_workers,
                # safetensors / pytorch_model 우선, doc/이미지 등 부산물 제외
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "tokenizer*",
                    "*.py",
                ],
            )
        except HfHubHTTPError as e:
            log.error(f"  ERROR: {e}")
            sys.exit(1)
        log.info(f"  -> {path}")

    log.info("\n완료. configs/prm.yaml 의 model_id 와 매칭되는 위치에 가중치가 저장되었습니다.")
    if args.local_dir:
        log.info(
            "  (local-dir 사용 시 configs/*.yaml 의 model_id 를 로컬 경로로 바꿔야 합니다.)"
        )


if __name__ == "__main__":
    main()
