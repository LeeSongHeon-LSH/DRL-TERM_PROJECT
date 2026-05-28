"""
Entry point for olympiad evaluation.

Usage:
  python scripts/run_olympiad_eval.py
  python scripts/run_olympiad_eval.py --config config/olympiad_config.yaml
  python scripts/run_olympiad_eval.py --config config/olympiad_config.yaml --samples 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so local packages resolve
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run olympiad math evaluation with vLLM")
    parser.add_argument(
        "--config",
        default="config/olympiad_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override total_samples in config",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name in config",
    )
    args = parser.parse_args()

    # Optional runtime overrides (avoid mutating the file)
    if args.samples is not None or args.model is not None:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        if args.samples is not None:
            cfg["dataset"]["total_samples"] = args.samples
        if args.model is not None:
            cfg["model"]["name"] = args.model

        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        yaml.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name
        cleanup = True
    else:
        config_path = args.config
        cleanup = False

    try:
        from eval.olympiad_eval import run_evaluation
        run_evaluation(config_path)
    finally:
        if cleanup:
            os.unlink(config_path)


if __name__ == "__main__":
    main()
