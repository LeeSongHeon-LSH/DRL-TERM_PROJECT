import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.math_eval import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/final"
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results = evaluate(model_path, cfg)
    print(f"\nOverall accuracy: {results['overall_accuracy']:.4f}")
    print("By level:")
    for lvl, acc in sorted(results["by_level"].items()):
        print(f"  {lvl}: {acc:.4f}")


if __name__ == "__main__":
    main()
