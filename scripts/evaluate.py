"""
Entry point for benchmark evaluation.
Usage: python scripts/evaluate.py --config config/base_config.yaml --checkpoint outputs/step_500
"""
import argparse
import yaml

from models.policy import PolicyModel
from eval.evaluator import Evaluator
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    policy = PolicyModel()
    policy.load(config["model"]["policy"]["name"])

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        # TODO: policy.load_checkpoint(args.checkpoint)

    # TODO: instantiate concrete benchmark once confirmed
    # benchmark = ConcreteBenchmark()
    # benchmark.load()
    # dataset = ConcreteDataset.from_config(config["data"])

    # evaluator = Evaluator(policy=policy, benchmark=benchmark)
    # metrics = evaluator.run(dataset, batch_size=config["data"]["batch_size"])
    # evaluator.log_metrics(metrics, step=0)


if __name__ == "__main__":
    main()
