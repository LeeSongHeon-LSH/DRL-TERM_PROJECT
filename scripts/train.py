"""
Entry point for RL training.
Usage: python scripts/train.py --config config/base_config.yaml
"""
import argparse
import yaml

from models.policy import PolicyModel
from models.reward_model import RewardModel
from reward.llm_judge import LLMJudgeReward
from trainer.ppo_trainer import PPOTrainer
from data.dataset import BaseDataset
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Loading policy model...")
    policy = PolicyModel()
    policy.load(config["model"]["policy"]["name"])

    logger.info("Loading judge (reward) model...")
    judge = RewardModel()
    judge.load(config["model"]["judge"]["name"])

    reward_fn = LLMJudgeReward(judge=judge)
    # TODO: set reward_fn.positive_token_ids once judge model and task are confirmed

    # TODO: load dataset via BaseDataset subclass
    # dataset = ConcreteDataset.from_config(config["data"])
    # dataloader = DataLoader(dataset, batch_size=config["data"]["batch_size"])

    trainer = PPOTrainer(policy=policy, reward_fn=reward_fn, config=config)
    trainer._init_optimizer()

    logger.info("Starting training...")
    # trainer.train(dataloader)


if __name__ == "__main__":
    main()
