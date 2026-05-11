import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.gsm8k import GSM8KDataset, make_dataloader
from trainer.grpo_trainer import GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    trainer = GRPOTrainer(cfg)
    dataset = GSM8KDataset(trainer.tokenizer, split="train")
    loader = make_dataloader(dataset, cfg["grpo"]["batch_size"])

    log_every = cfg["logging"]["log_steps"]
    save_every = cfg["logging"]["save_steps"]
    total_steps = cfg["grpo"]["total_steps"]
    output_dir = cfg["logging"]["output_dir"]

    logger.info(f"GSM8K train size: {len(dataset)} | total steps: {total_steps}")

    while trainer.step < total_steps:
        for batch in loader:
            if trainer.step >= total_steps:
                break
            metrics = trainer.train_step(
                list(batch["prompt"]), list(batch["reference"])
            )
            if trainer.step % log_every == 0:
                logger.info(
                    f"step {trainer.step}/{total_steps} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"reward={metrics['mean_reward']:.4f}"
                )
            if trainer.step % save_every == 0:
                trainer.save(os.path.join(output_dir, f"checkpoint-{trainer.step}"))

    trainer.save(os.path.join(output_dir, "final"))
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
