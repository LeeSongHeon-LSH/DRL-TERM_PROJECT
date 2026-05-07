from models.policy import PolicyModel
from data.dataset import BaseDataset
from eval.base import BaseBenchmark
from utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Runs a benchmark over the policy model and reports metrics."""

    def __init__(self, policy: PolicyModel, benchmark: BaseBenchmark):
        self.policy = policy
        self.benchmark = benchmark

    def run(self, dataset: BaseDataset, batch_size: int = 8) -> dict:
        """
        Iterate over dataset, generate responses, evaluate with benchmark.
        Returns: dict of metric_name → score
        """
        # TODO:
        # 1. DataLoader over dataset
        # 2. policy.generate(batch["prompts"]) → responses
        # 3. benchmark.evaluate(responses, batch["references"]) → metrics
        # 4. aggregate and return
        raise NotImplementedError

    def log_metrics(self, metrics: dict, step: int) -> None:
        for k, v in metrics.items():
            logger.info(f"[step {step}] {k}: {v:.4f}")
