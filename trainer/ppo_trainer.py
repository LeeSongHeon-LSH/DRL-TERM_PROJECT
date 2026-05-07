import torch
from .base import BaseTrainer
from models.policy import PolicyModel
from reward.base import BaseReward
from utils.logging import get_logger

logger = get_logger(__name__)


class PPOTrainer(BaseTrainer):
    """
    PPO Trainer for LLM fine-tuning.

    Uses Judge LLM probability distribution as reward signal.
    Reference policy (frozen copy of initial policy) is used for KL penalty.

    TBD: Algorithm may change — interface kept generic via BaseTrainer.
    """

    def __init__(self, policy: PolicyModel, reward_fn: BaseReward, config: dict):
        super().__init__(policy, reward_fn, config)
        ppo_cfg = config.get("trainer", {}).get("ppo", {})
        self.clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
        self.vf_coef = ppo_cfg.get("vf_coef", 0.1)
        self.kl_coef = ppo_cfg.get("kl_coef", 0.1)
        self.ppo_epochs = ppo_cfg.get("ppo_epochs", 4)
        self.grad_clip = ppo_cfg.get("grad_clip", 1.0)

        self.optimizer = None       # TODO: init optimizer after policy.load()
        self.ref_policy = None      # TODO: frozen reference policy for KL term

    def _init_optimizer(self) -> None:
        lr = self.config.get("trainer", {}).get("learning_rate", 1e-5)
        self.optimizer = torch.optim.AdamW(self.policy.model.parameters(), lr=lr)

    def rollout(self, prompts: list[str]) -> dict:
        """
        Generate responses and compute rewards.
        Returns dict with: prompts, responses, rewards, log_probs (old)
        """
        # TODO:
        # 1. policy.generate(prompts) → responses
        # 2. policy.get_token_log_probs(prompts, responses) → old_log_probs
        # 3. reward_fn.compute(prompts, responses) → rewards
        raise NotImplementedError

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        TBD: Compute GAE or simple reward-baseline advantages.
        """
        raise NotImplementedError

    def ppo_loss(
        self,
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped PPO surrogate loss."""
        ratio = torch.exp(log_probs_new - log_probs_old)
        clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        return -torch.min(ratio * advantages, clipped * advantages).mean()

    def kl_penalty(self, log_probs_policy: torch.Tensor, log_probs_ref: torch.Tensor) -> torch.Tensor:
        """KL divergence penalty to keep policy close to reference."""
        return (log_probs_policy - log_probs_ref).mean()

    def train_step(self, batch: dict) -> dict:
        # TODO:
        # 1. rollout(batch["prompts"])
        # 2. compute advantages
        # 3. ppo_epochs inner loop: ppo_loss + kl_penalty + vf_loss
        # 4. optimizer step
        raise NotImplementedError

    def train(self, dataloader) -> None:
        # TODO: full training loop with logging and eval
        raise NotImplementedError

    def save(self, path: str) -> None:
        # TODO: save policy model + optimizer state
        raise NotImplementedError

    def load(self, path: str) -> None:
        # TODO: load policy model + optimizer state
        raise NotImplementedError
