import torch
import torch.nn.functional as F
from .base import BaseReward
from models.reward_model import RewardModel


class LLMJudgeReward(BaseReward):
    """
    Computes reward from Judge LLM's output probability distribution.

    TBD: Define the exact reward extraction strategy once the judge model and
    target tokens (e.g. "Yes"/"No", score tokens, etc.) are confirmed.
    """

    def __init__(self, judge: RewardModel, positive_token_ids: list[int] | None = None):
        self.judge = judge
        # TBD: token IDs that represent a positive judgment (e.g. token for "Yes", "1", "Good")
        self.positive_token_ids = positive_token_ids

    def build_judge_prompt(self, prompt: str, response: str) -> str:
        """
        Format the evaluation prompt fed to the judge.
        TBD: Prompt template to be defined once judge model and task are confirmed.
        """
        # TODO: define judge prompt template
        raise NotImplementedError

    def compute(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """
        1. Build judge prompts from (prompt, response) pairs.
        2. Get logits from the judge over the vocabulary.
        3. Extract probability mass on positive_token_ids as scalar reward.
        Returns: Tensor of shape (batch,)
        """
        assert self.positive_token_ids is not None, "positive_token_ids must be set before compute()"

        judge_prompts = [
            self.build_judge_prompt(p, r) for p, r in zip(prompts, responses)
        ]

        # logits: (batch, vocab_size)
        logits = self.judge.get_reward_logits(judge_prompts)
        probs = F.softmax(logits, dim=-1)

        # Sum probability mass over all positive tokens → scalar reward per sample
        positive_ids = torch.tensor(self.positive_token_ids, device=logits.device)
        rewards = probs[:, positive_ids].sum(dim=-1)  # (batch,)
        return rewards
