import torch
from .base import BaseLLM


class PolicyModel(BaseLLM):
    """
    Policy LLM — the model being trained via RL.
    TBD: Concrete model (e.g. LLaMA, Mistral) to be plugged in here.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._device = None

    def load(self, model_name: str, dtype: torch.dtype = torch.bfloat16, **kwargs) -> None:
        # TODO: load HuggingFace model + tokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self._device)
        raise NotImplementedError(f"PolicyModel.load() — model '{model_name}' not configured yet.")

    def generate(self, prompts: list[str], max_new_tokens: int = 256, **kwargs) -> list[str]:
        # TODO: batched generation
        raise NotImplementedError

    def get_log_probs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        # TODO: compute sum of token log probs for each (prompt, response) pair
        raise NotImplementedError

    def get_token_log_probs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        # TODO: return per-token log probs for PPO value/advantage computation
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self._device
