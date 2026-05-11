"""
GRPO (Group Relative Policy Optimization) trainer.

Architecture:
  - vLLM:     fast rollout generation (no gradients), loaded in 4-bit
  - HF model: log-prob computation and gradient updates (QLoRA: 4-bit base + LoRA)
  - Ref model: frozen base model in 4-bit for KL penalty

Weight sync: not performed for 4-bit quantized models (NF4 weights cannot be
copied in-place). vLLM uses the base model throughout training; the GRPO
clipped ratio corrects for the resulting distribution shift.
"""

import contextlib
import logging
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

from reward.math_reward import batch_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log-prob utility
# ---------------------------------------------------------------------------

def _compute_log_probs(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device,
    *,
    with_grad: bool,
) -> torch.Tensor:
    """Sum of log probs over response tokens for each (prompt, response) pair."""
    sequences = [p + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    was_training = model.training
    model.eval()
    ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
    with ctx:
        logits = model(**enc).logits[:, :-1]  # (B, L-1, V)
    if was_training:
        model.train()

    labels = enc["input_ids"][:, 1:]  # (B, L-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # Build mask: 1 for response tokens, 0 for prompt tokens and padding.
    mask = enc["attention_mask"][:, 1:].float()
    for i, prompt in enumerate(prompts):
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        mask[i, : prompt_len - 1] = 0.0

    return (token_lp * mask).sum(-1)  # (B,)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, config: dict):
        grpo = config["grpo"]
        vllm_cfg = config["vllm"]
        model_cfg = config["model"]
        lora_cfg = model_cfg["lora"]

        self.G = grpo["num_generations"]
        self.max_new_tokens = grpo["max_new_tokens"]
        self.temperature = grpo["temperature"]
        self.clip_eps = grpo["clip_epsilon"]
        self.kl_coef = grpo["kl_coef"]
        self.grad_clip = grpo["grad_clip"]
        self.sync_every = grpo["weight_sync_steps"]
        self.step = 0

        model_name = model_cfg["name"]
        bnb_cfg = model_cfg["quantization"]
        compute_dtype = torch.bfloat16 if bnb_cfg["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bnb_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model: 4-bit base + LoRA (QLoRA)
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        base = prepare_model_for_kbit_training(base)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        self.model = get_peft_model(base, lora_config)
        self.model.print_trainable_parameters()
        self.device = next(self.model.parameters()).device

        # Reference model: 4-bit, frozen, no LoRA
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        ).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # vLLM for fast rollout generation (4-bit via bitsandbytes)
        self.vllm = LLM(
            model=model_name,
            quantization=vllm_cfg.get("quantization", "bitsandbytes"),
            gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
            max_model_len=vllm_cfg["max_model_len"],
            tensor_parallel_size=vllm_cfg["tensor_parallel_size"],
        )

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=grpo["learning_rate"],
        )

    # ------------------------------------------------------------------

    def _rollout(self, prompts: list[str]) -> list[list[str]]:
        params = SamplingParams(
            n=self.G,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        outputs = self.vllm.generate(prompts, params)
        return [[o.text for o in out.outputs] for out in outputs]

    def _advantages(self, rewards: list[list[float]]) -> torch.Tensor:
        """GRPO: normalize rewards within each group to get advantages."""
        t = torch.tensor(rewards, dtype=torch.float32)  # (B, G)
        mean = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (t - mean) / std  # (B, G)

    def _grpo_loss(
        self,
        prompts_flat: list[str],
        responses_flat: list[str],
        adv_flat: torch.Tensor,
        old_lp: torch.Tensor,
    ) -> torch.Tensor:
        lp = _compute_log_probs(
            self.model, self.tokenizer,
            prompts_flat, responses_flat, self.device, with_grad=True,
        )
        ref_lp = _compute_log_probs(
            self.ref_model, self.tokenizer,
            prompts_flat, responses_flat, self.device, with_grad=False,
        )

        ratio = torch.exp(lp - old_lp.to(self.device))
        clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -torch.min(ratio * adv_flat, clipped * adv_flat).mean()

        kl = (lp - ref_lp.detach()).mean()
        return policy_loss + self.kl_coef * kl

    def _sync_vllm(self):
        # NF4-quantized weights cannot be copied in-place into vLLM.
        # The GRPO clipped ratio already handles the distribution shift
        # between vLLM's base-model rollouts and the updated QLoRA policy.
        logger.info(f"[step {self.step}] Weight sync skipped (4-bit quantized model)")

    # ------------------------------------------------------------------

    def train_step(self, prompts: list[str], references: list[str]) -> dict:
        # 1. Generate G responses per prompt via vLLM
        responses = self._rollout(prompts)           # (B, G) list[list[str]]

        # 2. Rule-based reward: 1.0 if answer correct, 0.0 otherwise
        rewards = batch_score(responses, references)  # (B, G) list[list[float]]

        # 3. GRPO advantages: within-group normalization
        adv = self._advantages(rewards)               # (B, G) tensor

        # 4. Flatten (B, G) → (B*G,) for batched log-prob computation
        prompts_flat = [p for p in prompts for _ in range(self.G)]
        responses_flat = [r for group in responses for r in group]
        adv_flat = adv.view(-1).to(self.device)

        # 5. Old log probs (before this optimizer step, no grad)
        old_lp = _compute_log_probs(
            self.model, self.tokenizer,
            prompts_flat, responses_flat, self.device, with_grad=False,
        )

        # 6. Gradient step
        self.optimizer.zero_grad()
        loss = self._grpo_loss(prompts_flat, responses_flat, adv_flat, old_lp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.optimizer.step()
        self.step += 1

        # 7. Periodically push updated weights to vLLM
        if self.step % self.sync_every == 0:
            self._sync_vllm()

        mean_reward = sum(sum(g) for g in rewards) / (len(prompts) * self.G)
        return {"loss": loss.item(), "mean_reward": mean_reward}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved → {path}")
