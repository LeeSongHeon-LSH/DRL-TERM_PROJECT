"""정책 π의 multi-step rollout — vLLM 백엔드 (off-policy 평가/BoN용).

GRPO 학습 시에는 TRL이 자체 vLLM colocate를 관리하므로 이 모듈은 평가/BoN 전용.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .parser import split_steps


@dataclass
class VLLMRolloutConfig:
    model_id: str
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    stop: tuple[str, ...] = ("\n\n\n",)
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.55
    lora_path: str | None = None
    dtype: str = "bfloat16"


@dataclass
class Trajectory:
    problem: str
    full_text: str
    steps: list[str] = field(default_factory=list)


class VLLMRollout:
    """π에서 group_size 개 trajectory를 generate."""

    def __init__(self, cfg: VLLMRolloutConfig):
        self.cfg = cfg
        self._llm = None
        self._lora = None

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM

        kwargs = dict(
            model=self.cfg.model_id,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            dtype=self.cfg.dtype,
        )
        if self.cfg.lora_path is not None:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = 64

        self._llm = LLM(**kwargs)
        if self.cfg.lora_path is not None:
            from vllm.lora.request import LoRARequest

            self._lora = LoRARequest("pav-lora", 1, self.cfg.lora_path)

    @staticmethod
    def _build_prompt(problem: str) -> str:
        system = (
            "You solve math problems using natural-language steps only.\n"
            "Rules:\n"
            '- Output exactly one reasoning step per line.\n'
            '- Start every line with "Step k:" (k = 1,2,3,...).\n'
            '- Each line must contain a SINGLE calculation or deduction. '
            "Never put two on one line.\n"
            '- Do NOT write any code or use Python.\n'
            '- Do NOT write an introduction or a summary sentence.\n'
            '- The last line must be "Answer: <number>".'
        )
        few_shot_user = "A box has 2 dozen pens. 5 are given away. How many remain?"
        few_shot_assistant = (
            "Step 1: One dozen is 12, so 2 dozen is 2 × 12 = 24 pens.\n"
            "Step 2: 5 pens are given away.\n"
            "Step 3: Remaining pens = 24 − 5 = 19.\n"
            "Answer: 19"
        )
        return (
            f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"<|im_start|>user\n{few_shot_user}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{few_shot_assistant}\n<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def rollout(
        self,
        problems: Sequence[str],
        group_size: int = 8,
    ) -> list[list[Trajectory]]:
        """문제별로 group_size 개의 trajectory 반환. 결과 [B][G]."""
        self._ensure_loaded()
        from vllm import SamplingParams

        prompts = [self._build_prompt(p) for p in problems]
        sp = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
            stop=list(self.cfg.stop),
            n=group_size,
        )
        gen_kwargs = {}
        if self._lora is not None:
            gen_kwargs["lora_request"] = self._lora
        outputs = self._llm.generate(prompts, sp, **gen_kwargs)

        groups: list[list[Trajectory]] = []
        for problem, output in zip(problems, outputs):
            traj_group: list[Trajectory] = []
            for completion in output.outputs:
                full = completion.text
                traj_group.append(
                    Trajectory(
                        problem=problem,
                        full_text=full,
                        steps=split_steps(full),
                    )
                )
            groups.append(traj_group)
        return groups
