from dataclasses import dataclass, field
from typing import List, Optional
import argparse


@dataclass
class EvalConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"          # bfloat16 is optimal for Blackwell (RTX 50xx)

    # Generation — greedy for deterministic eval; set temperature>0 + do_sample=True for sampling
    max_new_tokens: int = 2048
    temperature: float = 0.0
    do_sample: bool = False

    # Evaluation
    years: List[int] = field(default_factory=lambda: [2023, 2024, 2025])
    batch_size: int = 4              # conservative for 8-12 GB VRAM; increase if you have headroom

    # Wandb
    wandb_project: str = "aime-qwen-eval"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Misc
    output_dir: str = "results"
    seed: int = 42


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="AIME evaluation with Qwen2.5-1.5B")
    parser.add_argument("--model-name",     default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dtype",          default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int,   default=2048)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--do-sample",      action="store_true")
    parser.add_argument("--years",          type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--batch-size",     type=int, default=4)
    parser.add_argument("--wandb-project",  default="aime-qwen-eval")
    parser.add_argument("--wandb-entity",   default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--output-dir",     default="results")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    return EvalConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        years=args.years,
        batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )
