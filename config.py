from dataclasses import dataclass, field
from typing import List, Optional
import argparse


@dataclass
class EvalConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7        # >0 required for pass@k sampling
    do_sample: bool = True

    # pass@k
    num_samples: int = 256          # samples generated per problem
    sample_batch_size: int = 16     # num_return_sequences per model.generate() call
    pass_k_values: List[int] = field(default_factory=lambda: [1, 8, 64, 256])

    # Evaluation
    years: List[int] = field(default_factory=lambda: [2023, 2024, 2025])

    # Wandb
    wandb_project: str = "aime-qwen-eval"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None   # base name; year/comparison suffix is appended

    # Misc
    output_dir: str = "results"
    seed: int = 42


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="AIME pass@k evaluation with Qwen2.5")
    parser.add_argument("--model-name",        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dtype",             default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens",    type=int,   default=2048)
    parser.add_argument("--temperature",       type=float, default=0.7)
    parser.add_argument("--no-sample",         action="store_true",
                        help="Greedy decoding — forces num_samples=1")
    parser.add_argument("--num-samples",       type=int,   default=256,
                        help="Number of samples per problem (pass@k denominator)")
    parser.add_argument("--sample-batch-size", type=int,   default=16,
                        help="num_return_sequences per model.generate() call")
    parser.add_argument("--pass-k-values",     type=int,   nargs="+",
                        default=[1, 8, 64, 256],
                        help="Which pass@k values to report (filtered to ≤ num_samples)")
    parser.add_argument("--years",             type=int,   nargs="+",
                        default=[2023, 2024, 2025])
    parser.add_argument("--wandb-project",     default="aime-qwen-eval")
    parser.add_argument("--wandb-entity",      default=None)
    parser.add_argument("--wandb-run-name",    default=None,
                        help="Base run name; '-2023', '-2024', etc. are appended automatically")
    parser.add_argument("--output-dir",        default="results")
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    do_sample = not args.no_sample
    num_samples = 1 if args.no_sample else args.num_samples

    return EvalConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=do_sample,
        num_samples=num_samples,
        sample_batch_size=args.sample_batch_size,
        pass_k_values=args.pass_k_values,
        years=args.years,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )
