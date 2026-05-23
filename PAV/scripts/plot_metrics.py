"""학습 metrics jsonl → PNG 그래프.

사용:
    uv run python scripts/plot_metrics.py
    uv run python scripts/plot_metrics.py --jsonl outputs/stage8_smoke/metrics.jsonl
    uv run python scripts/plot_metrics.py --out outputs/stage8_smoke/plots/

생성 그래프:
    reward.png         — reward / reward_std (시간 따라)
    learning_rate.png  — LR cosine schedule
    kl.png             — KL divergence
    grad_norm.png      — gradient L2 norm
    completion.png     — completion length
    overview.png       — 모든 metric 2x3 subplot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _key(records, key):
    xs = [r["step"] for r in records if key in r]
    ys = [r[key] for r in records if key in r]
    return xs, ys


def plot_single(records, key, ylabel, title, out_path: Path, color: str = "tab:blue"):
    xs, ys = _key(records, key)
    if not xs:
        print(f"  skip {key} (no data)")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_overview(records, out_path: Path):
    """6 subplot: reward, reward_std, kl, learning_rate, grad_norm, completion_length"""
    panels = [
        ("reward", "reward", "Reward (PAV)", "tab:blue"),
        ("reward_std", "reward_std", "Reward std (group variance)", "tab:cyan"),
        ("kl", "kl", "KL(π || π_ref)", "tab:red"),
        ("learning_rate", "lr", "Learning rate", "tab:green"),
        ("grad_norm", "grad_norm", "Gradient L2 norm", "tab:orange"),
        ("completion_length", "tokens", "Completion length", "tab:purple"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, ylabel, title, color) in zip(axes.flat, panels):
        xs, ys = _key(records, key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color=color)
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"(no '{key}' data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    fig.suptitle(f"PAV-RL training metrics ({len(records)} log entries)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", default="outputs/stage8_smoke/metrics.jsonl",
                    help="metrics.jsonl 경로")
    ap.add_argument("--out", default=None,
                    help="플롯 저장 디렉토리 (default: jsonl과 같은 폴더의 plots/)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"metrics jsonl 없음: {jsonl_path}\n"
                                "학습 한 번 돌아야 생성됨 (callbacks.py:JsonlMetricsCallback).")

    out_dir = Path(args.out) if args.out else jsonl_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(jsonl_path)
    print(f"loaded {len(records)} records from {jsonl_path}")
    print(f"saving plots → {out_dir}/")

    plot_single(records, "reward",            "reward",   "Reward (PAV)",            out_dir / "reward.png",         "tab:blue")
    plot_single(records, "reward_std",        "std",      "Reward std",              out_dir / "reward_std.png",     "tab:cyan")
    plot_single(records, "learning_rate",     "lr",       "Learning rate",           out_dir / "learning_rate.png",  "tab:green")
    plot_single(records, "kl",                "kl",       "KL(π || π_ref)",          out_dir / "kl.png",             "tab:red")
    plot_single(records, "grad_norm",         "norm",     "Gradient L2 norm",        out_dir / "grad_norm.png",      "tab:orange")
    plot_single(records, "completion_length", "tokens",   "Completion length",       out_dir / "completion.png",     "tab:purple")
    plot_overview(records, out_dir / "overview.png")
    print("done.")


if __name__ == "__main__":
    main()
